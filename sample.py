#! python3

"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider
import numpy as np

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 42069
device = torch.device("mps") # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'float32' # 'bfloat16' or 'float16'
show_probs = False # Set to True to see chart of top 10 tokens each iteration
show_attention = False # Set to True to diagram of which tokens the attention mechanism is focusing on
compile = True # use PyTorch 2.0 to compile the model to be faster
fixed_response = "" # Use a fixed completion instead of sampling stochastically
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.mps.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'mps' 
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    # Also disable flash attention if we want to visualise it
    model = GPT.from_pretrained(init_from, dict(dropout=0.0, flash=(not show_attention)))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()

# Make newlines work
start = start.replace("\\n", "\n")
fixed_response = fixed_response.replace("\\n", "\n")

start_ids = encode(start)
fixed_response_ids = None
if len(fixed_response) > 0:
    fixed_response_ids = encode(fixed_response)

x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            if show_probs or show_attention:
                generator = model.generate_generator(x, max_new_tokens, temperature=temperature, top_k=top_k, fixed_response=fixed_response_ids)
                
                print("\n\nCompletetion including prompt:\n" + start, end="")
                for generation in generator:
                    _token_y = generation["iteration_token"]
                    token_y = _token_y[0].tolist()
                    token_prob = generation["iteration_token_probability"]
                    probs = generation["iteration_probability_dist"]
                    all_prev_tokens = generation["all_previous_tokens"]

                    # Console "streaming" output
                    selected_token = decode(token_y)
                    print(selected_token, end="", flush=True) # Append to console output

                    # Whole string so far
                    whole_prev_completion = [decode([t]) for t in all_prev_tokens[0].tolist()]
                    n_tokens = len(whole_prev_completion)
                    if show_attention:
                        # print(model.last_token_attention_weights[-1].shape)
                        n_blocks = len(model.last_token_attention_weights)
                        n_heads = len(model.last_token_attention_weights[0][0, :, -1, :])
                        # print("BLocks",n_blocks,"Heads",n_heads)

                        # Changed in the chart
                        chart_state = {
                            "block_mode": "Individual",
                            "head_mode": "Mean",
                            "block_n": n_blocks - 1,
                            "head_n": n_heads - 1
                        }

                        def get_head_attention_weights(block_i, head=None):
                            block = model.last_token_attention_weights[block_i]
                            last_token_attention = block[0, :, -1, :]
                            # Shape is now (num_heads, sequence_length)

                            # print(last_token_attention.shape)

                            if head is None:
                                total_weights = [0 for _ in range(n_tokens)]
                                for i in range(n_heads):
                                    for j, token_weight in enumerate(last_token_attention[i]):
                                        # Convert from tensor to float
                                        total_weights[j] += float(token_weight)
                                return [total / n_heads for total in total_weights]

                            return [float(token_weight) for token_weight in last_token_attention[head]]

                        def get_block_attention_weights(block=None, head=None):
                            # Update vars, whilst keeping it usable as an indepent func
                            if block is None:
                                if chart_state["block_mode"] == "Mean":
                                    pass
                                else:
                                    block = chart_state["block_n"]
                            
                            if head is None:
                                if chart_state["head_mode"] == "Mean":
                                    pass
                                else:
                                    head = chart_state["head_n"]
                                
                            if block is None:
                                total_weights = [0 for _ in range(n_tokens)]
                                for i in range(n_blocks):
                                    for j, token_avg_weight in enumerate(get_head_attention_weights(i, head)):
                                        total_weights[j] += token_avg_weight
                                return [total / n_blocks for total in total_weights]

                            return get_head_attention_weights(block, head)

                        # Fancy named grid area layout
                        fig, ax = plt.subplot_mosaic(
                            [
                                ['main', 'main', "main"],
                                ['block_radio', 'head_radio', "blank"],
                                ['block_slider', 'head_slider', "controls"],
                            ],
                            height_ratios=[7, 1, 1],
                            width_ratios=[2, 2, 1],
                            layout='constrained',
                        )
                        # Hide 'blank' sections
                        ax["blank"].axis("off")
                        main = ax["main"]
                        fig.set_figwidth(12) # This is set in inches for some reason lol
                        fig.set_figheight(8) 
                        main.set_ylabel("Average attention weight in all blocks")


                        def update_bar_chart(data):
                            main.clear()
                            main.bar(whole_prev_completion, get_block_attention_weights())
                       
                        def update_labels():
                            fig.suptitle('Attention weights')
                            main.set_title(f"'{selected_token}' was selected as the next token, from a probability of {token_prob*100:0.2f}%")

                            if chart_state["block_mode"] == "Mean":
                                if chart_state["head_mode"] == "Mean":
                                    main.set_ylabel("Average head attention weight across all blocks")
                                else:
                                    main.set_ylabel(f"Head {chart_state["head_n"]} attention weight across all blocks")
                            else:
                                block_name = "final block" if chart_state["block_n"] == n_blocks - 1 else f"block {chart_state["block_n"]}"
                                if chart_state["head_mode"] == "Mean":
                                    main.set_ylabel(f"Average head attention weight in {block_name}")
                                else:
                                    main.set_ylabel(f"Head {chart_state["head_n"]} attention weight in {block_name}")

                        # Radio buttons for average vs slider
                        ax['block_radio'].set_title("Block mode")
                        block_radio = RadioButtons(ax['block_radio'], ('Mean', 'Individual'), active=1)
                        def block_radio_fn(label):
                            chart_state["block_mode"] = label
                            update_bar_chart(get_block_attention_weights())
                            update_labels()
                            fig.canvas.draw()
                        block_radio.on_clicked(block_radio_fn)

                        ax['head_radio'].set_title("Head mode")
                        head_radio = RadioButtons(ax['head_radio'], ('Mean', 'Individual'))
                        def head_radio_fn(label):
                            chart_state["head_mode"] = label
                            update_bar_chart(get_block_attention_weights())
                            update_labels()
                            fig.canvas.draw_idle()
                        head_radio.on_clicked(head_radio_fn)

                        # Sliders for selecting head or block
                        head_slider = Slider(
                            ax["head_slider"],
                            label="Head to show weights for",
                            valinit=chart_state["head_n"],
                            valstep=[i for i in range(n_heads)],
                            valmin=0,
                            valmax=n_heads-1
                        )
                        block_slider = Slider(
                            ax["block_slider"],
                            label="Block to show weights for",
                            valinit=chart_state["block_n"],
                            valstep=[i for i in range(n_blocks)],
                            valmin=0,
                            valmax=n_blocks-1
                        )
                        def on_slider_change(val):
                            chart_state["block_n"] = block_slider.val
                            chart_state["head_n"] = head_slider.val
                            update_bar_chart(get_block_attention_weights())
                            update_labels()
                            fig.canvas.draw()

                        head_slider.on_changed(on_slider_change)
                        block_slider.on_changed(on_slider_change)



                        # Add button to close and continue
                        next_button = Button(ax["controls"], "Next token >>")
                        def clicked_callback(_event):
                            plt.close(fig)
                        next_button.on_clicked(clicked_callback)


                        # Actual plot 
                        update_bar_chart(get_block_attention_weights())
                        update_labels()

                        # Show plot, which pauses execution until it's closed
                        plt.show()


                    if show_probs:
                        # Took wayyy to long to figure out how to get the top 10
                        sorted_probs, indices_probs = torch.sort(probs, descending=True)
                        top_10_probs = sorted_probs[0].tolist()[:10]
                        top_10_indices = indices_probs[0].tolist()[:10]
                        top_10_tokens = [decode([t]) for t in top_10_indices]
                        
                        # Show on plot
                        colours = ["green" if t == selected_token else "blue" for t in top_10_tokens]
                        fig, ax = plt.subplots()
                        fig.set_figwidth(10) # This is set in inches for some reason lol
                        ax.set_ylabel("Probability")
                        ax.bar(top_10_tokens, top_10_probs, color=colours)
                        
                        # Add button to close and continue
                        button_axis = fig.add_axes([0.7, 0.8, 0.2, 0.075])
                        next_button = Button(button_axis, "Next token")
                        def clicked_callback(_event):
                            plt.close(fig)
                        next_button.on_clicked(clicked_callback)

                        fig.suptitle('Top 10 tokens and their probabilities')
                        ax.set_title(f"'{selected_token}' was selected as the next token, from a probability of {token_prob*100:0.2f}%")


                        # Show plot, which pauses execution until it's closed
                        plt.show()

                print("\nDone.\n")
            else:
                y, y_prob_cond_prod = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, fixed_response=fixed_response_ids)
                print(decode(y[0].tolist()))
                print('---------------')
                print(f"Prob: {str(y_prob_cond_prod)}")
