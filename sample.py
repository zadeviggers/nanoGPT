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
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Apply seaborn default theme
sns.set_theme()

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
dtype = 'float16' # 'bfloat16' or 'float16'
show_probs = False # Set to True to see chart of top 10 tokens each iteration
compile = True # use PyTorch 2.0 to compile the model to be faster
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
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

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

start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            if show_probs:
                generator = model.generate_generator(x, max_new_tokens, temperature=temperature, top_k=top_k)
                
                print("\n\nCompletetion including prompt:\n" + start, end="")
                for generation in generator:
                    _token_y = generation["iteration_token"]
                    token_y = _token_y[0].tolist()
                    token_prob = generation["iteration_token_probability"]
                    probs = generation["iteration_probability_dist"]
                    # all_tokens = generation["all_tokens_so_far"]

                    # Console "streaming" output
                    selected_token = decode(token_y)
                    print(selected_token, end="", flush=True) # Append to console output

                    # Whole string so far
                    # whole_completion = decode(all_tokens[0].tolist())

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
                    ax.set_title(f"'{selected_token}' was selected as the next token, from a probability of {token_prob*100:.2f}%")


                    # Show plot, which pauses execution until it's closed
                    plt.show()

                print("\nDone.\n")
            else:
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                print(decode(y[0].tolist()))
                print('---------------')
