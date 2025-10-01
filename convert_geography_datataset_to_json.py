#! python3

import csv
import json
import random

in_file = "Geography_11th_Cleaned.csv"
out_file = "eval_data.json"

with open(out_file, 'w') as jsonfile:
    rows = []
    with open(in_file) as csvfile:
        reader = csv.reader(csvfile)
        first_row = True
        for row in reader:
            if first_row:
                # Skip headers
                first_row = False
                continue

            question = row[2]
            answer = row[3]
            rows.append({"prompt": question, "response": answer})
    
    # Take 10 random items
    random.shuffle(rows)
    rows = rows[:10]
    json.dump(rows, jsonfile)

