import json
import os


input_file = "/gscratch/zlab/rulins/data/lm-eval-data/raw_mmlu.jsonl"
output_file = "/gscratch/zlab/rulins/data/lm-eval-data/mmlu.jsonl"


raw_data = []

with open(input_file, "r") as fin:
    for line in fin:
        raw_data.append(json.loads(line))


def deduplicate_dicts(dict_list):
    unique_dicts = set()
    unique_items = []
    for item in dict_list:
        # Make a hashable version of the dictionary by sorting it
        hashable_item = tuple(sorted(item.items()))
        if hashable_item not in unique_dicts:
            unique_dicts.add(hashable_item)
            unique_items.append(item)
    return unique_items


unique_data = deduplicate_dicts(raw_data)
print(len(unique_data))

with open(output_file, "w") as fout:
    for ex in unique_data:
        fout.write(json.dumps(ex) + "\n")
