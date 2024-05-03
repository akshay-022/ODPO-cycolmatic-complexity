import json
import math

# Function to load JSON file and remove entries with NaN values
def load_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    # Remove entries with NaN values
    #data = {key: value for key, value in data.items() if not math.isnan(value)}
    return data

# Load the contents of the JSON files
#dpo_0_5_0_5_1785 = load_json('eval/important_jsons/dpo_0.5_0.5_1785.json')
#dpo_0_5_0_5 = load_json('eval/important_jsons/dpo_0.5_0.5_5000.json')
dpo_0_5_0_5_1785 = load_json('eval/results/all_codes.json')
dpo_0_5_0_5 = load_json('eval/important_jsons/all_codes_second.json')

# Merge the dictionaries
dpo_merged = {**dpo_0_5_0_5_1785, **dpo_0_5_0_5}

# Write the merged dictionary to a new JSON file
with open('eval/important_jsons/deepseek_samples.json', 'w') as file:
    json.dump(dpo_merged, file, indent=4)