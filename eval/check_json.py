import json

# Read the contents of all_codes_1.json
with open('all_codes_1.json', 'r') as file1:
    data1 = json.load(file1)

# Read the contents of all_codes_2.json
with open('all_codes_2.json', 'r') as file2:
    data2 = json.load(file2)

# Combine the data from both files
combined_data = {**data1, **data2}

# Write the combined data to all_codes.json
with open('all_codes.json', 'w') as outfile:
    json.dump(combined_data, outfile)