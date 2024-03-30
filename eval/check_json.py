import json

file_path = '/Users/akshayiyer/Desktop/MS CS Columbia/Sem 2 - Spring 2024/CodeGen/ODPO-cyclomatic-complexity/eval/all_codes.json'

with open(file_path, 'r') as file:
    data = json.load(file)

print(len(data))

# Now you can use the 'data' variable to access the contents of the JSON file