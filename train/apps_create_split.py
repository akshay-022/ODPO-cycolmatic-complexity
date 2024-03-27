import json
import os
import pathlib

def create_split(split="train", name="train"):
    paths = []
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    roots = sorted(os.listdir(os.path.join(parent_dir,split)))
    for folder in roots:
        root_path = os.path.join(split, folder)
        paths.append(root_path)


    with open(name+".json", "w") as f:
        json.dump(paths, f)
    
    return paths

# insert path to train and test
# path should be relative to root directory or absolute paths
paths_to_probs = ["data/APPS/train", "data/APPS/test"]
names = ["train", "test"]

all_paths = []
for index in range(len(paths_to_probs)):
    all_paths.extend(create_split(split=paths_to_probs[index], name=names[index]))

with open("train_and_test.json", "w") as f:
    print(f"Writing all paths. Length = {len(all_paths)}")
    json.dump(all_paths, f)
