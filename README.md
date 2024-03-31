# CyCoder
This is the repository for **CyCoder -- Optimizing Code Generation via Reinforcement Learning** 


This repository contains both training and evaluation code referenced from [here](#References).

## Setup

> **Dataset**
  - Download the [**APPS dataset here**](https://people.eecs.berkeley.edu/~hendrycks/APPS.tar.gz). (~1.3GB)
    - The dataset is also available in [Hugging Face datasets](https://huggingface.co/datasets/codeparrot/apps) under apps.

> **Model**
  - Using [deepseek-ai/deepseek-coder-1.3b-instruct](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct) Model


- First use [train/apps_create_split.py](train/apps_create_split.py) to create the `train.json` and `test.json`. Note the paths specified in `apps_create_split.py` should point to relative paths from training directory or absolute paths.

<!---
# Training

## How to train

 - We use the following command to run and train. 

  ```
  python3 train/tune_apps_gpt.py  --save-dir=<save_path> --load=<model_path> --apps-train-files <data_path>/train --apps-dataroot <data_path> --grad-acc-steps=8 --epochs=10 --fp16 --batch-size-per-replica=2
  ```
--->

# Eval
The evaluation instructions are specified in [eval/README](eval/README.md).

### References 
- [Measuring Coding Challenge Competence With APPS](https://arxiv.org/pdf/2105.09938) -- Hendrycks et al.
- DeepSeek-coder documentation : https://github.com/deepseek-ai/DeepSeek-Coder
