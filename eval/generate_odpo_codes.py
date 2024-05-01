"""
Run a tranined model to generate Python code.
"""

import io
import json
import logging
import math
import random
import numpy as np
import os
import pprint
import sys
import time
import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from reindent import run as run_reindent

# for timing and debugging
from datetime import datetime, date
from tqdm import tqdm


def reindent_code(codestr):
    """
    Given code string, reindent it in the same way that the
    Github dataset was indented
    """
    codestr = io.StringIO(codestr)
    ret = io.StringIO()

    run_reindent(
        codestr, 
        ret, 
        config = {
            "dry-run": False,
            "help": False,
            "to": 10,
            "from": -1,
            "tabs": True,
            "encoding": "utf-8",
            "is-tabs": False,
            "tabsize": 10,
            "all-tabs": False
        }
    )

    return ret.getvalue()

def generate_prompt(args, test_case_path, prompt_path, solutions_path, tokenizer, starter_path=None):
    _input = "\nQUESTION:\n"
    with open(prompt_path, "r", encoding="utf-8") as f:
        data = f.readlines()
        data = "".join(data)
    _input += data
    if starter_path != None:
        with open(starter_path, "r") as f:
            data = f.readlines()
            data = "".join(data)
            data = "\n" + data #+ "\n"
        _input += data
    else:
        #_input += "\n\n"
        pass

    with open(test_case_path, "r") as f:
        data = json.load(f)
    if not data.get("fn_name"):
        _input += "\nUse Standard Input format"#\n"
    else:
        _input += "\nUse Call-Based format"#\n"
    
    _input += "\nANSWER:\n"

    if args.peeking > 0.0:
        # Need to do some peeking. 

        # Read one example solution
        with open(solutions_path, 'r') as f:
            sols = json.load(f)

        # Choose the shortest solution for the model to use.
        # This is so we can conserve tokens (1024 max)
        # sample_sol = min(sols, key=len)

        # # Add args.peeking% of that solution to the prompt
        # sample_sol_token_ids = tokenizer.encode(sample_sol, verbose=False)
        # num_to_keep = int(len(sample_sol_token_ids) * args.peeking)
        # sample_sol_token_ids = sample_sol_token_ids[:num_to_keep]
        # _input += tokenizer.decode(sample_sol_token_ids)

        # Alternatively take a random solution
        sample_sol = random.choice(sols)
        rand_sol = reindent_code(sample_sol)
        rand_sol = tokenizer.encode(rand_sol, verbose=False)
        tokens_taken = int(args.peek_frac * len(rand_sol))
        rand_sol = rand_sol[:tokens_taken]
        _input += tokenizer.decode(rand_sol)
    else:
        sample_sol = None

    return _input, sample_sol


def main(args):

    # Convert the data into the desired format
    formatted_data = {}

    # Load the JSON data from output_dpo.json
    with open('important_jsons/dpo_0.5_0.5.json', 'r') as f:
        data_odpo = json.load(f)
    with open('important_jsons/deepseek_samples.json', 'r') as f:
        data_prompts = json.load(f)

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    with open(args.test_loc, "r") as f:
        problems = json.load(f)
    problems = sorted(problems) # Pin some ordering

    gpt_codes = {}
    if not os.path.exists(args.save):
        os.makedirs(args.save, exist_ok=True)
    codes_loc = "important_jsons/odpo_alg_inputs.json"
    # Only do the problems that are specified.
    if args.index:
        problems = [problems[args.index]]
    else:
        if args.start > len(problems) or args.start < 0:
            print(f"start index {args.start} > number of problems {len(problems)}")
            return
        start = args.start
        if args.end is None or args.end > len(problems):
            end = len(problems)
        else:
            end = args.end
        problems = problems[start:end]

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct", trust_remote_code=True)

    # Set up model
    #print("Loading model...")
    #model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct", trust_remote_code=True, torch_dtype=torch.float16)
    #model.cuda()
    #print(f"Loaded {args.load}.")

    # main eval loop
    for index, problem in enumerate(tqdm(problems)):
        if str(index) in data_odpo and int(index)<1787:
            prob_path = os.path.join(args.root, problem)
            print(prob_path)
            if args.debug:
                print(f"problem path = {prob_path}")

            test_case_path = os.path.join(prob_path, "input_output.json")
            prompt_path = os.path.join(prob_path, "question.txt")
            starter_path = os.path.join(prob_path, "starter_code.py")
            solutions_path = os.path.join(prob_path, "solutions.json")
            if not os.path.exists(starter_path):
                    starter_path = None
            if not os.path.exists(test_case_path) or not os.path.exists(prompt_path):
                print(os.path.exists(test_case_path))
                print(os.path.exists(prompt_path))
                continue

            # Read the question in
            prompt_text, sample_sol = generate_prompt(args, test_case_path, prompt_path, solutions_path, tokenizer, starter_path)
            if args.debug:
                print("PROMPT_TEXT:")
                print(prompt_text)
            start = time.time()
            print(prompt_text)
            key = prompt_text
            offset_value = data_odpo[str(index)]
            if offset_value > 0:
                pairs = [(0,1)]
            else:
                pairs = [(1,0)]
            formatted_data[key] = {
                'responses': data_prompts[str(index)],
                'pairs': pairs,
                'sft_target': sample_sol,  # You need to define what this value should be
                'value': abs(offset_value)
            }

    with open(codes_loc, "w") as f:
        json.dump(formatted_data, f)


if __name__ == "__main__":
    import argparse
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    parser = argparse.ArgumentParser(description="Run a tranined model to generate Python code.")
    parser.add_argument("--arch", default="gpt2", choices=transformers.GPT2_PRETRAINED_MODEL_ARCHIVE_LIST)
    parser.add_argument("-t","--test_loc", default=os.path.join(parent_dir, 'train', 'test.json'), type=str)
    parser.add_argument("-r","--root", default="../", type=str, help="where the data is stored.")
    parser.add_argument("-l","--load", default="~/apps/models/checkpoints/final", type=str)
    parser.add_argument("--peeking", default=0.0, type=float)
    parser.add_argument("--num-beams", default=5, type=int)
    parser.add_argument("-s","--start", default=0, type=int)
    parser.add_argument("-e","--end", default=None, type=int)
    parser.add_argument("-i", "--index", default=None, type=int)
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    parser.add_argument("--save", type=str, default="./results")
 
    args = parser.parse_args()

    main(args)
