"""
Eval outpus via Cyclometric Complexity measure.
"""

import json
import logging
import math
import numpy as np
import os
import pprint
import random
import sys
import time
import re


# for timing debugging
from datetime import datetime, date
from tqdm import tqdm

from typing import List

#complexity imports
import radon
from radon.complexity import cc_visit
from radon.cli.harvest import CCHarvester

md = MosesDetokenizer(lang='en')

random.seed(12345678987654321)

def calc_complexity(output):
    def extract_substring(text):
        # Define the regular expression pattern to match the substring inside triple quotes
        pattern = r'"""python\n(.*?)"""'

        # Use re.search() to find the first occurrence of the pattern
        match = re.search(pattern, text, re.DOTALL)

        if match:
            # Extract and return the substring
            return match.group(1)
        else:
            return None
    # isolate code string inside triple quotation marks from output
    code_string = extract_substring(output)


    # Calculate cyclomatic complexity
    complexity_results = cc_visit(code_string)
    print(complexity_results)
    total_complexity = sum([result.complexity for result in complexity_results])

    return total_complexity



def eval_and_save_complexity_scores(args):
    with open(args.test_loc, "r") as f:
        problems = json.load(f)

    gpt_codes = {}
    gpt_complexity = {}
    codes_loc = os.path.join(args.save, f"all_codes.json")
    if not os.path.exists(codes_loc):
        codes_loc = os.path.join(args.save, f"{args.start}-{args.end}_codes.json")
 
    if os.path.exists(codes_loc):
        with open(codes_loc, "r") as f:
            gpt_codes = json.load(f)

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

    # main eval loop
    for index, problem in enumerate(tqdm(problems)):
        prob_path = os.path.join(args.root, problem)
        if args.debug:
            print(f"problem path = {problem}")
        try:
            output_strs = gpt_codes[str(index+args.start)]
        except:
            continue

        if args.debug:
            print(output_str)



        # this is if we generated multiple outputs per problem
        if isinstance(output_strs, list):
            gpt_complexity[index+args.start] = []
            for output_str in output_strs:
                gpt_complexity[index+args.start].extend(calc_complexity(output_str))
        # one output per problem
        else:
            output_str = output_strs
            gpt_complexity[index+args.start] = calc_complexity(output_str)

        if not os.path.exists(args.save):
            os.makedirs(args.save)

        if args.end is None and args.index is None:
            complexity_loc = os.path.join(args.save, f"all_complexity_results.json")
        elif args.index:
            complexity_loc = os.path.join(args.save, f"{args.index}_complexity_results.json")
        else:
            complexity_loc = os.path.join(args.save, f"{args.start}-{args.end}_complexity_results.json")

        with open(complexity_loc, "w") as f:
            json.dump(gpt_complexity, f)

    return gpt_complexity

def print_results(results):
    complexity_scores = []

    for res in results:
        complexity_scores.append(results[res])
    print(f"Average Complexity Score = {np.mean(complexity_scores)}")



def main(args):

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    if args.print_results:
        complexity_loc = os.path.join(args.save, f"all_complexity_results.json")
        if os.path.exists(complexity_loc):
            with open(complexity_loc, "r") as f:
                results = json.load(f)
        else:
            print(f"Error file does not exist in this path {complexity_loc}. Exiting.")
            return
    else:
        results = eval_and_save_complexity_scores(args)

    print_results(results)


if __name__ == "__main__":
    import argparse

    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    parser = argparse.ArgumentParser(description="Complexity Evaluation")
    parser.add_argument("-t", "--test_loc", default=os.path.join(parent_dir, 'train', 'test.json'), type=str)
    parser.add_argument("-r","--root", default="../", type=str, help="where the data is stored.")
    parser.add_argument("-s","--start", default=0, type=int)
    parser.add_argument("-e","--end", default=None, type=int)
    parser.add_argument("-i", "--index", default=None, type=int)
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-p", "--print_results", action="store_true", help="If you have already evaluated the results and only want to print them.")
    parser.add_argument("--save", type=str, default="./results")
 
    args = parser.parse_args()

    main(args)
