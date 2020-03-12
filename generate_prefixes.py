
import argparse
from typing import Tuple
import numpy as np
import random
from sklearn.utils import shuffle
import copy
from typing import List, Dict
import tqdm
import json


def generate_prefix(n, opening2closing = {"(": ")", "[": "]"}, branching_p = 0.5):

    """
    this function generate a Dyck string, which is a valid prefix of a balanced expression (but not necessarily a balanced one).
    :param n - the length of the sequence
    :param opening2closing: a dictionary, mapping opening brackets to closing brackets
    :param branching_p: probability of opening a new bracket
    """
    
    if n % 2 != 0: raise Exception("length should be even")
    opening_bracket_types = list(opening2closing.keys())
    stack = []

    is_closed = [False] * n
    seq = [random.choice(opening_bracket_types)]
    stack.append((seq[0], 0)) # the stack contains the currently top open bracket, and its linear index
    depths = [1] # we start with depth of 1 becasue of the opening bracket
    current_top_of_stack = [stack[0]]
    
    for i in range(n-1):
        
        if random.random() < branching_p or len(stack) == 0: # open a new bracket
        
            opening_bracket = random.choice(opening_bracket_types)
            depths.append(depths[-1] + 1)
            seq.append(opening_bracket)
            stack.append((opening_bracket, i + 1))
            
        else: # close the current open bracket
        
            top, ind = stack.pop()
            closing_bracket = opening2closing[top]
            seq.append(closing_bracket)
            is_closed[ind] = True
            depths.append(depths[-1] - 1)
            
        current_top_of_stack.append(stack[-1] if len(stack) > 0 else ("-", "-"))
    
    # collect some useful stats
    
    top_of_stack_brackets, top_of_stack_idx = list(zip(*current_top_of_stack))
    not_matched_idx = [i for i in range(len(seq)) if (seq[i] in opening_bracket_types) and not is_closed[i]]
    is_balanced = len(not_matched_idx) == 0
    next_closing_bracket = opening2closing[seq[not_matched_idx[-1]]] if not_matched_idx else "-"
    
    # calculate embedded depth and distance from Yu et al. 
    
    if not is_balanced:
        last_open_i = not_matched_idx[-1]
        distance = len(seq) - last_open_i
        embedded_depth = max(depths[last_open_i:])
    else:
        distance, embedded_depth = 0, 0
        
    return {"seq": "".join(seq), "not_matched_idx": not_matched_idx, "top_of_stack_brackets": top_of_stack_brackets, 
            "top_of_stack_idx": top_of_stack_idx, "depths": depths, "is_balanced": is_balanced, 
            "next_closing_bracket": next_closing_bracket, "distance": distance, "embedded_depth": embedded_depth, "length": n}

        
        
def write_to_file(name: str, data: List[Dict]):

    print("Writing to file {}".format(name))
    
    with open(name, "w") as f:
    
        for data_dict in tqdm.tqdm(data, total = len(data)):
            json.dump(data_dict, f)
            f.write("\n")
            
            
def main(train_size: int, dev_size: int, train_lengths: Tuple[int, int], dev_lengths: Tuple[int, int], branching_probability: float, opening2closing: dict):

    # generate number of examples that is proportional to length (more challenging examples)
    
    train, dev = [], []
    len2num_examples_train = {}
    len2num_examples_dev = {}
    train_length_range = list(range(*train_lengths, 2))
    dev_length_range = list(range(*dev_lengths, 2))
    
    for length in train_length_range:
    
        len2num_examples_train[length] = int((length / sum(train_length_range)) * train_size)
        
    for length in dev_length_range:
    
        len2num_examples_dev[length] = int((length / sum(dev_length_range)) * dev_size)
            
    # collect training and dev examples
    pbar = tqdm.tqdm(total=train_size + dev_size)
    
    for length in train_length_range:
    
        # generate seqs for this lengths
        num_examples = len2num_examples_train[length]
        
        for i in range(num_examples):
            pbar.update(1)
            data = generate_prefix(length, opening2closing = opening2closing, branching_p = branching_probability)
            train.append(data)      
        
    for length in dev_length_range:
    
        # generate seqs for this lengths
        num_examples = len2num_examples_dev[length]
        
        for i in range(num_examples):
            pbar.update(1)
            data = generate_prefix(length, opening2closing = opening2closing, branching_p = branching_probability)
            dev.append(data)      
    
    pbar.close()
    
    random.shuffle(train)
    random.shuffle(dev)

    write_to_file("train.jsonl", train)
    write_to_file("dev.jsonl", dev)
    
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='balanced brackets generation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train-size', dest='train_size', type=int,
                        default=75000,
                        help='number of training examples to generate')
    parser.add_argument('--dev-size', dest='dev_size', type=int,
                        default=75000,
                        help='number of dev examples to generate')
    parser.add_argument('--min-len-train', dest='min_len_train', type=int,
                        default= 30,
                        help='min length in the training set')
    parser.add_argument('--max-len-train', dest='max_len_train', type=int,
                        default= 100,
                        help='max length in the training set')
    parser.add_argument('--min-len-dev', dest='min_len_dev', type=int,
                        default= 30,
                        help='min length in the dev set')
    parser.add_argument('--max-len-dev', dest='max_len_dev', type=int,
                        default= 100,
                        help='min length in the dev set')
    parser.add_argument('--branching-probability', dest='branching_prob', type=float,
                        default= 0.5,
                        help='probability of adding a new opening bracket')
    parser.add_argument('--double-brackets', dest='double', type=bool,
                        default= True,
                        help='Whether to use double brackets')                                       
    args = parser.parse_args()
    

    train_lens = (args.min_len_train, args.max_len_train)
    dev_lens = (args.min_len_dev, args.max_len_dev)
    opening2closing = {"(": ")"}
    if args.double:
        opening2closing["["] = "]"
        
    main(args.train_size, args.dev_size, train_lens, dev_lens, args.branching_prob, opening2closing)
