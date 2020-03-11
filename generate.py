
import argparse
from typing import Tuple
import numpy as np
import random
from sklearn.utils import shuffle
import copy

import tqdm


# from https://sudonull.com/post/123077-How-to-generate-random-parenthesis-sequences
def tryAndFix(n):
  seq  = ['(', ')'] * n
  random.shuffle(seq)
  stack   = []
  result  = []
  balance = 0
  prev    = 0
  for pos in range( len(seq) ):
    balance += 1 if seq[pos] == '(' else -1
    if balance == 0:
      if seq[prev] == '(':
        result.extend( seq[ prev : pos + 1 ] )
      else:
        result.append('(')
        stack.append( [ ')' if v == '(' else '(' for v in seq[ prev + 1 : pos ] ] )
      prev = pos + 1   
  for lst in reversed(stack):
    result.append(')')
    result.extend(lst)
  return result


def get_opposite_bracket(bracket):

    if bracket == "[": return "]"
    elif bracket == "]": return "["
    elif bracket == "(": return ")"
    elif bracket == ")": return "("
    
def corrupt_sequence(positive_sequence: str) -> str:
    original = copy.copy(positive_sequence)
    
    positive_sequence_lst = list(positive_sequence)
    already_chosen = set()
    relevant = [i for i in range(len(positive_sequence)) if positive_sequence[i] != "*"]
    
    # swipe randomly a number of brackets of different types to create negative examples
    
    if np.random.random() < 0.5:
        
        ind = random.choice([j for j in relevant if j not in already_chosen])
        already_chosen.add(ind)
        bracket = positive_sequence_lst[ind]
        positive_sequence_lst[ind] = get_opposite_bracket(bracket)
            
    # randomly remove an even number of brackets from the same type
    else:
    
        while True:
            ind, ind2 = np.random.choice(relevant, size = 2)
            if positive_sequence_lst[ind] == positive_sequence_lst[ind2]:
            
                positive_sequence_lst = [x for i,x in enumerate(positive_sequence_lst) if i!=ind and i!=ind2]
            break
   
    #print("Original: {}; now: {}".format(positive_sequence, original))
    return "".join(positive_sequence_lst)
                
        
            
        
def write_to_file(name: str, X: list, Y: list):


    with open(name, "w") as f:
    
        for x,y in zip(X,Y):
        
            f.write(x + "\t" + str(y) + "\n")
            
            
            
def main(train_size: int, dev_size: int, train_lengths: Tuple[int, int], dev_lengths: Tuple[int, int]):

    train, dev = [], []
    num_seqs_per_length_train = train_size // (train_lengths[1] - train_lengths[0])
    num_seqs_per_length_dev = dev_size // (dev_lengths[1] - dev_lengths[0])
    
    for length in tqdm.tqdm(range(*train_lengths), total = train_lengths[1] - train_lengths[0]):
    
        # generate seqs for this lengths
        for i in range(num_seqs_per_length_train):
        
            seq = "".join(tryAndFix(length))
            train.append(seq)      
        
    for length in tqdm.tqdm(range(*dev_lengths), total = dev_lengths[1] - dev_lengths[0]):
    
        # generate seqs for this lengths
        for i in range(num_seqs_per_length_dev):
        
            seq = "".join(tryAndFix(length))
            dev.append(seq)   
    
    random.shuffle(train)
    random.shuffle(dev)
    
    l1, l2 = len(train)//2, len(dev)//2
    
    for i in range(l1):
    
        train[i] = corrupt_sequence(train[i])

    for i in range(l2):
    
        dev[i] = corrupt_sequence(dev[i])

   
    labels_train = [0] * l1 + [1] * l1
    labels_dev = [0] * l2 + [1] * l2
    
    train_x, train_y = shuffle(train, labels_train)
    dev_x, dev_y = shuffle(dev, labels_dev)
    
    write_to_file("train.txt", train_x, train_y)
    write_to_file("dev.txt", dev_x, dev_y)
    
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

    parser.add_argument('--double-brackets', dest='double', type=bool,
                        default= True,
                        help='Whether to use double brackets')                                       
    args = parser.parse_args()
    

    train_lens = (args.min_len_train, args.max_len_train)
    dev_lens = (args.min_len_dev, args.max_len_dev)
    
    main(args.train_size, args.dev_size, train_lens, dev_lens)
