import argparse
from typing import Tuple
import numpy as np
import random
from sklearn.utils import shuffle
import copy


class PCFG(object):

    def __init__(self, len_range: Tuple[int, int], add_distractors: bool):
    
        self.terminate_prob = 10./(len_range[1] - len_range[0])
        self.min_len, self.max_len = len_range
        self.expression = "S"
        self.rules = ["[S]S"]
        if add_distractors:
        
            self.rules.append("[S]*S")
            self.rules.append("*[S]S")
            self.rules.append("[*S]S")
            self.rules.append("[S*]S")
        self.recurse()
        
    def get_current_len(self):
    
        return self.expression.count("[") + self.expression.count("]")  #+ self.expression.count("S")  
            
    def recurse(self):
    
        # if exceeds min length, randomly choose whether to stop. If exceeds max len, stop.
        
        if (self.get_current_len() >= self.min_len and random.random() < self.terminate_prob) or self.get_current_len() + 2 > self.max_len:
        
            self.expression = self.expression.replace("S", "") #S -> epsilon
            return
        
        # otherwise, randomly recurse
        
        rule = np.random.choice(self.rules)
        self.expression = rule.replace("S", self.expression)
        self.recurse()
        

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



def main(train_size: int, dev_size: int, train_lengths: Tuple[int, int], dev_lengths: Tuple[int, int], add_distractors: bool):

    train, dev = [], []
    
    for i in range(train_size):
        seq = PCFG(train_lengths, add_distractors).expression
        train.append(seq)
        
    for i in range(dev_size):
        seq = PCFG(dev_lengths, add_distractors).expression
        dev.append(seq)
    
    lengths_train = [len(seq) for seq in train]
    print("min lengths_train", np.min(lengths_train), "max lengths_train", np.max(lengths_train), "mean lengths_train", np.mean(lengths_train), "std lengths_train", np.std(lengths_train))

    lengths_dev = [len(seq) for seq in dev]
    print("min lengths_dev", np.min(lengths_dev), "max lengths_dev", np.max(lengths_dev), "mean lengths_dev", np.mean(lengths_dev), "std lengths_dev", np.std(lengths_dev))
    
    
    
    for i in range(train_size//2):
    
        train[i] = corrupt_sequence(train[i])

    for i in range(dev_size//2):
    
        dev[i] = corrupt_sequence(dev[i])

   
    labels_train = [0] * (train_size//2) + [1] * (train_size//2)
    labels_dev = [0] * (dev_size//2) + [1] * (dev_size//2)
    
    train_x, train_y = shuffle(train, labels_train)
    dev_x, dev_y = shuffle(dev, labels_dev)
    
    write_to_file("train.txt", train_x, train_y)
    write_to_file("dev.txt", dev_x, dev_y)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='balanced brackets generation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train-size', dest='train_size', type=int,
                        default=15000,
                        help='number of training examples to generate')
    parser.add_argument('--dev-size', dest='dev_size', type=int,
                        default=15000,
                        help='number of dev examples to generate')
    parser.add_argument('--min-len-train', dest='min_len_train', type=int,
                        default= 8,
                        help='min length in the training set')
    parser.add_argument('--max-len-train', dest='max_len_train', type=int,
                        default= 30,
                        help='max length in the training set')
    parser.add_argument('--min-len-dev', dest='min_len_dev', type=int,
                        default= 15,
                        help='min length in the dev set')
    parser.add_argument('--max-len-dev', dest='max_len_dev', type=int,
                        default= 45,
                        help='min length in the dev set')
    parser.add_argument('--add-distractors', dest='add_distractors', type=bool,
                        default= True,
                        help='Whether to add distractor symbols (*)')                                      
    args = parser.parse_args()
    
    train_lens = (args.min_len_train, args.max_len_train)
    dev_lens = (args.min_len_dev, args.max_len_dev)
    
    main(args.train_size, args.dev_size, train_lens, dev_lens, args.add_distractors)
    
