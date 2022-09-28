from __future__ import print_function
import os
import sys
import json
import argparse
from utils import Dictionary
from datetime import datetime
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('../')


def create_dictionary(question_path, add_choice):
    """
    Generate the question word dictionary
    - question_path: question data path
    - add_choice: add choice words to dictionary or not
    """
    dictionary = Dictionary()

    qs = json.load(open(question_path))
    for pid, data in qs.items():
        ques = data['question']
        # add choice words to dictionary
        if add_choice and data['ques_type'] == 'choose_txt':
            choice_text = ' '.join(data['choices'])
            ques = ques + ' ' + choice_text
        dictionary.tokenize(ques, True)

    return dictionary


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='../data')
    parser.add_argument('--output', type=str, default='../data')

    args = parser.parse_args()
    # General
    ques_file = args.input + '/iconqa_data/problems.json'
    out_path = args.output
    add_choice = True

    d = create_dictionary(ques_file, add_choice)
    d.dump_to_file(os.path.join(out_path, 'dictionary.pkl'))
    print(datetime.now().isoformat(), " ","dictionary size:", len(d)) # 2263 with choice words, 2085 without choice words
