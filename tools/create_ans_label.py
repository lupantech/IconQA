from __future__ import print_function
import os
import sys
import json
# import cPickle # python2
import pickle as cPickle # python3
import utils

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_ans2label(problems, pid_splits, task, cache_root, name='trainval'):
    """
    Generate the answer-to-label dictionary and label-to-answer list file for three sub-tasks.
    - problems: problem data
    - pid_splits: problem id lists of different sub-tasks and splits
    - task: 'fill_in_blank', 'choose_txt', 'choose_img'
    - cache_root: output directory
    - name: prefix of the output file
    """
    print("\nname:", name, "task:", task)

    ans2label = {}
    label2ans = []
    label = 0

    # generate unique answer list
    if task == 'fill_in_blank':
        answers = []
         # only include the trainval split of the fill_in_blank sub-task
        trainval_pids = pid_splits['fill_in_blank_trainval']
        for pid in trainval_pids:
            ans = problems[pid]['answer']
            if ans not in answers:
                answers.append(ans)
        print("unique answers:", len(answers)) # 502

    elif task == 'choose_txt':
        answers = [str(i) for i in range(10)]
        print("unique answers:", len(answers)) # 10

    elif task == 'choose_img':
        answers = [str(i) for i in range(5)]
        print("unique answers:", len(answers)) # 5

    # generate the label2ans dict and ans2label list
    for answer in answers:
        label2ans.append(answer)
        ans2label[answer] = label
        label += 1
    print(label2ans, ans2label)

    utils.create_dir(cache_root)

    cache_file = os.path.join(cache_root, name+'_'+task+'_ans2label.pkl')
    cPickle.dump(ans2label, open(cache_file, 'wb'))
    cache_file = os.path.join(cache_root, name+'_'+task+'_label2ans.pkl')
    cPickle.dump(label2ans, open(cache_file, 'wb'))
    print('Done!')


if __name__ == '__main__':

    OUTPUT_FILE = '../data'

    problem_file = '../data/iconqa_data/problems.json'
    problems = json.load(open(problem_file))

    pid_split_file = '../data/iconqa_data/pid_splits.json'
    pid_splits = json.load(open(pid_split_file))

    # generate answer dictionary
    create_ans2label(problems, pid_splits, 'fill_in_blank', OUTPUT_FILE)
    create_ans2label(problems, pid_splits, 'choose_txt', OUTPUT_FILE)
    create_ans2label(problems, pid_splits, 'choose_img', OUTPUT_FILE)
