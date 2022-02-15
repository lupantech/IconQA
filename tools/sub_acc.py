import os
import json
import argparse


def process_results(results):
    
    skill_accs, skill_nums = {}, {}
    task_accs, task_nums = {}, {}
    ans_accs, ans_nums = {}, {}
    chon_accs, chon_nums = {}, {}
    qlen_accs, qlen_nums = {}, {}

    prob_num = len(results.keys())
    correct_num = 0

    for pid, ans in results.items():

        question = problems[pid]["question"]
        task = problems[pid]["ques_type"]
        skills = pid2skills[pid]
        ans_label = problems[pid]["answer"]

        # update skill_nums
        task_nums[task] = 1 if task not in task_nums else task_nums[task]+1
        for skill in skills:
            skill_nums[skill] = 1 if skill not in skill_nums else skill_nums[skill]+1
        ans_nums[ans_label] = 1 if ans_label not in ans_nums else ans_nums[ans_label]+1
        
        if 'choose' in task:
            chon = str(len(problems[pid]["choices"]))
            chon_nums[chon] = 1 if chon not in chon_nums else chon_nums[chon]+1
        
        qlen = str(len(question.replace(',', '').replace('?', '').replace('\'s', ' \'s').split(' ')))
        qlen_nums[qlen] = 1 if qlen not in qlen_nums else qlen_nums[qlen]+1
        
        # count corret answers
        if str(ans) == str(ans_label):
            correct_num += 1
            # update accs
            task_accs[task] = 1 if task not in task_accs else task_accs[task]+1
            for skill in skills:
                skill_accs[skill] = 1 if skill not in skill_accs else skill_accs[skill]+1
            ans_accs[ans_label] = 1 if ans_label not in ans_accs else ans_accs[ans_label]+1
            
            if 'choose' in task:
                chon_accs[chon] = 1 if chon not in chon_accs else chon_accs[chon]+1

            qlen_accs[qlen] = 1 if qlen not in qlen_accs else qlen_accs[qlen]+1

    print("[TotalAcc] \t%.2f" % (100*correct_num/prob_num), "(%d problems)\n" % prob_num)
    
    return task_accs, task_nums, skill_accs, skill_nums


def print_final_result(result_files):
    results = {}
    tasks = ['choose_img', 'choose_txt', 'fill_in_blank']
    skills = ['geometry', 'counting', 'comparing', 'spatial', 'scene', 'pattern', 'time', 
              'fraction', 'estimation', 'algebra', 'measurement', 'commonsense', 'probability']

    # read all result json files and merge them
    for result_file in result_files:
        result_data = json.load(open(result_file))
        results.update(result_data['results'])

    # process result files
    task_accs, task_nums, skill_accs, skill_nums = process_results(results)
    
    # print results for different sub-tasks and skills
    for task in tasks:
        print("[{}]          ".format(task)[:15], "%.2f" % (100*task_accs[task]/task_nums[task]))
    print("")
    for skill in skills:
        print("[{}]          ".format(skill)[:15], "%.2f" % (100*skill_accs[skill]/skill_nums[skill]))

    # # print result for Table 6 in the paper
    # print("\nLatex format:")
    # result_srt = ''
    # for task in tasks:
    #     result_srt += "& %.2f " % (100*task_accs[task]/task_nums[task])
    # for skill in skills:
    #     result_srt = result_srt + "& %.2f " % (100*skill_accs[skill]/skill_nums[skill])
    # print(result_srt + '\\\\')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fill_in_blank_result", default="exp_paper_patch_transformer_ques_bert.json")
    parser.add_argument("--choose_txt_result", default="exp_paper_patch_transformer_ques_bert.json")
    parser.add_argument("--choose_img_result", default="exp_paper_patch_transformer_ques_bert.json")
    args = parser.parse_args()

    problem_file = "../data/iconqa_data/problems.json"
    skill_file = "../data/iconqa_data/pid2skills.json"
    result_path = "../results"

    problems = json.load(open(problem_file))
    pid2skills = json.load(open(skill_file))

    fill_in_blank_result = os.path.join(result_path, "fill_in_blank", args.fill_in_blank_result)
    choose_txt_result = os.path.join(result_path, "choose_txt", args.choose_txt_result)
    choose_img_result = os.path.join(result_path, "choose_img", args.choose_img_result)
    result_files = [fill_in_blank_result, choose_txt_result, choose_img_result]

    print_final_result(result_files)
