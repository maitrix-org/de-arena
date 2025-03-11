import json
import numpy as np
import copy
import tqdm
import csv
from utils_final import existing_model_paths, gt_scores
import fire
import random
import scipy.stats
import pandas as pd
import os
from tqdm import tqdm
from scipy.stats import spearmanr
import re
from multiprocessing import Pool, cpu_count
import fire
import random
import uuid
import scipy.stats
import pandas as pd
import os
from tqdm import tqdm
from scipy.stats import spearmanr
import itertools
from openai import OpenAI
import google.generativeai as genai
import anthropic 
from tokencost import calculate_completion_cost, calculate_prompt_cost
from decimal import Decimal
import sys
import google.generativeai as genai
import time
import re
from judge_responses_new import get_question_with_reference, judge_prompt_pair_reference, judge_prompt_pairwise, fetch_responses, determine_winner, load_records

judeg_api_model = ['gpt3.5-turbo-0125','ChatGPT-4o-latest (2024-09-03)','gpt-4o-mini-2024-07-18','gpt-4o-2024-08-06','gemini-1.5-flash-001']
judge_open_model = ['athene-70b', 'gemma-1.1-7b-it', 'gemma-2-27b-it', 'gemma-2-9b-it-simpo', 'google-gemma-2-9b-it', 'gemma-1.1-2b-it', 'gemma-2b-it', 'gemma-7b-it', 'yi-1.5-34b-chat', 'mistral-7b-instruct-1', 'mistral-8x7b-instruct-v0.1', 'llama2-13b-chat', 'llama2-7b-chat', 'command-r-(04-2024)', 'command-r-(08-2024)', 'qwen1.5-14b-chat', 'qwen1.5-32b-chat', 'qwen2-72b-instruct', 'qwen1.5-4b-chat', 'qwen1.5-72b-chat', 'openchat-3.5', 'openchat-3.5-0106', 'vicuna-7b', 'vicuna-13b', 'vicuna-33b', 'starling-lm-7b-alpha', 'koala-13b', 'openassistant-pythia-12b','gemma-2-2b-it']#'mistral-7b-instruct-2'
judge_model_list = judeg_api_model+judge_open_model
overall_ids = [i for i in range(81,160)]

def rank_scores(scores):
    indexed_scores = list(enumerate(scores))
    sorted_scores = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
    ranks = [0] * len(scores)
    for rank, (index, _) in enumerate(sorted_scores):
        ranks[index] = rank
    return ranks

def rugged_rank(base_dir,new_model,base_model_list,base_model_ranking,model_weights,judge_model_list,valid_question_ids=overall_ids):
    rank_list = list()
    total_weight = 0
    weighted_rank_sum = 0
    judge_models = [model for model in base_model_list if model in judge_model_list]
    # print(judge_models)
    for judge_model in judge_models:
        # print(rank,judge_model)
        models_to_sort = [model for model in base_model_list if model != judge_model]
        left, right = 0, len(models_to_sort)
        while left < right:
            mid = (left + right) // 2
            # print(new_model,mid,models_to_sort[mid],get_vote_result_for_judge(base_dir, judge_model, new_model, models_to_sort[mid]))
            if get_vote_result_for_judge(base_dir, judge_model, new_model, models_to_sort[mid],valid_question_ids) < 0:
                left = mid + 1
            else:
                right = mid
        if left == 0:
            rank = 1
        else:
            rank = base_model_ranking[models_to_sort[left-1]]+1
        # print(left, rank)
        if rank == base_model_ranking[judge_model]:
            vote_diff = 0
            for model in judge_models:
                if model == judge_model:
                    continue
                vote_diff += get_vote_result_for_judge(base_dir, model, new_model, judge_model,valid_question_ids)
            if vote_diff > 0:
                rank = base_model_ranking[judge_model]
            else:
                rank = base_model_ranking[judge_model]+1
        rank_list.append(rank)
        total_weight += model_weights.get(judge_model, 0.3)  # Default weight is 1 if not found
        weighted_rank_sum += rank * model_weights.get(judge_model, 1)
    weighted_average_rank = weighted_rank_sum / total_weight
    # print(rank_list)
    return weighted_average_rank

def full_comparsion(base_dir,new_model,base_model_list,sort_rank,model_weights,judge_model_list,window=1,valid_question_ids=overall_ids):
    rank_idx = int(sort_rank)
    min_rank_idx = max(1,rank_idx-window+1)
    max_rank_idx = min(len(base_model_list),rank_idx+window)
    model_names = list()
    # print(min_rank_idx,max_rank_idx)
    for i in range(min_rank_idx-1,max_rank_idx):
        model_names.append(base_model_list[i])
    model_names.append(new_model)
    base_model_list.append(new_model)
    judge_models = [model for model in base_model_list if model in judge_model_list]
    # print(judge_models)
    sort_model_index_map = {name: idx for idx, name in enumerate(model_names)}
    judge_model_index_map = {name: idx for idx, name in enumerate(judge_models)}
    # Initialize an empty comparison matrix
    comparison_matrix = np.zeros((len(judge_models), len(judge_models)))
    weights = list()
    for subdir in os.listdir(base_dir):
        # print(subdir)
        if subdir not in base_model_list:
            continue 
        if subdir not in judge_models:
            continue
        # print(f"Working on {subdir}")
        jsonl_path = os.path.join(base_dir, subdir, "voting_records.jsonl")
        if os.path.exists(jsonl_path):
            weight = model_weights.get(subdir,0.3)
            weights.append(weight)
            with open(jsonl_path, 'r') as file:
                for line in file:
                    record = json.loads(line,strict=False)
                    for each in record:
                        if valid_question_ids and each.get('question_id') not in valid_question_ids:
                            continue

                        model1 = each['response_A']
                        model2 = each['response_B']
                        winner = each['Won']

                        idx1 = sort_model_index_map.get(model1)
                        idx2 = sort_model_index_map.get(model2)
                        judge_idx = judge_model_index_map.get(subdir)


                        if idx1 is not None and idx2 is not None and judge_idx is not None:
                            if winner == model1:
                                comparison_matrix[judge_idx, idx1] += 1
                            elif winner == model2:
                                comparison_matrix[judge_idx, idx2] += 1
                            else:
                                comparison_matrix[judge_idx, idx1] += 0
                                comparison_matrix[judge_idx, idx2] += 0
        jsonl_path = os.path.join(base_dir, subdir, "voting_records_1.jsonl")
        if os.path.exists(jsonl_path):
            weight = model_weights.get(subdir,0.3)
            weights.append(weight)
            with open(jsonl_path, 'r') as file:
                for line in file:
                    record = json.loads(line,strict=False)
                    for each in record:
                        if valid_question_ids and each.get('question_id') not in valid_question_ids:
                            continue

                        model1 = each['response_A']
                        model2 = each['response_B']
                        winner = each['Won']

                        idx1 = sort_model_index_map.get(model1)
                        idx2 = sort_model_index_map.get(model2)
                        judge_idx = judge_model_index_map.get(subdir)


                        if idx1 is not None and idx2 is not None and judge_idx is not None:
                            if winner == model1:
                                comparison_matrix[judge_idx, idx1] += 1
                            elif winner == model2:
                                comparison_matrix[judge_idx, idx2] += 1
                            else:
                                comparison_matrix[judge_idx, idx1] += 0
                                comparison_matrix[judge_idx, idx2] += 0
        jsonl_path = os.path.join(base_dir, subdir, "voting_records_2.jsonl")
        if os.path.exists(jsonl_path):
            # weight = model_weights.get(subdir,0.3)
            # weights.append(weight)
            with open(jsonl_path, 'r') as file:
                for line in file:
                    record = json.loads(line,strict=False)
                    for each in record:
                        if valid_question_ids and each.get('question_id') not in valid_question_ids:
                            continue

                        model1 = each['response_A']
                        model2 = each['response_B']
                        winner = each['Won']

                        idx1 = sort_model_index_map.get(model1)
                        idx2 = sort_model_index_map.get(model2)
                        judge_idx = judge_model_index_map.get(subdir)


                        if idx1 is not None and idx2 is not None and judge_idx is not None:
                            if winner == model1:
                                comparison_matrix[judge_idx, idx1] += 1
                            elif winner == model2:
                                comparison_matrix[judge_idx, idx2] += 1
                            else:
                                comparison_matrix[judge_idx, idx1] += 0
                                comparison_matrix[judge_idx, idx2] += 0
    # print(comparison_matrix,weights) 
    # 计算每一行的和
    row_sums = comparison_matrix.sum(axis=1, keepdims=True)

    # 创建归一化矩阵，初始化为原始矩阵
    normalized_matrix = np.copy(comparison_matrix)

    # 对每一行进行归一化（忽略和为0的行）
    nonzero_row_indices = row_sums.flatten() != 0
    normalized_matrix[nonzero_row_indices] = comparison_matrix[nonzero_row_indices] / row_sums[nonzero_row_indices]
    return sort_model_index_map,min_rank_idx,normalized_matrix,weights

def get_vote_result_for_judge(base_dir, judge_model, model1, model2, valid_question_ids=overall_ids):
    vote_diff = 0
    jsonl_path = os.path.join(base_dir, judge_model, "voting_records.jsonl")
    # print(judge_model,model1,model2)
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r') as file:
            for line in file:
                # print(jsonl_path)
                record = json.loads(line)
                for each in record:
                    if valid_question_ids and each.get('question_id') not in valid_question_ids:
                        continue

                    if each['response_A'] == model1 and each['response_B'] == model2:
                        # print(each)
                        if each['Won'] == model1:
                            vote_diff += 1
                        elif each['Won'] == model2:
                            vote_diff -= 1
                    elif each['response_A'] == model2 and each['response_B'] == model1:
                        # print(each)
                        if each['Won'] == model2:
                            vote_diff -= 1
                        elif each['Won'] == model1:
                            vote_diff += 1

    jsonl_path = os.path.join(base_dir, judge_model, "voting_records_1.jsonl")
    # print(judge_model,model1,model2)
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r') as file:
            for line in file:
                # print(jsonl_path)
                record = json.loads(line)
                for each in record:
                    if valid_question_ids and each.get('question_id') not in valid_question_ids:
                        continue

                    if each['response_A'] == model1 and each['response_B'] == model2:
                        # print(each)
                        if each['Won'] == model1:
                            vote_diff += 1
                        elif each['Won'] == model2:
                            vote_diff -= 1
                    elif each['response_A'] == model2 and each['response_B'] == model1:
                        # print(each)
                        if each['Won'] == model2:
                            vote_diff -= 1
                        elif each['Won'] == model1:
                            vote_diff += 1
    jsonl_path = os.path.join(base_dir, judge_model, "voting_records_2.jsonl")
    # print(judge_model,model1,model2)
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r') as file:
            for line in file:
                # print(jsonl_path)
                record = json.loads(line)
                for each in record:
                    if valid_question_ids and each.get('question_id') not in valid_question_ids:
                        continue

                    if each['response_A'] == model1 and each['response_B'] == model2:
                        # print(each)
                        if each['Won'] == model1:
                            vote_diff += 1
                        elif each['Won'] == model2:
                            vote_diff -= 1
                    elif each['response_A'] == model2 and each['response_B'] == model1:
                        # print(each)
                        if each['Won'] == model2:
                            vote_diff -= 1
                        elif each['Won'] == model1:
                            vote_diff += 1
    # breakpoint()
    return vote_diff

def integrate_rankings(original_ranking, new_ranking, relative_ranking):
    # 找到new_ranking中对应模型在original_ranking中的最小排名
    min_new_rank = min(original_ranking[model] for model in new_ranking)

    # 创建一个字典来存储最终的整体排名，先复制原始排名
    final_ranking = original_ranking.copy()

    max_ranking = max(relative_ranking)+min_new_rank
    for model,rank in final_ranking.items():
        if rank >= max_ranking:
            final_ranking[model] += 1

    # 遍历new_ranking中的每个模型
    for model, idx in new_ranking.items():
        # 获取该模型在relative_ranking中的排名
        new_relative_rank = relative_ranking[idx]
        # 更新final_ranking中的排名，新排名加上new_ranking中的最小排名减一
        final_ranking[model] = new_relative_rank + min_new_rank

    return final_ranking

def bubble_window(base_dir,new_model,base_model_list,new_model_rank,model_weights,judge_model_list,window=1,valid_question_ids=overall_ids):
    # print(new_model_rank,base_model_list)
    model_names = list()
    model_names.append(base_model_list[new_model_rank-2])
    model_names.append(base_model_list[new_model_rank])
    model_names.append(base_model_list[new_model_rank-1])
    judge_models = [model for model in base_model_list if model in judge_model_list]
    sort_model_index_map = {name: idx for idx, name in enumerate(model_names)}
    judge_model_index_map = {name: idx for idx, name in enumerate(judge_models)}
    # Initialize an empty comparison matrix
    comparison_matrix = np.zeros((len(judge_models), len(judge_models)))
    weights = list()
    for subdir in os.listdir(base_dir):
        if subdir not in base_model_list:
            continue 
        if subdir not in judge_models:
            continue
        # print(f"Working on {subdir}")
        jsonl_path = os.path.join(base_dir, subdir, "voting_records.jsonl")
        if os.path.exists(jsonl_path):
            weight = model_weights.get(subdir, 0.3)
            weights.append(weight)
            with open(jsonl_path, 'r') as file:
                for line in file:
                    record = json.loads(line)
                    for each in record:
                        if valid_question_ids and each.get('question_id') not in valid_question_ids:
                            continue

                        model1 = each['response_A']
                        model2 = each['response_B']
                        winner = each['Won']

                        idx1 = sort_model_index_map.get(model1)
                        idx2 = sort_model_index_map.get(model2)
                        judge_idx = judge_model_index_map.get(subdir)

                        if idx1 is not None and idx2 is not None and judge_idx is not None:
                            if winner == model1:
                                comparison_matrix[judge_idx, idx1] += 1*weight
                            elif winner == model2:
                                comparison_matrix[judge_idx, idx2] += 1*weight
                            else:
                                comparison_matrix[judge_idx, idx1] += 0*weight
                                comparison_matrix[judge_idx, idx2] += 0*weight
        jsonl_path = os.path.join(base_dir, subdir, "voting_records_1.jsonl")
        if os.path.exists(jsonl_path):
            weight = model_weights.get(subdir, 0.3)
            weights.append(weight)
            with open(jsonl_path, 'r') as file:
                for line in file:
                    record = json.loads(line)
                    for each in record:
                        if valid_question_ids and each.get('question_id') not in valid_question_ids:
                            continue

                        model1 = each['response_A']
                        model2 = each['response_B']
                        winner = each['Won']

                        idx1 = sort_model_index_map.get(model1)
                        idx2 = sort_model_index_map.get(model2)
                        judge_idx = judge_model_index_map.get(subdir)

                        if idx1 is not None and idx2 is not None and judge_idx is not None:
                            if winner == model1:
                                comparison_matrix[judge_idx, idx1] += 1*weight
                            elif winner == model2:
                                comparison_matrix[judge_idx, idx2] += 1*weight
                            else:
                                comparison_matrix[judge_idx, idx1] += 0*weight
                                comparison_matrix[judge_idx, idx2] += 0*weight
        jsonl_path = os.path.join(base_dir, subdir, "voting_records_2.jsonl")
        if os.path.exists(jsonl_path):
            # weight = model_weights.get(subdir, 0.3)
            # weights.append(weight)
            with open(jsonl_path, 'r') as file:
                for line in file:
                    record = json.loads(line)
                    for each in record:
                        if valid_question_ids and each.get('question_id') not in valid_question_ids:
                            continue

                        model1 = each['response_A']
                        model2 = each['response_B']
                        winner = each['Won']

                        idx1 = sort_model_index_map.get(model1)
                        idx2 = sort_model_index_map.get(model2)
                        judge_idx = judge_model_index_map.get(subdir)

                        if idx1 is not None and idx2 is not None and judge_idx is not None:
                            if winner == model1:
                                comparison_matrix[judge_idx, idx1] += 1*weight
                            elif winner == model2:
                                comparison_matrix[judge_idx, idx2] += 1*weight
                            else:
                                comparison_matrix[judge_idx, idx1] += 0*weight
                                comparison_matrix[judge_idx, idx2] += 0*weight
    # print(comparison_matrix)
    # 计算每一行的和
    row_sums = comparison_matrix.sum(axis=1, keepdims=True)

    # 创建归一化矩阵，初始化为原始矩阵
    normalized_matrix = np.copy(comparison_matrix)

    # 对每一行进行归一化（忽略和为0的行）
    nonzero_row_indices = row_sums.flatten() != 0
    normalized_matrix[nonzero_row_indices] = comparison_matrix[nonzero_row_indices] / row_sums[nonzero_row_indices]
    # print(comparison_matrix) 
    return model_names,sort_model_index_map,normalized_matrix,weights

def vote_to_rank(vote_matrix,weights,n):
    # 提取前三列
    first_three_columns = vote_matrix[:, :n]
    vote_sum = list()
    # 分别计算前三列的和
    for i in range(n):
        vote_sum.append(np.sum(first_three_columns[:, i]))
        # vote_sum.append(np.sum(first_three_columns[:, i]))
        sorted_indices = sorted(range(len(vote_sum)), key=lambda i: (vote_sum[i] == 0, -vote_sum[i]))
        # 计算排序后的排名
    ranking = [0] * n
    for rank, index in enumerate(sorted_indices):
        ranking[index] = rank  
    return ranking

def update_model_weight(initial_weight,base_model_ranking,judge_model_list):
    decrement = 0.7 / (len(base_model_ranking)-1)
    model_weights = {model: initial_weight - (rank - 1) * decrement for model, rank in base_model_ranking.items() if model in judge_model_list}
    return model_weights

def update_bubble_window_rank(base_model_ranking,model_names,new_model_rank,ranking):
    base_model_ranking[model_names[0]] = new_model_rank+ranking[0]-1
    base_model_ranking[model_names[1]] = new_model_rank+ranking[1]-1
    base_model_ranking[model_names[2]] = new_model_rank+ranking[2]-1
    return base_model_ranking

def judge_bubble(sort_model_index_map,min_rank_idx,base_model_list,ranking,new_model_rank):
    flag_bubble = False
    if len(sort_model_index_map) == 2 and min_rank_idx == 1:
        if ranking[1] == 1:
            flag_bubble=-1
    if len(sort_model_index_map) == 2 and min_rank_idx == len(base_model_list)-1:
        if ranking[1] == 0:
            flag_bubble=1
    if len(sort_model_index_map) == 3 and ranking[2] != 1:
        if new_model_rank != 1 and new_model_rank != len(base_model_list):
            if ranking[2] == 0:
                flag_bubble=1
            if ranking[2] == 2:
                flag_bubble=-1
    return flag_bubble

def judge_continue_bubble(sort_model_index_map,base_model_list,ranking,new_model_rank):
    flag_bubble = False
    if len(sort_model_index_map) == 3 and ranking[2] != 1:
        if new_model_rank != 1 and new_model_rank != len(base_model_list):
            if ranking[2] == 0:
                flag_bubble=1
            if ranking[2] == 2:
                flag_bubble=-1
    return flag_bubble

# 用来排名base model，第一步base model先进行full sample
def topmodel_judge(base_dir,judge_model,sort_model_list,valid_question_ids=overall_ids):
    sort_model_index_map = {name: idx for idx, name in enumerate(sort_model_list)}
    jsonl_path = os.path.join(base_dir, judge_model, "voting_records.jsonl")
    comparison_list = [0] * len(sort_model_index_map)
    # print(jsonl_path)
    # print(comparison_list)
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r') as file:
            for line in file:
                record = json.loads(line)
                for each in record:
                    if valid_question_ids and each.get('question_id') not in valid_question_ids:
                        continue

                    model1 = each['response_A']
                    model2 = each['response_B']
                    winner = each['Won']

                    idx1 = sort_model_index_map.get(model1)
                    idx2 = sort_model_index_map.get(model2)

                    if idx1 is not None and idx2 is not None:
                        if winner == model1:
                            comparison_list[idx1] += 1
                        elif winner == model2:
                            comparison_list[idx2] += 1
                        else:
                            comparison_list[idx1] += 0
                            comparison_list[idx2] += 0
    print(comparison_list) 
    # 计算行的和
    total_sum = sum(comparison_list)
    # 进行除法运算
    comparison_list = [x / total_sum for x in comparison_list]
    # 打印结果
    print(comparison_list)
    return comparison_list

def generate_ranking_list(original_list, ranked_list):
    # 创建模型到排名的映射字典
    ranking_dict = {model: rank for rank, model in enumerate(ranked_list)}
    # print(original_list)
    # print(ranking_dict)
    # 使用映射字典生成原始列表的排名列表
    ranking_list = [ranking_dict.get(model, -1) for model in original_list]
    
    return ranking_list

def is_zero_matrix(matrix):
    for row in matrix:
        for element in row:
            if element != 0:
                return False
    return True

def question_select(base_dir,models_to_sort,ori_base_model_list,ori_base_model_ranking,model_weights,judge_model_list,valid_question_ids,beg,end,ori_model_names):
    ranking_list = list()
    for question in tqdm(valid_question_ids[beg:end]): 
        print(question)
        flag_zero = False
        question = [question]
        spearman_rank_correlation_list = list()
        final_corr = list()
        rugged_rank_diff = list()

        base_model_list = copy.deepcopy(ori_base_model_list)
        base_model_ranking = copy.deepcopy(ori_base_model_ranking)

        original_model_list = copy.deepcopy(base_model_list)
        spearman_rank_correlation_list.append([])
        rugged_rank_diff.append([])
        # 初始化weights
        initial_weight = 1
        model_weights = update_model_weight(initial_weight,base_model_ranking,judge_model_list)
        # print(model_weights)
        for new_model in models_to_sort:
            # 粗粒度排名
            sort_rank = rugged_rank(base_dir,new_model,base_model_list,base_model_ranking,model_weights,judge_model_list,question)
            decrement = 0.7 / (len(base_model_ranking)-1)
            model_weights[new_model] = max(initial_weight - (sort_rank - 1) * decrement,1.0)
            # print(new_model,sort_rank)
            # 细粒度排名
            sort_model_index_map, min_rank_idx, vote_matrix, weights= full_comparsion(base_dir,new_model,base_model_list,sort_rank,model_weights,judge_model_list,valid_question_ids=question)
            if is_zero_matrix(vote_matrix):
                flag_zero = True 
                break
            ranking = vote_to_rank(vote_matrix,weights,len(sort_model_index_map))
            if len(sort_model_index_map) == 1:
                base_model_ranking[new_model] = len(vote_matrix)
            else:
                base_model_ranking[new_model] = 10000

            gt_scores_list = [gt_scores[model] for model in base_model_list]
            new_model_gt_ranks = rank_scores(gt_scores_list)[-1]
            base_model_ranking = integrate_rankings(base_model_ranking,sort_model_index_map,ranking)
            base_model_list = sorted(base_model_ranking, key=base_model_ranking.get)
            model_weights = update_model_weight(initial_weight,base_model_ranking,judge_model_list)
                        
            # 判断是否进行bubble window
            new_model_rank = base_model_ranking[new_model]
            flag_bubble = judge_bubble(sort_model_index_map,min_rank_idx,base_model_list,ranking,new_model_rank)
            while flag_bubble:
                # print(1)
                model_names, sort_model_index_map, vote_matrix,weights= bubble_window(base_dir,new_model,base_model_list,new_model_rank,model_weights,judge_model_list,valid_question_ids=question)
                # print(vote_matrix)
                if is_zero_matrix(vote_matrix):
                    flag_zero = True 
                    break
                ranking = vote_to_rank(vote_matrix,weights,len(sort_model_index_map))
                # 更新排名
                base_model_ranking = update_bubble_window_rank(base_model_ranking,model_names,new_model_rank,ranking)
                base_model_list = sorted(base_model_ranking, key=base_model_ranking.get)
                model_weights = update_model_weight(initial_weight,base_model_ranking,judge_model_list)
                new_model_rank = base_model_ranking[new_model]
                # 判断是否continue
                if flag_bubble != judge_continue_bubble(sort_model_index_map,base_model_list,ranking,new_model_rank):
                    flag_bubble = False
            # print(base_model_list)
        ranking = list()
        if flag_zero == True:
            ranking_list.append([100]*11)
            # print(ranking_list)
            continue
        for i in base_model_list:
            ranking.append(base_model_ranking[i])
        gt_scores_list = [gt_scores[model] for model in base_model_list]
        gt_ranks = rank_scores(gt_scores_list)
        spearman_rank_correlation, _ = spearmanr(ranking, gt_ranks)

        ranking_list.append(generate_ranking_list(ori_model_names, base_model_list))
    return ranking_list

def main(base_dir="/data/shared/decentralized_arena/judgements_mt_bench_kun/judgements_mt_bench",valid_question_ids=overall_ids,existing_model_paths=existing_model_paths,gt_scores=gt_scores):

    model_list = list(existing_model_paths.keys())
    # print(valid_question_ids)
    # ori_base_model_list全部换成开源模型
    ori_base_model_list = ["athene-70b","gemma-2-27b-it","gemma-2-9b-it-simpo","qwen2-72b-instruct",'command-r-(08-2024)']
    ori_base_model_ranking = dict()
    for rank, model in enumerate(ori_base_model_list):
        ori_base_model_ranking[model] = rank+1
    initial_weight = 1
    model_weights = update_model_weight(initial_weight,ori_base_model_ranking,judge_model_list)
    valid_question_ids_list = list()
    # sample出来的小规模model lists
    filtered_models_list = ['google-gemma-2-9b-it',"yi-1.5-34b-chat",'mistral-8x7b-instruct-v0.1',"qwen1.5-14b-chat",'mistral-7b-instruct-2',"starling-lm-7b-alpha"]
    # 创建三个打乱的列表
    shuffled_lists = []
    for _ in range(1):
        shuffled_list = filtered_models_list[:]  # 复制原始列表
        random.shuffle(shuffled_list)      # 打乱复制的列表
        shuffled_lists.append(shuffled_list)
    print(shuffled_lists)
    scores = np.zeros(len(valid_question_ids))
    for cnt in range(1):
        models_to_sort = shuffled_lists[cnt]
        ori_model_names = ori_base_model_list+models_to_sort
        ranking_list = list()
        # print(models_to_sort)
        # 每个问题排一遍序
        paras = list()
        for i in range(40):
            beg = i*2
            end = beg+2
            paras.append([base_dir,models_to_sort,ori_base_model_list,ori_base_model_ranking,model_weights,judge_model_list,valid_question_ids,beg,end,ori_model_names])
        with Pool(processes=(cpu_count() - 1)) as pool:
            result = pool.starmap(question_select, paras)
        print(result)
        for i in result:
            ranking_list += i
        print(ranking_list)
        # 计算模型的平均排名
        # 过滤掉包含100的行
        filtered_ranking_list = list()
        for i in ranking_list:
            if 100 in i:
                continue
            else:
                filtered_ranking_list += [i]
        print(filtered_ranking_list)
        # 计算均值
        mean_ranking = np.mean(filtered_ranking_list, axis=0)
        # mean_ranking = np.mean(ranking_list, axis=0)
        print(mean_ranking)
        # 计算 L2 范数
        l2_norms = np.linalg.norm(ranking_list - mean_ranking, axis=1)
        # print(l2_norms)
        scores += l2_norms #一个array，统计总分
        print(scores)
    top_indices = np.argsort(scores)[:100]
    # print(top_indices)
    valid_question_ids_list.append([valid_question_ids[i] for i in top_indices])
    print(valid_question_ids_list)
    
    
if __name__ == "__main__":
    fire.Fire(main)