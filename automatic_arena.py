import json
import numpy as np
import copy
import tqdm
import csv
from utils_final import existing_model_paths
from multiprocessing import Pool, cpu_count
import fire
import random
import uuid
import scipy.stats
from scipy.optimize import minimize
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
from judge_responses_new import get_question_with_reference, judge_prompt_pair_reference, judge_prompt_pairwise, \
    fetch_responses, determine_winner, load_records
from sklearn.linear_model import LogisticRegression
import pandas as pd
import math
 
# 假设df是你的DataFrame
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

judge_open_model = []
judge_api_model = ['o1-mini', 'o1-preview', 'ChatGPT-4o-latest', 'gpt-4o-2024-05-13', 'gpt-4o-2024-08-06', 'gpt-4-1106-preview', 'gpt-4-turbo-2024-04-09', 'gpt-3.5-turbo-0125', 'gpt-4o-mini-2024-07-18']
judge_model_list = ['o1-mini', 'o1-preview', 'ChatGPT-4o-latest', 'gpt-4o-2024-05-13', 'gpt-4o-2024-08-06', 'gpt-4-1106-preview', 'gpt-4-turbo-2024-04-09', 'gpt-3.5-turbo-0125', 'gpt-4o-mini-2024-07-18']
overall_ids = [i for i in range(81,102)]+[i for i in range(103,121)]
save_output_file_path = 'mt_bench ranking result.txt'

def rank_scores(scores):
    indexed_scores = list(enumerate(scores))
    sorted_scores = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
    ranks = [0] * len(scores)
    for rank, (index, _) in enumerate(sorted_scores):
        ranks[index] = rank
    return ranks

def save_to_jsonl(data, filename):
    """Saves a Python data structure to a .jsonl file."""
    with open(filename, 'w') as f:
        f.write(json.dumps(data) + '\n')

def update_voting_records(model, response_A_name, response_B_name, won, question_id, data_id):
    """Updates the voting records with a new voting result."""
    records_path = f"/home/yanbin/De-Arena/judgements_mt_bench/{model}/voting_records.jsonl"
    # Ensure the directory exists
    os.makedirs(os.path.dirname(records_path), exist_ok=True)

    # Load existing records or create an empty list if the file does not exist
    try:
        records = load_records(records_path)[0]
    except:
        records = []

    # Append a new record to the list of records
    new_record = {
        "response_A": response_A_name,
        "response_B": response_B_name,
        "Won": won,
        "question_id": question_id,
        "data_id": data_id
    }
    records.append(new_record)  # Ensure this is a flat append operation

    # Save updated records back to the JSONL file
    save_to_jsonl(records, records_path)

def run_judging_trials(judge_model, model_name, path="/home/yanbin/De-Arena/mt_bench_questions.jsonl", openai_api = "", tensor_parallel_size=1):
    # print(judge_model,model_name)
    model_index_map = {name: idx for idx, name in enumerate(model_name)}
    initial_question_ids = overall_ids
    responses_dict = dict()
    # Fetch responses for each model
    for model in model_name:
        responses_dict[model] = fetch_responses("/home/yanbin/De-Arena/mt_bench_responses", model)
    # print(responses_dict)
    combination_models = list(itertools.combinations(model_name, 2))

    # Iterate over combinations of model pairs for comparison
    for model_a, model_b in tqdm(combination_models):
        responses_a = responses_dict[model_a]
        responses_b = responses_dict[model_b]

        batch_size = 40  # Set batch size for processing
        num_batches = (len(initial_question_ids) + batch_size - 1) // batch_size  # Calculate the number of batches

        for batch_idx in tqdm(range(num_batches)):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(initial_question_ids))
            prompts = list()
            swapped_prompts = list()
            question_ids = list()

            # Create prompts and swapped prompts for comparison
            for idx in range(start_idx, end_idx):
                question_id = initial_question_ids[idx]

                question, reference = get_question_with_reference(path, question_id)
                response_a = responses_a[idx]['response']
                response_b = responses_b[idx]['response']
                if reference != "":
                    prompt = judge_prompt_pair_reference(question, response_a, response_b, reference)
                    swapped_prompt = judge_prompt_pair_reference(question, response_b, response_a, reference)
                else:
                    prompt = judge_prompt_pairwise(question, response_a, response_b)
                    swapped_prompt = judge_prompt_pairwise(question, response_b, response_a)
                    # print(prompt)
                # breakpoint()
                prompts.append(prompt)
                swapped_prompts.append(swapped_prompt)
                question_ids.append(question_id)
            try:
                # Adjust logic based on the type of judge_model
                if 'gpt' in judge_model or 'GPT' in judge_model or 'o1' in judge_model:  # For OpenAI models
                    judge_responses = run_openai_model(openai_api, prompts, judge_model)
                    swapped_judge_responses = run_openai_model(openai_api, swapped_prompts, judge_model)
                elif "gemini" in judge_model:  # For Gemini models
                    judge_responses = run_gemini_model(prompts, judge_model)
                    swapped_judge_responses = run_gemini_model(swapped_prompts, judge_model)

            except Exception as e:
                print(f"Error evaluating model pair ({model_a}, {model_b}) with judge {judge_model}: {e}")
                continue  # Skip to the next model pair if there's an error

            cnt = 0
            # Process responses and determine winners
            for response, swapped_response in zip(judge_responses, swapped_judge_responses):
                winner = determine_winner(response, model_a, model_b)
                swapped_winner = determine_winner(swapped_response, model_b, model_a)
                final_winner = winner if winner == swapped_winner else "TIE"
                data_id = str(uuid.uuid4())
                update_voting_records(judge_model, model_a, model_b, final_winner, question_ids[cnt], data_id)
                cnt += 1

def run_openai_model(openai_api, prompts, model_name, max_tokens=15):
    # Handle model selection for OpenAI models
    if "3.5-turbo-0125" in model_name:
        model_name = "gpt-3.5-turbo-0125"
        client = OpenAI(api_key=openai_api)
    elif "gpt-4o-mini" in model_name:
        model_name = "gpt-4o-mini-2024-07-18"
        client = OpenAI(api_key=openai_api)
    elif "gpt-4o-2024-05-13" in model_name:
        model_name = "gpt-4o-2024-05-13"
        client = OpenAI(api_key=openai_api)
    elif "ChatGPT-4o-latest" in model_name:
        model_name = "chatgpt-4o-latest"
        client = OpenAI(api_key=openai_api)
    elif "gpt-4o-2024-08-06" in model_name:
        model_name = "gpt-4o-2024-08-06"
        client = OpenAI(api_key=openai_api)
    elif "o1-mini" in model_name:
        model_name = "o1-mini"
        client = OpenAI(api_key=openai_api)
    elif "o1-preview" in model_name:
        model_name = "o1-preview"
        client = OpenAI(api_key=openai_api)

    responses = []
    # Modify each prompt to ask the model to evaluate dataset quality
    for prompt in prompts:
        # Call OpenAI API with the modified quality evaluation prompt
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=max_tokens
        )

        # Extract and store the response
        text = completion.choices[0].message.content
        responses.append(str(text))

    return responses

def run_gemini_model(prompts, model_name="gemini-1.5-flash", max_tokens=3):
    if model_name == "gemini-1.5-pro-001":
        model_name = "gemini-1.5-pro-001"
        client = ''
    elif model_name == "gemini-1.0-pro-001":
        model_name = "gemini-1.0-pro-001"
        client = ""
    elif model_name == "gemini-1.5-flash-001":
        model_name = "gemini-1.5-flash-001"
        client = ""

    responses = []
    genai.configure(api_key=client)
    model = genai.GenerativeModel(model_name)
    for prompt in prompts:
        cnt = 0
        while 1:
            cnt += 1
            if cnt > 5:
                responses.append("")
                break
            try:
                message = model.generate_content(
                    prompt
                )
                response_text = message.text
                responses.append(response_text)
                break
            except Exception as e:
                print(f"Error : {e}")
                time.sleep(5)
                continue

    return responses

def rugged_rank(base_dir, new_model, base_model_list, base_model_ranking, model_weights, judge_model_list, judge_model_states, valid_question_ids=overall_ids):
    final_binary_judge_dict = dict()
    rank_list = list()
    weight_list = list()
    total_weight = 0
    weighted_rank_sum = 0
    judge_models = [model for model in base_model_list if model in judge_model_list]
    print(judge_models)

    paras = list()
    for judge_model in judge_models:
        models_to_sort = [model for model in base_model_list if model != judge_model]
        paras.append(
            [base_dir, judge_model, new_model, models_to_sort, valid_question_ids, base_model_ranking, model_weights])
    with Pool(processes=(cpu_count() - 1)) as pool:
        result = pool.starmap(binary_search, paras)
    ranking = list()
    for i in range(len(result)):
        ranking.append((result[i][0],result[i][2]))
        weighted_rank_sum += result[i][0] * result[i][1]
        total_weight += result[i][1]
        judge_model_states[result[i][2]] += result[i][3]
        for key,value in result[i][4].items():
            battle_id = len(final_binary_judge_dict)
            final_binary_judge_dict[battle_id] = value
    print(ranking)

    weighted_average_rank = weighted_rank_sum / total_weight
    return weighted_average_rank,judge_model_states,final_binary_judge_dict

def binary_search(base_dir, judge_model, new_model, models_to_sort, valid_question_ids, base_model_ranking,
                  model_weights):
    binary_judge_dict = dict()
    model_pair_list = list()
    left, right = 0, len(models_to_sort)
    # print(models_to_sort)
    while left < right:
        mid = (left + right) // 2
        model_pair_list.append((new_model, models_to_sort[mid]))
        vote_diff, tmp_judge_dict = get_vote_result_for_judge(base_dir, judge_model, new_model, models_to_sort[mid], valid_question_ids)
        for key,value in tmp_judge_dict.items():
            battle_id = len(binary_judge_dict)
            binary_judge_dict[battle_id] = value
        if vote_diff <= 0:
            left = mid + 1
        else:
            right = mid
    if left == 0:
        rank = 1
    else:
        rank = base_model_ranking[models_to_sort[left - 1]] + 1

    if rank == base_model_ranking[judge_model]:
        rank = base_model_ranking[judge_model] + 1
    weight = model_weights.get(judge_model, 0.3)
    print(rank,judge_model)
    return rank, weight, judge_model, model_pair_list, binary_judge_dict

def get_vote_result_for_judge(base_dir, judge_model, model1, model2, valid_question_ids=overall_ids):
    tmp_judge_dict = dict()
    vote_diff = 0
    jsonl_path = os.path.join(base_dir, judge_model, "voting_records.jsonl")
    if judge_model in judge_open_model:
        jsonl_path = os.path.join(base_dir, judge_model, "voting_records.jsonl")
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)
        if not os.path.exists(jsonl_path):
            with open(jsonl_path, 'w') as file:
                pass  
        if os.path.exists(jsonl_path):
            with open(jsonl_path, 'r') as file:
                for line in file:
                    # print(jsonl_path)
                    record = json.loads(line)
                    for each in record:
                        if valid_question_ids and each.get('question_id') not in valid_question_ids:
                            continue
                        if each['response_A'] == model1 and each['response_B'] == model2:
                            battle_id = len(tmp_judge_dict)
                            response_A = fetch_responses("/home/yanbin/De-Arena/mt_bench_responses", model1)
                            response_B = fetch_responses("/home/yanbin/De-Arena/mt_bench_responses", model2)
                            # print(response_A)
                            question_id = each.get('question_id')
                            response_a = next((item['response'] for item in response_A if item['question_id'] == question_id), None)
                            response_b = next((item['response'] for item in response_B if item['question_id'] == question_id), None)
                            dict_a = count_markdown_elements(response_a, '_a')
                            dict_b = count_markdown_elements(response_b, '_b')
                            metadata_dict = {**dict_a, **dict_b}
                            metadata_dict["sum_assistant_a_length"] = len(response_a)
                            metadata_dict["sum_assistant_b_length"] = len(response_b) 
                            tmp_judge_dict[battle_id] = {"judge_model":judge_model,"model_A":model1,"model_B":model2,"question_id":each.get('question_id'),"winner":each['Won'],"metadata":metadata_dict}
                            # print(each)
                            if each['Won'] == model1:
                                vote_diff += 1
                            elif each['Won'] == model2:
                                vote_diff -= 1
                        elif each['response_A'] == model2 and each['response_B'] == model1:
                            battle_id = len(tmp_judge_dict)
                            response_A = fetch_responses("/home/yanbin/De-Arena/mt_bench_responses", model1)
                            response_B = fetch_responses("/home/yanbin/De-Arena/mt_bench_responses", model2)
                            # print(response_A)
                            question_id = each.get('question_id')
                            response_a = next((item['response'] for item in response_A if item['question_id'] == question_id), None)
                            response_b = next((item['response'] for item in response_B if item['question_id'] == question_id), None)
                            dict_a = count_markdown_elements(response_a, '_a')
                            dict_b = count_markdown_elements(response_b, '_b')
                            metadata_dict = {**dict_a, **dict_b}
                            metadata_dict["sum_assistant_a_length"] = len(response_a)
                            metadata_dict["sum_assistant_b_length"] = len(response_b) 
                            tmp_judge_dict[battle_id] = {"judge_model":judge_model,"model_A":model1,"model_B":model2,"question_id":each.get('question_id'),"winner":each['Won'],"metadata":metadata_dict}
                            # print(each)
                            if each['Won'] == model2:
                                vote_diff -= 1
                            elif each['Won'] == model1:
                                vote_diff += 1
 
    elif judge_model in judge_api_model:
        print("---------")
        print(judge_model)
        print("---------")
        if not os.path.exists(jsonl_path):
            directory = os.path.join(base_dir, judge_model)
            if not os.path.exists(directory):
                os.makedirs(directory)
            if not os.path.exists(jsonl_path):
                with open(jsonl_path, 'w') as f:
                    pass
        if os.path.exists(jsonl_path):
            with open(jsonl_path, 'r') as file:
                flag = False
                for line in file:
                    # print(jsonl_path)
                    record = json.loads(line)
                    for each in record:
                        if valid_question_ids and each.get('question_id') not in valid_question_ids:
                            continue
                        if each['response_A'] == model1 and each['response_B'] == model2:
                            flag = True
                            battle_id = len(tmp_judge_dict)
                            response_A = fetch_responses("/home/yanbin/De-Arena/mt_bench_responses", model1)
                            response_B = fetch_responses("/home/yanbin/De-Arena/mt_bench_responses", model2)
                            # print(response_A)
                            question_id = each.get('question_id')
                            response_a = next((item['response'] for item in response_A if item['question_id'] == question_id), None)
                            response_b = next((item['response'] for item in response_B if item['question_id'] == question_id), None)
                            dict_a = count_markdown_elements(response_a, '_a')
                            dict_b = count_markdown_elements(response_b, '_b')
                            metadata_dict = {**dict_a, **dict_b}
                            metadata_dict["sum_assistant_a_length"] = len(response_a)
                            metadata_dict["sum_assistant_b_length"] = len(response_b) 
                            tmp_judge_dict[battle_id] = {"judge_model":judge_model,"model_A":model1,"model_B":model2,"question_id":each.get('question_id'),"winner":each['Won'],"metadata":metadata_dict}
                            # print(each)
                            if each['Won'] == model1:
                                vote_diff += 1
                            elif each['Won'] == model2:
                                vote_diff -= 1
                        elif each['response_A'] == model2 and each['response_B'] == model1:
                            flag = True
                            # print(each)
                            battle_id = len(tmp_judge_dict)
                            response_A = fetch_responses("/home/yanbin/De-Arena/mt_bench_responses", model1)
                            response_B = fetch_responses("/home/yanbin/De-Arena/mt_bench_responses", model2)
                            # print(response_A)
                            question_id = each.get('question_id')
                            response_a = next((item['response'] for item in response_A if item['question_id'] == question_id), None)
                            response_b = next((item['response'] for item in response_B if item['question_id'] == question_id), None)
                            dict_a = count_markdown_elements(response_a, '_a')
                            dict_b = count_markdown_elements(response_b, '_b')
                            metadata_dict = {**dict_a, **dict_b}
                            metadata_dict["sum_assistant_a_length"] = len(response_a)
                            metadata_dict["sum_assistant_b_length"] = len(response_b) 
                            tmp_judge_dict[battle_id] = {"judge_model":judge_model,"model_A":model1,"model_B":model2,"question_id":each.get('question_id'),"winner":each['Won'],"metadata":metadata_dict}
                            if each['Won'] == model2:
                                vote_diff -= 1
                            elif each['Won'] == model1:
                                vote_diff += 1
            # print(vote_diff)
            # 调用API
            if flag == False:
                with open(jsonl_path, 'r') as file:
                    # print(judge_model)
                    if 'gpt' in judge_model or "GPT" in judge_model or "o1" in judge_model:
                        # print(judge_model)
                        run_judging_trials(judge_model, [model1, model2])
                        for line in file:
                            # print(jsonl_path)
                            record = json.loads(line)
                            for each in record:
                                if valid_question_ids and each.get('question_id') not in valid_question_ids:
                                    continue
                                if each['response_A'] == model1 and each['response_B'] == model2:
                                    flag = True
                                    battle_id = len(tmp_judge_dict)
                                    response_A = fetch_responses("/home/yanbin/De-Arena/mt_bench_responses", model1)
                                    response_B = fetch_responses("/home/yanbin/De-Arena/mt_bench_responses", model2)
                                    # print(response_A)
                                    question_id = each.get('question_id')
                                    response_a = next((item['response'] for item in response_A if item['question_id'] == question_id), None)
                                    response_b = next((item['response'] for item in response_B if item['question_id'] == question_id), None)
                                    dict_a = count_markdown_elements(response_a, '_a')
                                    dict_b = count_markdown_elements(response_b, '_b')
                                    metadata_dict = {**dict_a, **dict_b}
                                    metadata_dict["sum_assistant_a_length"] = len(response_a)
                                    metadata_dict["sum_assistant_b_length"] = len(response_b) 
                                    tmp_judge_dict[battle_id] = {"judge_model":judge_model,"model_A":model1,"model_B":model2,"question_id":each.get('question_id'),"winner":each['Won'],"metadata":metadata_dict}
                                    # print(each)
                                    if each['Won'] == model1:
                                        vote_diff += 1
                                    elif each['Won'] == model2:
                                        vote_diff -= 1
                                elif each['response_A'] == model2 and each['response_B'] == model1:
                                    flag = True
                                    # print(each)
                                    battle_id = len(tmp_judge_dict)
                                    response_A = fetch_responses("/home/yanbin/De-Arena/mt_bench_responses", model1)
                                    response_B = fetch_responses("/home/yanbin/De-Arena/mt_bench_responses", model2)
                                    # print(response_A)
                                    question_id = each.get('question_id')
                                    response_a = next((item['response'] for item in response_A if item['question_id'] == question_id), None)
                                    response_b = next((item['response'] for item in response_B if item['question_id'] == question_id), None)
                                    dict_a = count_markdown_elements(response_a, '_a')
                                    dict_b = count_markdown_elements(response_b, '_b')
                                    metadata_dict = {**dict_a, **dict_b}
                                    metadata_dict["sum_assistant_a_length"] = len(response_a)
                                    metadata_dict["sum_assistant_b_length"] = len(response_b) 
                                    tmp_judge_dict[battle_id] = {"judge_model":judge_model,"model_A":model1,"model_B":model2,"question_id":each.get('question_id'),"winner":each['Won'],"metadata":metadata_dict}
                                    if each['Won'] == model2:
                                        vote_diff -= 1
                                    elif each['Won'] == model1:
                                        vote_diff += 1
                        print(vote_diff)

        print(judge_model)

    return vote_diff, tmp_judge_dict

def integrate_rankings(original_ranking, new_ranking, relative_ranking):
    # print(new_ranking.keys())
    if len(new_ranking) == 1:
        final_ranking = original_ranking.copy()
        final_ranking[next(iter(new_ranking))] = len(final_ranking)
        return final_ranking
    # 找到new_ranking中对应模型在original_ranking中的最小排名
    min_new_rank = min(original_ranking[model] for model in new_ranking)

    # 创建一个字典来存储最终的整体排名，先复制原始排名
    final_ranking = original_ranking.copy()

    max_ranking = max(relative_ranking) + min_new_rank
    for model, rank in final_ranking.items():
        if rank >= max_ranking:
            final_ranking[model] += 1

    # 遍历new_ranking中的每个模型
    for model, idx in new_ranking.items():
        # 获取该模型在relative_ranking中的排名
        new_relative_rank = relative_ranking[idx]
        # 更新final_ranking中的排名，新排名加上new_ranking中的最小排名减一
        final_ranking[model] = new_relative_rank + min_new_rank

    return final_ranking

def full_comparsion(base_dir, new_model, base_model_list, sort_rank, model_weights, judge_model_list, judge_model_states, window=1,
                    valid_question_ids=overall_ids):
    bottle_judge_dict=dict()
    rank_idx = int(sort_rank)
    min_rank_idx = max(1, rank_idx - window + 1)
    max_rank_idx = min(len(base_model_list), rank_idx + window)
    model_names = list()
    # print(min_rank_idx,max_rank_idx)
    for i in range(min_rank_idx - 1, max_rank_idx):
        model_names.append(base_model_list[i])
    model_names.append(new_model)
    print(model_names)
    combinations = list(itertools.combinations(model_names, 2))
    remaining_combinations = set(combinations)
    base_model_list.append(new_model)
    # judge_models = [model for model in base_model_list if model in judge_model_list]
    judge_models = [model for model in base_model_list if model in judge_model_list and model not in model_names]
    for i in judge_models:
        judge_model_states[i] += combinations
    # print(judge_models)
    sort_model_index_map = {name: idx for idx, name in enumerate(model_names)}
    judge_model_index_map = {name: idx for idx, name in enumerate(judge_models)}
    # Initialize an empty comparison matrix
    print(judge_model_index_map)
    # final_comparison_matrix = np.zeros((len(judge_models), len(judge_models)))
    final_comparison_matrix = np.zeros((len(judge_models), len(model_names)))
    weights = list()

    paras = list()
    for subdir in os.listdir(base_dir):
        # print(subdir)
        if subdir not in base_model_list:
            continue
        if subdir not in judge_models:
            continue
        if subdir not in judge_model_list:
            continue
        # comparison_matrix = np.zeros((len(judge_models), len(judge_models)))
        comparison_matrix = np.zeros((len(judge_models), len(model_names)))
        paras.append([base_dir, subdir, model_weights, sort_model_index_map, judge_model_index_map, comparison_matrix,
                      remaining_combinations, valid_question_ids])
    with Pool(processes=(cpu_count() - 1)) as pool:
        result = pool.starmap(pairwise_judge, paras)
    # print(result)
    # print(comparison_matrix)
    for i in range(len(result)):
        final_comparison_matrix += result[i][0]
        for key,value in result[i][1].items():
            battle_id = len(bottle_judge_dict)
            bottle_judge_dict[battle_id] = value
    print(final_comparison_matrix)
    # 计算每一行的和
    row_sums = final_comparison_matrix.sum(axis=1, keepdims=True)

    # 创建归一化矩阵，初始化为原始矩阵
    normalized_matrix = np.copy(final_comparison_matrix)

    # 对每一行进行归一化（忽略和为0的行）
    nonzero_row_indices = row_sums.flatten() != 0
    normalized_matrix[nonzero_row_indices] = final_comparison_matrix[nonzero_row_indices] / row_sums[
        nonzero_row_indices]
    return sort_model_index_map, min_rank_idx, normalized_matrix, model_names, judge_model_states, bottle_judge_dict

def bubble_window(base_dir, new_model, base_model_list, new_model_rank, model_weights, judge_model_list,judge_model_states,window=1,
                  valid_question_ids=overall_ids):
    bottle_judge_dict=dict()
    # print(new_model_rank,base_model_list)
    model_names = list()
    model_names.append(base_model_list[new_model_rank - 2])
    model_names.append(base_model_list[new_model_rank])
    model_names.append(base_model_list[new_model_rank - 1])
    combinations = list(itertools.combinations(model_names, 2))
    remaining_combinations = set(combinations)
    # print(model_names)
    # judge_models = [model for model in base_model_list if model in judge_model_list]
    judge_models = [model for model in base_model_list if model in judge_model_list and model not in model_names]
    for i in judge_models:
        judge_model_states[i] += combinations
    sort_model_index_map = {name: idx for idx, name in enumerate(model_names)}
    judge_model_index_map = {name: idx for idx, name in enumerate(judge_models)}
    print(judge_model_index_map)
    # Initialize an empty comparison matrix
    # final_comparison_matrix = np.zeros((len(judge_models), len(judge_models)))
    final_comparison_matrix = np.zeros((len(judge_models), len(model_names)))
    weights = list()

    paras = list()
    for subdir in os.listdir(base_dir):
        # print(subdir)
        if subdir not in base_model_list:
            continue
        if subdir not in judge_models:
            continue
        if subdir not in judge_model_list:
            continue
        # comparison_matrix = np.zeros((len(judge_models), len(judge_models)))
        comparison_matrix = np.zeros((len(judge_models), len(model_names)))
        paras.append([base_dir, subdir, model_weights, sort_model_index_map, judge_model_index_map, comparison_matrix,
                      remaining_combinations, valid_question_ids])
    with Pool(processes=(cpu_count() - 1)) as pool:
        result = pool.starmap(pairwise_judge, paras)
    # print(result)
    # print(comparison_matrix)
    for i in range(len(result)):
        final_comparison_matrix += result[i][0]
        for key,value in result[i][1].items():
            battle_id = len(bottle_judge_dict)
            bottle_judge_dict[battle_id] = value
    print(final_comparison_matrix)
    # 计算每一行的和
    row_sums = final_comparison_matrix.sum(axis=1, keepdims=True)

    # 创建归一化矩阵，初始化为原始矩阵
    normalized_matrix = np.copy(final_comparison_matrix)

    # 对每一行进行归一化（忽略和为0的行）
    nonzero_row_indices = row_sums.flatten() != 0
    normalized_matrix[nonzero_row_indices] = final_comparison_matrix[nonzero_row_indices] / row_sums[
        nonzero_row_indices]
    # print(comparison_matrix)
    return model_names, sort_model_index_map, normalized_matrix, model_names, judge_model_states, bottle_judge_dict

def pairwise_judge(base_dir, subdir, model_weights, sort_model_index_map, judge_model_index_map, comparison_matrix,
                   remaining_combinations, valid_question_ids):
    tmp_judge_dict = dict()
    print(subdir)
    if subdir in judge_open_model:
        jsonl_path = os.path.join(base_dir, subdir, "voting_records.jsonl")
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)
        if not os.path.exists(jsonl_path):
            with open(jsonl_path, 'w') as file:
                pass  
        if os.path.exists(jsonl_path):
            with open(jsonl_path, 'r') as file:
                for line in file:
                    record = json.loads(line, strict=False)
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
                            battle_id = len(tmp_judge_dict)
                            response_A = fetch_responses("/home/yanbin/De-Arena/mt_bench_responses", model1)
                            response_B = fetch_responses("/home/yanbin/De-Arena/mt_bench_responses", model2)
                            question_id = each.get('question_id')
                            response_a = next((item['response'] for item in response_A if item['question_id'] == question_id), None)
                            response_b = next((item['response'] for item in response_B if item['question_id'] == question_id), None)
                            dict_a = count_markdown_elements(response_a, '_a')
                            dict_b = count_markdown_elements(response_b, '_b')
                            metadata_dict = {**dict_a, **dict_b}
                            metadata_dict["sum_assistant_a_length"] = len(response_a)
                            metadata_dict["sum_assistant_b_length"] = len(response_b) 
                            tmp_judge_dict[battle_id] = {"judge_model":subdir,"model_A":model1,"model_B":model2,"question_id":each.get('question_id'),"winner":winner,"metadata":metadata_dict}
                            if winner == model1:
                                comparison_matrix[judge_idx, idx1] += 1
                            elif winner == model2:
                                comparison_matrix[judge_idx, idx2] += 1
                            else:
                                comparison_matrix[judge_idx, idx1] += 0
                                comparison_matrix[judge_idx, idx2] += 0

    elif subdir in judge_api_model:
        jsonl_path = os.path.join(base_dir, subdir, "voting_records.jsonl")
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)
        if not os.path.exists(jsonl_path):
            with open(jsonl_path, 'w') as file:
                pass
        if os.path.exists(jsonl_path):
            with open(jsonl_path, 'r') as file:
                flag = False
                for line in file:
                    record = json.loads(line, strict=False)
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
                            remaining_combinations.discard((model1, model2))
                            remaining_combinations.discard((model2, model1))
                            battle_id = len(tmp_judge_dict)
                            response_A = fetch_responses("/home/yanbin/De-Arena/mt_bench_responses", model1)
                            response_B = fetch_responses("/home/yanbin/De-Arena/mt_bench_responses", model2)
                            # print(response_A)
                            question_id = each.get('question_id')
                            response_a = next((item['response'] for item in response_A if item['question_id'] == question_id), None)
                            response_b = next((item['response'] for item in response_B if item['question_id'] == question_id), None)
                            dict_a = count_markdown_elements(response_a, '_a')
                            dict_b = count_markdown_elements(response_b, '_b')
                            metadata_dict = {**dict_a, **dict_b}
                            metadata_dict["sum_assistant_a_length"] = len(response_a)
                            metadata_dict["sum_assistant_b_length"] = len(response_b) 
                            tmp_judge_dict[battle_id] = {"judge_model":subdir,"model_A":model1,"model_B":model2,"question_id":each.get('question_id'),"winner":winner,"metadata":metadata_dict}
                            if winner == model1:
                                comparison_matrix[judge_idx, idx1] += 1
                            elif winner == model2:
                                comparison_matrix[judge_idx, idx2] += 1
                            else:
                                comparison_matrix[judge_idx, idx1] += 0
                                comparison_matrix[judge_idx, idx2] += 0
                # print(subdir,remaining_combinations)
                # 调用API
            with open(jsonl_path, 'r') as file:
                if len(remaining_combinations) != 0:
                    comparison_matrix = np.zeros_like(comparison_matrix)
                    judge_model = subdir
                    if 'gpt' in judge_model or 'GPT' in judge_model or 'o1' in judge_model:
                        for item in remaining_combinations:
                            run_judging_trials(judge_model, [item[0], item[1]])
                        for line in file:
                            record = json.loads(line, strict=False)
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
                                    flag = True
                                    battle_id = len(tmp_judge_dict)
                                    response_A = fetch_responses("/home/yanbin/De-Arena/mt_bench_responses", model1)
                                    response_B = fetch_responses("/home/yanbin/De-Arena/mt_bench_responses", model2)
                                    # print(response_A)
                                    question_id = each.get('question_id')
                                    response_a = next((item['response'] for item in response_A if item['question_id'] == question_id), None)
                                    response_b = next((item['response'] for item in response_B if item['question_id'] == question_id), None)
                                    dict_a = count_markdown_elements(response_a, '_a')
                                    dict_b = count_markdown_elements(response_b, '_b')
                                    metadata_dict = {**dict_a, **dict_b}
                                    metadata_dict["sum_assistant_a_length"] = len(response_a)
                                    metadata_dict["sum_assistant_b_length"] = len(response_b) 
                                    tmp_judge_dict[battle_id] = {"judge_model":subdir,"model_A":model1,"model_B":model2,"question_id":each.get('question_id'),"winner":winner,"metadata":metadata_dict}
                                    if winner == model1:
                                        comparison_matrix[judge_idx, idx1] += 1
                                    elif winner == model2:
                                        comparison_matrix[judge_idx, idx2] += 1
                                    else:
                                        comparison_matrix[judge_idx, idx1] += 0 
                                        comparison_matrix[judge_idx, idx2] += 0 
    return comparison_matrix,  tmp_judge_dict

def vote_to_rank(vote_matrix, weights, n):
    # 提取前三列
    first_three_columns = vote_matrix[:, :n]
    vote_sum = list()
    # 分别计算前三列的和
    for i in range(n):
        vote_sum.append(np.sum(first_three_columns[:, i] * weights))
        print(vote_sum)
        # vote_sum.append(np.sum(first_three_columns[:, i]))
        sorted_indices = sorted(range(len(vote_sum)), key=lambda i: (vote_sum[i] == 0, -vote_sum[i]))
        # print(sorted_indices)
        # 计算排序后的排名
    ranking = [0] * n
    for rank, index in enumerate(sorted_indices):
        ranking[index] = rank
        # print(ranking)
    return vote_sum, ranking

def update_model_weight(initial_weight, base_model_ranking, judge_model_list):
    return model_scores

def update_bubble_window_rank(base_model_ranking, model_names, new_model_rank, ranking):
    base_model_ranking[model_names[0]] = new_model_rank + ranking[0] - 1
    base_model_ranking[model_names[1]] = new_model_rank + ranking[1] - 1
    base_model_ranking[model_names[2]] = new_model_rank + ranking[2] - 1
    return base_model_ranking

def judge_bubble(sort_rank, new_model_rank, model_num):
    flag_bubble = False
    if new_model_rank == 1:
        return False
    if new_model_rank == model_num:
        return False
    # 向前bubble
    if new_model_rank < sort_rank:
        return 1
    # 向后bubble
    if new_model_rank >= sort_rank + 1:
        return -1
    
    return flag_bubble

def judge_continue_bubble(old_model_rank, new_model_rank, model_num):
    flag_bubble = False
    if new_model_rank == 1:
        return False
    if new_model_rank == model_num:
        return False
    if new_model_rank == old_model_rank:
        return False
    if new_model_rank < old_model_rank:
        return 1
    if new_model_rank > old_model_rank:
        return -1
    return flag_bubble

def get_final_avg_rank(final_model_list):
    # 用于存储模型排名信息的字典
    ranking_stats = {}
    # 遍历每种方法的排名
    for method_rank in final_model_list:
        for rank, model in enumerate(method_rank):
            if model not in ranking_stats:
                ranking_stats[model] = {
                    'total_rank': 0,
                    'count': 0,
                    'min_rank': float('inf'),
                    'max_rank': float('-inf')
                }

            # 更新总排名和计数（rank + 1 使其为 1-based ranking）
            ranking_stats[model]['total_rank'] += rank + 1
            ranking_stats[model]['count'] += 1

            # 更新最低和最高排名
            ranking_stats[model]['min_rank'] = min(ranking_stats[model]['min_rank'], rank + 1)
            ranking_stats[model]['max_rank'] = max(ranking_stats[model]['max_rank'], rank + 1)
    print(ranking_stats)
    # 计算平均排名并准备最终结果
    final_results = {}
    for model, stats in ranking_stats.items():
        average_rank = stats['total_rank'] / stats['count']
        final_results[model] = average_rank
    print(final_results)
    # 根据平均排名排序
    sorted_models = sorted(final_results.items(), key=lambda x: x[1])
    print(sorted_models)
    # 处理并列情况生成最终排名
    final_ranked_list = []
    last_rank = 0
    last_average = None
    for index, (model, average_rank) in enumerate(sorted_models):
        if last_average is not None and average_rank == last_average:
            final_ranked_list.append(last_rank)  # 并列情况，保持上一个排名
        else:
            last_rank = index + 1  # 1-based ranking
            final_ranked_list.append(last_rank)
        last_average = average_rank
    # 输出结果
    print("最终模型排名列表:", final_ranked_list)
    with open(save_output_file_path, 'a') as f:
        f.write(f"ranking_stats: {ranking_stats}\n")
        f.write(f"avg ranking: {sorted_models}\n")
        f.write(f"final_ranked_list: {final_ranked_list}\n")
    return sorted_models, final_ranked_list

# 用来排名base model，第一步base model先进行full sample
def base_model_judge(base_dir, base_model_list, valid_question_ids=overall_ids):
    judge_dict = dict()
    model_weights = {model: 1 for model in base_model_list}
    # print(judge_models)
    sort_model_index_map = {name: idx for idx, name in enumerate(base_model_list)}
    judge_model_index_map = {name: idx for idx, name in enumerate(base_model_list)}
    # print(judge_model_index_map)

    paras = list()
    for subdir in base_model_list:
        models_to_sort = [model for model in base_model_list if model != subdir]
        print(models_to_sort)
        combinations = list(itertools.combinations(models_to_sort, 2))
        remaining_combinations = set(combinations)
        comparison_matrix = np.zeros((len(base_model_list), len(base_model_list)))
        paras.append([base_dir, subdir, model_weights, sort_model_index_map, judge_model_index_map, comparison_matrix,
                      remaining_combinations, valid_question_ids])
    with Pool(processes=(cpu_count() - 1)) as pool:
        result = pool.starmap(pairwise_judge, paras)
    for i in range(len(result)):
        for key,value in result[i][1].items():
            battle_id = len(judge_dict)
            judge_dict[battle_id] = value

    return judge_dict

def fit_bt(X, Y, models, sample_weight, flag_weights=None, indices=None, SCALE=400, INIT_RATING=1000):
    p = len(models.index)

    lr = LogisticRegression(fit_intercept=False)
    if indices:
        if flag_weights == True:
            lr.fit(X[indices], Y[indices], sample_weight=sample_weight)
        else:
            lr.fit(X[indices], Y[indices])
    else:
        if flag_weights == True:
            lr.fit(X, Y, sample_weight=sample_weight)
        else:
            lr.fit(X, Y)

    elo_scores = SCALE * lr.coef_[0] + INIT_RATING
    # calibrate llama-13b to 800 if applicable
    if "mixtral-8x7b-instruct-v0.1" in models.index:
        elo_scores += 1114 - elo_scores[models["mixtral-8x7b-instruct-v0.1"]]

    # 0-1 normalization of elo_scores
    min_score = elo_scores[:p].min()
    max_score = elo_scores[:p].max()
    print(elo_scores)
    normalized_elo_scores = (elo_scores - min_score) / (max_score - min_score)

    # Convert the normalized scores to a dict
    elo_scores_dict = dict(zip(models.index, normalized_elo_scores))

    return (
        pd.Series(elo_scores[:p], index=models.index).sort_values(ascending=False),
        lr.coef_[0][p:],
        elo_scores_dict,
    )

def construct_matrices(
    df,
    elo_scores_dict,
    BASE=10,
    apply_ratio=[1],
    # apply_ratio=[1,1,1,1],
    style_elements=[
    "sum_assistant_a_length",
    "header_count_a",
    "list_count_a",
    "bold_count_a",
    "sum_assistant_b_length",
    "header_count_b",
    "list_count_b",
    "bold_count_b",
    ],
    add_one=True,
    style_ctl=False,
):
    style_elements = ["sum_assistant_a_length",
    "header_count_a",
    "list_count_a",
    "sum_assistant_b_length",
    "header_count_b",
    "list_count_b",]
    models = pd.concat([df["model_A"], df["model_B"]]).unique()
    models = pd.Series(np.arange(len(models)), index=models)
    print(models)
    # breakpoint()
    # duplicate battles
    # df = pd.concat([df, df], ignore_index=True)
    p = len(models.index)
    n = df.shape[0]
    # assert len(style_elements) % 2 == 0
    k = int(len(style_elements) / 2)
    print(k)
    print("-------")
    weights = []
    for i in range(n):
        try:
            weight = elo_scores_dict[df.iloc[i]['judge_model']]*5
        except: # single judge 如compassjudge 不参与排名
            weight = 1
        weights.append(weight)

    # sorted_elo = sorted(elo_scores_dict.items(), key=lambda x: x[1], reverse=True)
    # ranking_dict = {judge_model: rank for rank, (judge_model, _) in enumerate(sorted_elo)}
    # # 根据排名来分配权重
    # for i in range(n):
    #     judge_model = df.iloc[i]['judge_model']
    #     try:
    #         rank = ranking_dict[judge_model]
    #         # 第一名权重为5，最后一名权重为0，线性递减
    #         weight = 5 - (rank / (len(elo_scores_dict) - 1)) * 5
    #     except KeyError:  # 如果某个judge_model没有在elo_scores_dict中，如compassjudge等
    #         weight = 1 
    #     weights.append(weight)
    # print(weights)
    # breakpoint()
    metadata_list = []

    if style_ctl:
        X = np.zeros([n, p+k])
    else:
        X = np.zeros([n, p])       
    X[np.arange(n), models[df["model_A"]]] = +math.log(BASE)
    X[np.arange(n), models[df["model_B"]]] = -math.log(BASE)
    # print(X)
    # # creates turn each of the specified column in "conv_metadata" into a vector
    if style_ctl:
        style_vector = np.array(
            [
                df.metadata.map(
                    lambda x: x[element]
                    if type(x[element]) is int
                    else sum(x[element].values())
                ).tolist()
                for element in style_elements
            ]
        )

        style_diff = (style_vector[:k] - style_vector[k:]).astype(float)
        style_sum = (style_vector[:k] + style_vector[k:]).astype(float)

        if add_one:
            style_sum = style_sum + np.ones(style_diff.shape)

        apply_ratio = np.flatnonzero(apply_ratio)

        style_diff[apply_ratio] /= style_sum[
            apply_ratio
        ]  # Apply ratio where necessary (length, etc)

        style_mean = np.mean(style_diff, axis=1)
        style_std = np.std(style_diff, axis=1)

        X[:, -k:] = ((style_diff - style_mean[:, np.newaxis]) / style_std[:, np.newaxis]).T

    # one A win => two A win
    Y = np.zeros(n)
    Y[df["winner"] == df["model_A"]] = 1.0
    # print(Y)
    # one tie => one A win + one B win
    # find tie + tie (both bad) index
    # tie_idx = (df["winner"] == "TIE") | (df["winner"] == "Tie")
    # tie_idx[len(tie_idx) // 2 :] = False
    # Y[tie_idx] = 1.0

    return X, Y, models, weights

def count_markdown_elements(markdown_text, suffix):
    counters = {
        f"header_count{suffix}": {
            "h1": len(re.findall(r"^#{1}\s", markdown_text, re.MULTILINE)),
            "h2": len(re.findall(r"^#{2}\s", markdown_text, re.MULTILINE)),
            "h3": len(re.findall(r"^#{3}\s", markdown_text, re.MULTILINE)),
            "h4": len(re.findall(r"^#{4}\s", markdown_text, re.MULTILINE)),
            "h5": len(re.findall(r"^#{5}\s", markdown_text, re.MULTILINE)),
            "h6": len(re.findall(r"^#{6}\s", markdown_text, re.MULTILINE)),
        },
        f"list_count{suffix}": {
            "ordered": len(re.findall(r"^\s*\d+\.\s", markdown_text, re.MULTILINE)),
            "unordered": len(re.findall(r"^\s*[-*+]\s", markdown_text, re.MULTILINE)),
        },
        f"bold_count{suffix}": {
            "**": len(re.findall(r"\*\*[^*\n]+\*\*", markdown_text)),
            "__": len(re.findall(r"__[^_\n]+__", markdown_text)),
        },
    }
    return counters

def main(base_dir="/home/yanbin/De-Arena/judgements_mt_bench", valid_question_ids=overall_ids,existing_model_paths=existing_model_paths):
    sort_model_list = ['o1-mini', 'o1-preview', 'ChatGPT-4o-latest', 'gpt-4o-2024-05-13', 'gpt-4o-2024-08-06', 'gpt-4-1106-preview', 'gpt-4-turbo-2024-04-09', 'gpt-3.5-turbo-0125', 'gpt-4o-mini-2024-07-18']
    judge_model_list = ['o1-mini', 'o1-preview', 'ChatGPT-4o-latest', 'gpt-4o-2024-05-13', 'gpt-4o-2024-08-06', 'gpt-4-1106-preview', 'gpt-4-turbo-2024-04-09', 'gpt-3.5-turbo-0125', 'gpt-4o-mini-2024-07-18']
    judge_api_model = ['o1-mini', 'o1-preview', 'ChatGPT-4o-latest', 'gpt-4o-2024-05-13', 'gpt-4o-2024-08-06', 'gpt-4-1106-preview', 'gpt-4-turbo-2024-04-09', 'gpt-3.5-turbo-0125', 'gpt-4o-mini-2024-07-18']
    model_list = list(existing_model_paths.keys())
    judge_model_states = {model: list() for model in judge_model_list}

    base_model_list = ["o1-mini","gpt-4o-mini-2024-07-18","gpt-3.5-turbo-0125",]

    judge_dict = base_model_judge(base_dir,base_model_list)

    print(judge_dict)
    df = pd.DataFrame(judge_dict)
    df = df.T
    df = df[df['winner'] != 'TIE']
    df = df[df['winner'] != 'Tie']
    # print(df)
    print(len(df))

    init_elo_weight = {model:1 for model in base_model_list}
    X, Y, models, weights = construct_matrices(df,init_elo_weight)

    elo_rating_style, style_coef, elo_scores_dict = fit_bt(X, Y, models, weights, flag_weights=False)
    print(elo_rating_style, style_coef, elo_scores_dict)
    # breakpoint()

    start_time = time.time()

    judge_model_list = judge_open_model+judge_api_model
    models_to_sort = [model for model in sort_model_list if model not in base_model_list]
    random.seed(2025)
    random.shuffle(models_to_sort)
    add_model = list()

    model_weights = elo_scores_dict
    bottle_model_list = list()
    base_model_ranking = dict()
    for rank, model in enumerate(base_model_list):
        base_model_ranking[model] = rank + 1
    for new_model in tqdm(models_to_sort, desc="Processing sorting"):
        # 二分查找排名
        sort_rank,judge_model_states,final_binary_judge_dict = rugged_rank(base_dir, new_model, base_model_list, base_model_ranking, model_weights,
                                judge_model_list, judge_model_states)
        print(sort_rank)
        # breakpoint()

        for key,value in final_binary_judge_dict.items():
            battle_id = len(judge_dict)
            judge_dict[battle_id] = value
        df = pd.DataFrame(judge_dict)
        df = df.T
        df = df[df['winner'] != 'TIE']
        df = df[df['winner'] != 'Tie']
        df = df.drop_duplicates(subset=['judge_model', 'model_A', 'model_B', 'question_id']).reset_index(drop=True)
        # print(df)
        print(len(df))
        X, Y, models, sample_weight = construct_matrices(df,elo_scores_dict)
        elo_rating_style, style_coef, elo_scores_dict = fit_bt(X, Y, models, sample_weight)
        print(elo_rating_style, style_coef)
        with open('mtbench_elo_rating.txt', 'a') as f:
            f.write(f"Elo Rating Style:\n{elo_rating_style}\n")
            f.write(f"Style Coefficient:\n{style_coef}\n")
            f.write(f"---------next step---------\n")
        # breakpoint()
        model_weights = elo_scores_dict
        print(new_model, sort_rank)

        # 第一次细粒度排名
        sort_model_index_map, min_rank_idx, vote_matrix, bottle_model_list, judge_model_states,bottle_judge_dict = full_comparsion(base_dir,
                                                                                                        new_model,
                                                                                                        base_model_list,
                                                                                                        sort_rank,
                                                                                                        model_weights,
                                                                                                        judge_model_list,
                                                                                                        judge_model_states)

        for key,value in bottle_judge_dict.items():
            battle_id = len(judge_dict)
            judge_dict[battle_id] = value
        df = pd.DataFrame(judge_dict)
        df = df.T
        df = df[df['winner'] != 'TIE']
        df = df[df['winner'] != 'Tie']
        df = df.drop_duplicates(subset=['judge_model', 'model_A', 'model_B', 'question_id']).reset_index(drop=True)
        print(len(df))
        X, Y, models, sample_weight = construct_matrices(df,elo_scores_dict)
        elo_rating_style, style_coef, elo_scores_dict = fit_bt(X, Y, models, sample_weight)
        print(elo_rating_style, style_coef)
        print(elo_scores_dict)
        model_weights = elo_scores_dict
        with open('mtbench_elo_rating.txt', 'a') as f:
            f.write(f"Elo Rating Style:\n{elo_rating_style}\n")
            f.write(f"Style Coefficient:\n{style_coef}\n")
            f.write(f"---------next step---------\n")
        # breakpoint()

        # 根据值从高到低排序
        sorted_items = sorted(elo_rating_style.items(), key=lambda item: item[1], reverse=True)

        # 获取排名
        base_model_ranking = {item[0]: rank + 1 for rank, item in enumerate(sorted_items)}
        print(base_model_ranking)
        base_model_list = sorted(base_model_ranking, key=elo_rating_style.get, reverse=True)
        print(base_model_list)
        # 判断是否继续进行bubble window
        new_model_rank = base_model_list.index(new_model)+1
        model_num = len(base_model_list)
        print(sort_rank, new_model_rank)
        flag_bubble = judge_bubble(sort_rank, new_model_rank, model_num)

        while flag_bubble:
            print("============")
            old_model_rank = new_model_rank
            model_names, sort_model_index_map, weights, bottle_model_list,judge_model_states,bottle_judge_dict = bubble_window(base_dir,
                                                                                                        new_model,
                                                                                                        base_model_list,
                                                                                                        new_model_rank,
                                                                                                        model_weights,
                                                                                                        judge_model_list,
                                                                                                        judge_model_states)

            for key,value in bottle_judge_dict.items():
                battle_id = len(judge_dict)
                judge_dict[battle_id] = value
            df = pd.DataFrame(judge_dict)
            df = df.T
            df = df[df['winner'] != 'TIE']
            df = df[df['winner'] != 'Tie']
            df = df.drop_duplicates(subset=['judge_model', 'model_A', 'model_B', 'question_id']).reset_index(drop=True)
            # print(df)
            print(len(df))
            X, Y, models, sample_weight = construct_matrices(df,elo_scores_dict)
            elo_rating_style, style_coef, elo_scores_dict = fit_bt(X, Y, models, sample_weight)
            print(elo_rating_style, style_coef)
            with open('mtbench_elo_rating.txt', 'a') as f:
                f.write(f"Elo Rating Style:\n{elo_rating_style}\n")
                f.write(f"Style Coefficient:\n{style_coef}\n")
                f.write(f"---------next step---------\n")
            # 根据值从高到低排序
            sorted_items = sorted(elo_rating_style.items(), key=lambda item: item[1], reverse=True)

            # 获取排名
            base_model_ranking = {item[0]: rank + 1 for rank, item in enumerate(sorted_items)}
            base_model_list = sorted(base_model_ranking, key=elo_rating_style.get, reverse=True)
            new_model_rank = base_model_list.index(new_model)+1
            model_num = len(base_model_list)
            print(base_model_list)
            print(old_model_rank,new_model_rank)
            model_weights = elo_scores_dict
            flag_bubble = judge_continue_bubble(old_model_rank,new_model_rank,model_num)

    with open(save_output_file_path, 'a') as f:
        f.write(f"judge model states: {judge_model_states}\n")

    end_time = time.time()
    print(end_time - start_time)

if __name__ == "__main__":
    fire.Fire(main)