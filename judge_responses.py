import os
# Set the multiprocess method to 'spawn' to avoid CUDA initialization issues
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
# os.environ["CUDA_VISIBLE_DEVICES"] = "7,4"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:False'
import json
import random
import argparse
from tqdm import tqdm
import fire
import uuid
import torch
from vllm import SamplingParams
import gc
from vllm import LLM
from openai import OpenAI
import anthropic 
from decimal import Decimal
import re
import numpy as np
from utils_final import existing_model_paths
from tokencost import calculate_completion_cost, calculate_prompt_cost
import time
from decimal import Decimal
import sys
import copy
import itertools
total_completion_cost = 0
total_prompt_cost = 0

def save_to_jsonl(data, filename):
    """Saves a Python data structure to a .jsonl file."""
    with open(filename, 'w') as f:
        f.write(json.dumps(data) + '\n')

def load_records(filename):
    with open(filename, 'r') as file:
        return [json.loads(line.strip()) for line in file]

def update_voting_records(model, response_A_name, response_B_name, won, question_id, data_id, dimension, split=0):
    """Updates the voting records with a new voting result."""
    if split!=0:
        records_path = f"judgements_{dimension}/{model}/voting_records_{split}.jsonl"
    else:
        records_path = f"judgements_{dimension}/{model}/voting_records.jsonl"

    # Ensure the directory exists
    os.makedirs(os.path.dirname(records_path), exist_ok=True)

    # Load existing records or create an empty list if the file does not exist
    if os.path.exists(records_path):
        records = load_records(records_path)[0]
    else:
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

def format_prompt(model_name, prompt, tokenizer=None):
    if "vicuna" in model_name.lower():
        return f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {prompt} ASSISTANT:"
    elif "llama2-13b-chat" in model_name.lower() or "llama2-7b-chat" in model_name.lower():
        return f"<s>[INST] {prompt} [/INST] {{ model_answer }}</s>"
    elif "openchat-3.5" in model_name.lower():
        return f"You are a helpful assistant. GPT4 Correct User: {prompt} GPT4 Correct Assistant:"
    elif "koala-13b" in model_name.lower():
        text = f"BEGINNING OF CONVERSATION: USER: {prompt} GPT:"
        return text
    elif "openassistant-pythia-12b" in model_name.lower():
        text = f"<|prompter|>{prompt}<|endoftext|><|assistant|>"
        return text
    return prompt

# Function to run the HuggingFace model with specific settings (temperature, max_tokens)
def run_hf_model(prompts, judge_name, tokenizer, engine, temperature=0.7, max_tokens=15):
    # Set the max tokens based on the judge name
    if judge_name == "athene-70b" or judge_name == "gemma-2-2b-it" or judge_name == "gemma-1.1-2b-it" or judge_name == "llama2-13b-chat" or judge_name == "gemma-1.1-7b-it":
        max_new_tokens = 16
    else:
        max_new_tokens = 15
    if judge_name == "koala-13b" or judge_name == "openassistant-pythia-12b":
        prompts = [format_prompt(judge_name,prompt) for prompt in prompts]
    # Generate responses based on the sampling parameters
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_new_tokens)
    outputs = engine.generate(prompts, sampling_params=sampling_params)
    responses = []
    for output in outputs:
        responses.append(output.outputs[0].text)
    return responses

# Function to run OpenAI models
def run_openai_model(prompts, model_name, client, temperature=0.7, max_tokens=15):
    # Handle model selection for OpenAI models
    if "3.5-turbo-0125" in model_name: 
        model_name = "gpt-3.5-turbo-0125"
    elif "4-1106" in model_name: 
        model_name = "gpt-4-1106-preview"
    elif "gpt-4o-mini" in model_name:
        model_name = "gpt-4o-mini-2024-07-18"
    elif "ChatGPT-4o-latest" in model_name:
        model_name = "chatgpt-4o-latest"
    elif "gpt-4-turbo-2024-04-09" in model_name: 
        model_name = "gpt-4-turbo-2024-04-09"
    elif "gpt-4o-2024-05-13" in model_name:
        model_name = "gpt-4o-2024-05-13"
    elif "gpt-4o-2024-08-06" in model_name:
        model_name = "gpt-4o-2024-08-06"
    elif "o1-mini" in model_name:
        model_name = "o1-mini-2024-09-12"
        responses = []
        # Modify each prompt to ask the model to evaluate dataset quality
        for prompt in prompts:
            # Call OpenAI API with the modified quality evaluation prompt
            text = ""
            while not text.strip():  # 当文本为空时继续循环
                # 调用 OpenAI API
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                )
                
                # 提取和存储响应
                text = completion.choices[0].message.content
            responses.append(str(text))
        
        return responses
    elif "o1-preview" in model_name:
        model_name = "o1-preview-2024-09-12"
        responses = []
        # Modify each prompt to ask the model to evaluate dataset quality
        for prompt in prompts:
            # Call OpenAI API with the modified quality evaluation prompt
            text = ""
            while not text.strip():  # 当文本为空时继续循环
                # 调用 OpenAI API
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                )
                
                # 提取和存储响应
                text = completion.choices[0].message.content
            responses.append(str(text))
        
        return responses
    responses = []
    
    # Modify each prompt to ask the model to evaluate dataset quality
    for prompt in prompts:
        # Call OpenAI API with the modified quality evaluation prompt
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Extract and store the response
        text = completion.choices[0].message.content
        responses.append(str(text))
    
    return responses

def run_claude_model(prompts, client, model_name="claude-3-opus", max_tokens=2048):
    if model_name=="claude-3.5-sonnet":
        model_name="claude-3-5-sonnet-20240620"
    elif model_name=="claude-3-opus":
        model_name="claude-3-opus-20240229"
    elif model_name=="claude-3-sonnet":
        model_name="claude-3-sonnet-20240229"
    elif model_name=="claude-3-haiku":
        model_name="claude-3-haiku-20240307"
    elif model_name=="claude-1":
        model_name="claude-instant-1.2"
    elif model_name=="claude-2.0":
        model_name="claude-2.0"
    else:
        model_name="claude-2.1"
    responses = []
    
    for prompt in prompts:
        message = client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        response_text = ''.join([block.text for block in message.content])
        responses.append(response_text)
    
    return responses

# Function to load the appropriate model (OpenAI, Anthropic, or HuggingFace)
def load_model(model_name,tensor_parallel_size, enforce_eager=True):
    model_info = existing_model_paths.get(model_name)

    if not model_info:
        raise ValueError("Unsupported model")

    # Return OpenAI or Anthropic models if applicable
    if model_info == "OPENAI":
        return None, "OPENAI"
    
    if model_info == "Claude":
        return None, "Anthropic"

    if model_info == "gemini":
        return None, "gemini"

    # Set attention backend for specific models
    if "gemma-2" in model_name.lower():
        os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"

    ## Load other API models


    # Check if the model path exists and load the vLLM model
    if os.path.exists(model_info):
        print(f"vLLM model detected, loading from: {model_info}")
        vllm_model = LLM(model=model_info, gpu_memory_utilization=0.7, tensor_parallel_size=tensor_parallel_size, enforce_eager=True)
            
        if hasattr(vllm_model, "to"):
            vllm_model.to("cuda")
        else:
            print("The model does not support `to` method.")
        return None, vllm_model  

# Function to load JSONL files line by line
def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, start=1):
            try:
                json_data = json.loads(line.strip())
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            yield json_data
    return json_data

# Function to get a question and its reference answer from a JSONL file
def get_question_with_reference(path, prompt_id):
    questions = load_jsonl(path)
    for question in questions:
        if question['question_id'] == prompt_id:
            return question['turns'][0], question.get('reference', [""])[0]
    return None, ""

# Function to fetch responses for a given model from a JSONL file
def fetch_responses(path,model):
    directory = f"{path}/{model}.jsonl"
    with open(directory, 'r', encoding='utf-8') as file:
        data = file.read()
    
    # Process the JSON to clean up and format correctly
    data = data.strip().replace('}\n{', '},{')
    data = f'[{data}]'  # Add square brackets to make it a valid JSON array
    return json.loads(data)

# Function to generate a pairwise judgment prompt
def judge_prompt_pairwise(question, answer_a, answer_b):
    prompt = (
        "[System]\n"
        'Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user’s instructions and answers the user’s question better. You should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Answer by providing the model identifier of the best model. We will use your output as the name of the best model, so make sure your output only contains one of the following model identifiers and nothing else (no quotes, no spaces, no new lines, ...): "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.\n\n'
        "[User Question]\n"
        f"{question}\n\n"
        "[The Start of Assistant A's Answer]\n"
        f"{answer_a}\n"
        "[The End of Assistant A's Answer]\n\n"
        "[The Start of Assistant B's Answer]\n"
        f"{answer_b}\n"
        "[The End of Assistant B's Answer]\n\n"
        '[The Verdict(only contains one model identifier)]\n'
    )
    return prompt

# Function to generate a reference-based judgment prompt
def judge_prompt_pair_reference(question, answer_a, answer_b, ref_answer):
    prompt = (
        "[System]\n"
        "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should consider correctness and helpfulness. You will be given a reference answer, assistant A's answer, and assistant B's answer. Your job is to evaluate which assistant's answer is better. You should compare both assistants' answers with the reference answer. Identify and correct any mistakes. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. Answer by providing the model identifier of the best model. We will use your output as the name of the best model, so make sure your output only contains one of the following model identifiers and nothing else (no quotes, no spaces, no new lines, ...): \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.\n\n"
        "[User Question]\n"
        f"{question}\n\n"
        "[The Start of Reference Answer]\n"
        f"{ref_answer}\n"
        "[The End of Reference Answer]\n\n"
        "[The Start of Assistant A's Answer]\n"
        f"{answer_a}\n"
        "[The End of Assistant A's Answer]\n\n"
        "[The Start of Assistant B's Answer]\n"
        f"{answer_b}\n"
        "[The End of Assistant B's Answer]\n\n"
        '[The Verdict(only contains one model identifier)]\n'
    )
    return prompt

# Function to determine the winner between two responses based on judge output
def determine_winner(judge_response, model_a, model_b):
    if "[[A]]" in judge_response and "[[B]]" not in judge_response:
        winner = model_a
    elif "[[B]]" in judge_response and "[[A]]" not in judge_response:
        winner = model_b
    else:
        if "[A]" in judge_response and "[B]" not in judge_response:
            winner = model_a
        elif "[B]" in judge_response and "[A]" not in judge_response:
            winner = model_b
        else:
            winner = "Tie"
    return winner

def resume_check(combination_models, initial_question_ids, model, dimension, split=0):
    """Updates the voting records with a new voting result."""
    if split != 0:
        records_path = f"judgements_{dimension}/{model}/voting_records_{split}.jsonl"
    else:
        records_path = f"judgements_{dimension}/{model}/voting_records.jsonl"

    if os.path.exists(records_path):
        try:
            records = load_records(records_path)[0]
        except:
            records = []
        pair2count = {}
        for record in records:
            pair = (record["response_A"], record["response_B"])
            pair2count[pair] = pair2count.get(pair, 0) + 1
        new_combination_models = []
        for res_A, res_B in combination_models:
            pair = (res_A, res_B)
            if pair2count.get(pair, 0) < len(initial_question_ids):
                new_combination_models.append(pair)
        return new_combination_models
    else:
        return combination_models

# Main function to run the judging trials, handling multiple models
def run_judging_trials(path="math_questions.jsonl", model_name="command-r-v01", model_names="", tensor_parallel_size=1, dimension="math_algebra", split=0):
    print(path, model_name, model_names, dimension)
    global total_completion_cost, total_prompt_cost
    model_names = model_names.split(',')
    print(model_names)

    start_time = time.time()
    initial_question_ids = list(range(81, 102)) + list(range(103, 121))

    responses_dict = dict()
    # Fetch responses for each model
    for model in model_names:
        responses_dict[model] = fetch_responses(f"{dimension}_responses", model)

    # Select specific models for the judging trials
    for model in [model_name]:
        pair_models = copy.deepcopy(model_names)
        pair_models.remove(model)
        combination_models = list(itertools.combinations(pair_models, 2))
        if split==1:
            c_len = len(combination_models) // 2
            combination_models = combination_models[:c_len]
        elif split==2:
            c_len = len(combination_models) // 2
            combination_models = combination_models[c_len:]

        new_combination_models = resume_check(combination_models, initial_question_ids, model_name, dimension, split)
        print(f"After Resume Check, {len(combination_models)} pairs are reduced into {len(new_combination_models)} pairs.")
        combination_models = new_combination_models

        tokenizer, judge_model = load_model(model, tensor_parallel_size)
        client = None
        print(judge_model)
        if judge_model == "OPENAI":
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if judge_model == "Anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            client = anthropic.Anthropic(api_key=api_key)
        
        # Iterate over combinations of model pairs for comparison
        for model_a,model_b in tqdm(combination_models):
            responses_a = responses_dict[model_a]
            responses_b = responses_dict[model_b]
            print(model_a,model_b)

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
                    if reference !="":
                        prompt = judge_prompt_pair_reference(question, response_a,response_b,reference)
                        swapped_prompt = judge_prompt_pair_reference(question, response_b,response_a,reference)
                    else:
                        prompt = judge_prompt_pairwise(question, response_a,response_b)
                        swapped_prompt = judge_prompt_pairwise(question, response_b,response_a)                       
                    prompts.append(prompt)
                    swapped_prompts.append(swapped_prompt)
                    question_ids.append(question_id)

                try:
                    # Adjust logic based on the type of judge_model
                    if judge_model == 'OPENAI':  # For OpenAI models
                        judge_responses = run_openai_model(prompts, model, client)
                        swapped_judge_responses = run_openai_model(swapped_prompts, model, client)
                    elif judge_model == "Anthropic":  # For Anthropic models (e.g., Claude)
                        # Placeholder for Anthropic, no operation for now
                        judge_responses = run_claude_model(prompts, client, model_name)
                        swapped_judge_responses = run_claude_model(swapped_prompts, client, model_name)
                    else:  # For other HuggingFace (HF) models
                        judge_responses = run_hf_model(prompts, model, tokenizer, judge_model)
                        swapped_judge_responses = run_hf_model(swapped_prompts, model, tokenizer, judge_model)
                except Exception as e:
                    print(f"Error evaluating model pair ({model_a}, {model_b}) with judge {model}: {e}")
                    continue  # Skip to the next model pair if there's an error

                cnt = 0
                # Process responses and determine winners
                for response, swapped_response in zip(judge_responses, swapped_judge_responses):
                    winner = determine_winner(response, model_a, model_b)
                    swapped_winner = determine_winner(swapped_response, model_b, model_a)
                    final_winner = winner if winner == swapped_winner else "TIE"
                    data_id = str(uuid.uuid4())
                    update_voting_records(model, model_a, model_b, final_winner, question_ids[cnt], data_id, dimension, split)


        end_time = time.time()
        duration = end_time - start_time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run all models with specified parameters.')
    
    parser.add_argument('--path', type=str, default='math_questions.jsonl', help='Path to the input file')
    parser.add_argument('--model_name', type=str, default="vicuna-33b", help='Comma-separated list of model names')
    parser.add_argument('--dimension', type=str, default="math_algebra", help='new dimension names')
    parser.add_argument('--tensor_parallel_size', type=int, default=2, help='Tensor parallel size')
    parser.add_argument('--model_names', type=str, default="vicuna-33b", help='Comma-separated list of model names')
    
    args = parser.parse_args()

    fire.Fire(run_judging_trials(path=args.path, model_name=args.model_name, model_names=args.model_names, tensor_parallel_size=args.tensor_parallel_size, dimension=args.dimension))