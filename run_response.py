import os
# Set the multiprocess method to 'spawn' to avoid CUDA initialization issues
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
# os.environ["CUDA_VISIBLE_DEVICES"] = "7,4"  # Set this before any call to torch.cuda.device_count()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:False'
import torch
import argparse
import json
import fire
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from utils_final import existing_model_paths
from tqdm import tqdm
import re
from tokencost import calculate_completion_cost, calculate_prompt_cost
import time
from openai import OpenAI
import anthropic 
import google.generativeai as genai
import time
from zhipuai import ZhipuAI
from mistralai import Mistral
total_completion_cost = 0
total_prompt_cost = 0

def load_model(model_name, gpu_memory_utilization=1, tensor_parallel_size=2):
    model_info = existing_model_paths.get(model_name)
    print(existing_model_paths)
    print(model_info)
    if not model_info:
        raise ValueError("Unsupported model")
    if "gemma-2" in model_name.lower():
        os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    if os.path.exists(model_info):
        print(f"HF model detected, loading from: {model_info}")
        vllm_model = LLM(model=model_info, gpu_memory_utilization=gpu_memory_utilization, tensor_parallel_size=tensor_parallel_size, enforce_eager=True, trust_remote_code=True)
        return vllm_model

    raise FileNotFoundError("Model path does not exist")

def load_tokenizer(model_name, use_auth_token=None):
    if "llama3-8b-instruct" in model_name.lower():
        return AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    elif "llama-3.1-tulu-8b" in model_name.lower():
        return AutoTokenizer.from_pretrained("/data/shared/huggingface/hub/llama-3.1-tulu-8b")
    elif "zephyr-7b-beta" in model_name.lower():
        return AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta") 
    elif "meta-llama-3.1-8b-instruct" in model_name.lower():
        return AutoTokenizer.from_pretrained("/data/shared/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/07eb05b21d191a58c577b4a45982fe0c049d0693") 
    elif "gemma-2-9b-it-simpo" in model_name.lower():
        return AutoTokenizer.from_pretrained("/data/shared/huggingface/hub/gemma-2-9b-it-simpo") 
    elif "google-gemma-2-9b-it" in model_name.lower():
        return AutoTokenizer.from_pretrained("/data/shared/huggingface/hub/google-gemma-2-9b-it") 
    elif "vicuna-7b" in model_name.lower():
        return AutoTokenizer.from_pretrained("/data/shared/huggingface/hub/vicuna-7b") 
    return None 

def format_prompt(model_name, prompt, tokenizer=None):
    if "vicuna" in model_name.lower():
        return f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {prompt} ASSISTANT:"
    elif "qwen1.5" in model_name.lower() and tokenizer:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Please only answer in English. No chinese characters. {prompt} (No chinese characters.) (Please only answer in English.) " }
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return text
    elif "qwen2" in model_name.lower() and tokenizer:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return text
    elif "mistral-7b-instruct-2" in model_name.lower() or "mistral-8x7b-instruct-v0.1" in model_name.lower() or "mistral-7b-instruct-1" in model_name.lower():
        return f"[INST] {prompt} [/INST]"
    elif "llama3-8b-instruct" in model_name.lower() or "llama-3-70b-instruct" in model_name.lower() or "meta-llama-3.1" in model_name.lower() and tokenizer :
        conversations = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}],
            tokenize=False,
        )
        return conversations
    elif "llama2-13b-chat" in model_name.lower() or "llama2-7b-chat" in model_name.lower():
        return f"<s>[INST] {prompt} [/INST] {{ model_answer }}</s>"
    elif "openchat-3.5" in model_name.lower():
        return f"You are a helpful assistant. GPT4 Correct User: {prompt} GPT4 Correct Assistant:"
    elif "zephyr-7b-beta" in model_name.lower() and tokenizer:
        messages = [
            {
                "role": "system",
                "content": "You are a friendly chatbot who always responds in the style of a pirate",
            },
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return text
    elif "gemma-2-27b-it" in model_name.lower() or "gemma-1.1-7b-it" in model_name.lower() or "google-gemma-2-9b-it" in model_name.lower() or "gemma-2-2b-it" in model_name.lower() or "gemma-1.1-2b-it" in model_name.lower() or "gemma-7b-it" in model_name.lower() or "gemma-2b-it" in model_name.lower() and tokenizer:
        chat = [
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True
        )
        return text
    elif "yi" in model_name.lower() and tokenizer:
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            conversation=messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        return text
    elif "command" in model_name.lower():
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            conversation=messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        return text
    elif 'athene-70b' in model_name.lower():
        messages = [
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            conversation=messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        return text
    elif "gemma-2-9b-it-simpo" in model_name.lower():
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            conversation=messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        return text
    elif "jamba-1.5-mini" in model_name.lower():
        messages = [
        {"role": "system", "content": "You are an ancient oracle who speaks in cryptic but wise phrases, always hinting at deeper meanings."},
        {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            conversation=messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        return text
    elif "starling-lm" in model_name.lower():
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            conversation=messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        return text
    elif "koala-13b" in model_name.lower():
        text = f"BEGINNING OF CONVERSATION: USER: {prompt} GPT:"
        return text
    elif "openassistant-pythia-12b" in model_name.lower():
        text = f"<|prompter|>{prompt}<|endoftext|><|assistant|>"
        return text
    elif "olmo-7b-instruct" in model_name.lower():
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            conversation=messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        return text
    elif "nemotron-70b" in model_name.lower():
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            conversation=messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        return text
    return prompt

def save_time_estimation(model,cost_data):
    """Save response time usage to a text file."""
    path = "response_time/response_time.txt"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a') as f:
        f.write(model + " time:"+ str(cost_data) + '\n')

def run_vllm_model(prompts, model, model_name, max_tokens=1024, temperature=0.7):
    global total_completion_cost, total_prompt_cost
    # Calculating prompts costs of the model using prompts
    if model_name.lower() in ["gpt3", "gpt4"]:
        total_prompt_cost += sum([calculate_prompt_cost(prompt, model_name) for prompt in prompts])

    tokenizer = load_tokenizer(model_name)
    formatted_prompts = [format_prompt(model_name, prompt, tokenizer) for prompt in prompts]
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        top_p=0.95,
        temperature=temperature,
    )

    if model_name.lower() in ["gemma-2-9b-it-simpo","meta-llama-3.1-70b-instruct","koala-13b"]:
        sampling_params = SamplingParams(
            max_tokens=max_tokens*3,
            top_p=0.95,
            temperature=temperature,
        )
    if model_name.lower() in ['gemma-2-27b-it','gemma-2-2b-it',"yi-1.5-34b-chat","yi-34b-chat","llama2-13b-chat","openchat-3.5","openchat-3.5-0106","llama2-7b-chat"]:
        sampling_params = SamplingParams(
            max_tokens=max_tokens*2,
            top_p=0.95,
            temperature=temperature,
        )
    
    if model_name.lower() in ["qwen1.5-14b-chat", "qwen1.5-32b-chat", "qwen2-72b-instruct"]:
        sampling_params = SamplingParams(
            max_tokens=max_tokens*2,
            repetition_penalty=1.05,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            stop_token_ids=[151645, 151643]
        )
    
    if model_name.lower() in ["vicuna-33b"]:
        sampling_params = SamplingParams(
            max_tokens=max_tokens*2,
            repetition_penalty=1.1,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            # stop_token_ids=[2, 0]
        )
    
    if model_name.lower() in ["mistral-8x7b-instruct-v0.1"]:
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            repetition_penalty=1.1,
            temperature=0.7,
            top_p=0.95,
            top_k=40
        )

    outputs = model.generate(formatted_prompts, sampling_params=sampling_params)

    responses = []
    for output in outputs:
        response = output.outputs[0].text

        if model_name.lower() in ["gpt3", "gpt4"]:
            completion_cost = calculate_completion_cost(response, model_name)
            total_completion_cost += completion_cost
        responses.append(response)

    print(responses)
    print('/n')

    return responses

def run_openai_model(prompts, model_name, client, temperature=0.7, max_tokens=2048):
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
        model_name = "o1-mini"
        responses = []
        # Modify each prompt to ask the model to evaluate dataset quality
        for prompt in prompts:
            text = ""
            cnt = 0
            while not text.strip():  # 当文本为空时继续循环
                cnt += 1
                if cnt >= 5:
                    break
                print(cnt)
                try:
                    # 调用 OpenAI API
                    completion = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        max_completion_tokens=max_tokens
                    )
                    # 提取和存储响应
                    text = completion.choices[0].message.content
                except Exception as e:
                    print(f"Error encountered: {e}")
                    text = ""  # 重置文本以便继续重试
            responses.append(str(text))
        
        print(responses)  # Debugging output to verify the responses
        return responses

    elif "o1-preview" in model_name:
        model_name = "o1-preview-2024-09-12"
        responses = []
        # Modify each prompt to ask the model to evaluate dataset quality
        for prompt in prompts:
            text = ""
            cnt = 0
            while not text.strip():  # 当文本为空时继续循环
                cnt += 1
                if cnt >= 10:
                    break
                print(cnt)
                try:
                    # 调用 OpenAI API
                    completion = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        max_completion_tokens=max_tokens
                    )
                    # 提取和存储响应
                    text = completion.choices[0].message.content
                except Exception as e:
                    print(f"Error encountered: {e}")
                    text = ""  # 重置文本以便继续重试
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
    
    print(responses)  # Debugging output to verify the responses
    return responses

def run_gemini_model(prompts, client, model_name="gemini-1.5-flash", max_tokens=2048):
    if model_name=="gemini-1.5-flash-exp-0827":
        model_name="gemini-1.5-flash-exp-0827"
    elif model_name=="gemini-1.5-flash-8b-exp-0827":
        model_name="gemini-1.5-flash-8b-exp-0827"
    elif model_name=="gemini-1.5-pro-exp-0827":
        model_name="gemini-1.5-pro-exp-0827"
    elif model_name=="gemini-1.5-pro-001":
        model_name="gemini-1.5-pro-001"
    elif model_name=="gemini-1.0-pro-001":
        model_name="gemini-1.0-pro-001"
    elif model_name == "gemini-1.5-flash-001":
        model_name="gemini-1.5-flash-001"

    responses = []
    genai.configure(api_key=client)
    model = genai.GenerativeModel(model_name)
    for prompt in prompts:
        cnt = 0
        while 1:
            cnt += 1
            if cnt >= 5:
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
                    time.sleep(2)
                    continue 
    
    return responses

# Run the Claude model to generate responses
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
        model_name="claude-instant-1"
    elif model_name=="claude-2.0":
        model_name="claude-2.0"
    elif model_name=="claude-3.5-sonnet-20241022":
        model_name="claude-3-5-sonnet-20241022"
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

# Run the Mistral model to generate responses
def run_mistral_model(prompts, client, model_name="mistral-medium", max_tokens=2048):
    responses = []
    
    for prompt in prompts:
        message = client.chat.complete(
            model=model_name,
            max_tokens=max_tokens,
            messages=[
                {
            "role": "user",
            "content": prompt,
        },
            ]
        )
        response_text = message.choices[0].message.content
        responses.append(response_text)
    
    return responses

def run_yi_model(prompts, model_name, temperature=0.7, max_tokens=2048):
    API_BASE = ""
    API_KEY = ""
    client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE
    )
    responses = []
    # Modify each prompt to ask the model to evaluate dataset quality
    for prompt in prompts:
        print(prompt)
        # Call OpenAI API with the modified quality evaluation prompt
        completion = client.chat.completions.create(
            model="yi-lightning",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        print(completion)
        # Extract and store the response
        text = completion.choices[0].message.content
        responses.append(str(text))
    
    print(responses)  # Debugging output to verify the responses
    return responses

def run_glm_model(prompts, model_name, temperature=0.7, max_tokens=2048):
    responses = []
    client = ZhipuAI(api_key="") # 填写您自己的APIKey
    # Modify each prompt to ask the model to evaluate dataset quality
    for prompt in prompts:
        # Call OpenAI API with the modified quality evaluation prompt
        completion = client.chat.completions.create(
            model="glm-4-plus",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        # Extract and store the response
        text = completion.choices[0].message.content
        responses.append(str(text))
    
    print(responses)  # Debugging output to verify the responses
    return responses

def save_responses(responses, model_name, output_dir, prompt_ids, prompts, question_ids, flag):
    empty_responses = []
    for i, response in enumerate(responses):
        if response.strip() == "" and flag == True:
            empty_responses.append((model_name, question_id, prompts[i]))
            continue
        question_id = question_ids[i]
        prompt_id = prompt_ids[i]
        directory = output_dir
        os.makedirs(directory, exist_ok=True)
        output_file = os.path.join(directory, f"{model_name}.jsonl")
        with open(output_file, 'a') as f:
            json.dump({"question_id":question_id,"response": response}, f, indent=4)
            f.write('\n')

    if empty_responses:
        print("Empty responses detected for the following model and question IDs:")
        for model, qid, prompt in empty_responses:
            print(f"Model: {model}, Question ID: {qid}")
            print(f"Prompt: {prompt}")

    return empty_responses

def re_prompt_empty_responses(empty_responses, model, model_name, max_tokens, temperature, max_attempts=5):
    new_prompts = [f"Please provide a brief answer and do not leave it empty. {prompt}" for model, qid, prompt in empty_responses]
    new_responses = []

    for i in range(len(empty_responses)):
        model_name, qid, prompt = empty_responses[i]
        response = ""
        attempts = 0
        while response.strip() == "" and attempts < max_attempts:
            print(f"Retrying empty response for Model: {model_name}, Question ID: {qid}, Attempt: {attempts + 1}")
            response = run_vllm_model([new_prompts[i]], model, model_name, max_tokens, temperature)[0]
            attempts += 1
        new_responses.append(response)

    return new_responses

def get_responses(prompts, question_ids, model, model_name, output_dir="model_responses", max_tokens=1024, temperature=0.7):
    responses = run_vllm_model(prompts, model, model_name, max_tokens, temperature)
    empty_responses = save_responses(responses, model_name, output_dir, list(range(len(prompts))), prompts, question_ids, flag=True)

    if empty_responses:
        new_responses = re_prompt_empty_responses(empty_responses, model, model_name, max_tokens, temperature)
        save_responses(new_responses, model_name, output_dir, [qid for _, qid, _ in empty_responses], [prompt for _, _, prompt in empty_responses], [qid for _, qid, _ in empty_responses],flag=False)

    torch.cuda.empty_cache()
    gc.collect()
    return responses

def load_jsonl(filename):
    with open(filename, 'r') as file:
        return [json.loads(line.strip()) for line in file]

def get_questions(path):
    questions = load_jsonl(path)
    question_map = {question['question_id']: question['turns'][0] for question in questions}
    return question_map

def run_all_models(output_dir="/home/yanbin/De-Arena/mt_bench_responses", model_names="vicuna-33b", path='/home/yanbin/De-Arena/mt_bench_questions.jsonl', openai_api='111' ,tensor_parallel_size=4, max_tokens=1024, batch_size=100, temperature=0.7, gpu_memory_utilization=0.5, client=None):
    print(model_names)
    question_map = get_questions(path)
    prompts = list(question_map.values())
    question_ids = list(question_map.keys())

    # print(question_ids)
    model_names = model_names.split(',')
    print(model_names)
    print(openai_api)

    # 根据给定的 question_ids 索引 prompt
    prompts = [question_map[qid] for qid in question_ids if qid in question_map]

    os.makedirs(output_dir, exist_ok=True)

    for model_name in tqdm(model_names):
        start_time = time.time()
        final_responses = dict()
        print(f"Processing model: {model_name}")
        if "gpt" in model_name.lower() or "o1-" in model_name.lower():
            num_batches = (len(prompts) + batch_size - 1) // batch_size
            client = OpenAI(api_key="")
            
            for i in range(num_batches):
                batch_prompts = prompts[i * batch_size : (i + 1) * batch_size]
                batch_question_ids = question_ids[i * batch_size : (i + 1) * batch_size]
                responses = run_openai_model(batch_prompts, model_name, client, temperature, 2048)
                print(responses)
                save_responses(responses, model_name, output_dir, list(range(len(batch_prompts))), batch_prompts, batch_question_ids, False)
        elif "claude" in model_name.lower():
            num_batches = (len(prompts) + batch_size - 1) // batch_size
            api_key = ''
            client = anthropic.Anthropic(api_key=api_key)
            for i in range(num_batches):
                batch_prompts = prompts[i * batch_size : (i + 1) * batch_size]
                batch_question_ids = question_ids[i * batch_size : (i + 1) * batch_size]
                print(batch_question_ids)
                # Correct the function call by removing the extra arguments
                responses = run_claude_model(batch_prompts, client, model_name, max_tokens)
                print(responses)
                save_responses(responses, model_name, output_dir, list(range(len(batch_prompts))), batch_prompts, batch_question_ids, False)  
        elif "gemini" in model_name.lower():
            # Fill in for Claude
            num_batches = (len(prompts) + batch_size - 1) // batch_size
            api_key = ''
            client = api_key
            for i in range(num_batches):
                batch_prompts = prompts[i * batch_size : (i + 1) * batch_size]
                batch_question_ids = question_ids[i * batch_size : (i + 1) * batch_size]
                print(batch_question_ids)
                # Correct the function call by removing the extra arguments
                responses = run_gemini_model(batch_prompts, client, model_name, max_tokens)
                print(responses)
                save_responses(responses, model_name, output_dir, list(range(len(batch_prompts))), batch_prompts, batch_question_ids, False)      
        elif "yi-lightning" in model_name.lower():
            # Fill in for Claude
            num_batches = (len(prompts) + batch_size - 1) // batch_size
            for i in range(num_batches):
                batch_prompts = prompts[i * batch_size : (i + 1) * batch_size]
                batch_question_ids = question_ids[i * batch_size : (i + 1) * batch_size]
                print(batch_question_ids)
                # Correct the function call by removing the extra arguments
                responses = run_yi_model(batch_prompts,model_name, max_tokens)
                print(responses)
                save_responses(responses, model_name, output_dir, list(range(len(batch_prompts))), batch_prompts, batch_question_ids, False)      
        elif "glm-4-plus" in model_name.lower():
            # Fill in for Claude
            num_batches = (len(prompts) + batch_size - 1) // batch_size
            for i in range(num_batches):
                batch_prompts = prompts[i * batch_size : (i + 1) * batch_size]
                batch_question_ids = question_ids[i * batch_size : (i + 1) * batch_size]
                print(batch_question_ids)
                # Correct the function call by removing the extra arguments
                responses = run_glm_model(batch_prompts,model_name, max_tokens)
                print(responses)
                save_responses(responses, model_name, output_dir, list(range(len(batch_prompts))), batch_prompts, batch_question_ids, False)      
        else:
            model = load_model(model_name, gpu_memory_utilization, tensor_parallel_size)
            print("working")
            num_batches = (len(prompts) + batch_size - 1) // batch_size
            for i in range(num_batches):
                batch_prompts = prompts[i * batch_size : (i + 1) * batch_size]
                batch_question_ids = question_ids[i * batch_size : (i + 1) * batch_size]
                get_responses(batch_prompts, batch_question_ids, model, model_name, output_dir, max_tokens, temperature)

        total_time = time.time() - start_time
        save_time_estimation(model_name,total_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run all models with specified parameters.')
    
    parser.add_argument('--output_dir', type=str, default="mt_bench_responses", help='Output directory')
    parser.add_argument('--model_names', type=str, default="vicuna-33b", help='Comma-separated list of model names')
    parser.add_argument('--path', type=str, default='mt_bench_questions.jsonl', help='Path to the input file')
    parser.add_argument('--openai_api', type=str, default='111', help='API Key')
    parser.add_argument('--tensor_parallel_size', type=int, default=2, help='Tensor parallel size')
    
    args = parser.parse_args()

    # print(args)

    fire.Fire(run_all_models(output_dir=args.output_dir, model_names=args.model_names, path=args.path, openai_api=args.openai_api, tensor_parallel_size=args.tensor_parallel_size))
