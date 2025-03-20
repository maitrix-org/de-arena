q_set='mt_bench'
path="mt_bench_questions.jsonl"
model_name="vicuna-7b"
model_names="qwen2.5-3b,gemma-2-9b-it-simpo,google-gemma-2-9b-it,mistral-7b-instruct-2,llama-3.1-tulu-8b,zephyr-7b-beta,vicuna-7b"

CUDA_VISIBLE_DEVICES=1 python judge_responses.py --path ${path} --model_name ${model_name} --model_names ${model_names} --tensor_parallel_size 1 --dimension ${q_set}
