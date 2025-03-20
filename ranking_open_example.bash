export OPENAI_API="" 
export OVERALL_IDS="81-102,103-121"
export SAVE_OUTPUT_FILE_PATH="mt_bench ranking result.txt"
export JUDGE_OPEN_MODEL="qwen2.5-3b,gemma-2-9b-it-simpo,google-gemma-2-9b-it,mistral-7b-instruct-2,llama-3.1-tulu-8b,zephyr-7b-beta,vicuna-7b"
export JUDGE_API_MODEL=""
export BASE_MODEL_LIST="qwen2.5-3b,gemma-2-9b-it-simpo,llama-3.1-tulu-8b"
export SORT_MODEL_LIST="qwen2.5-3b,gemma-2-9b-it-simpo,google-gemma-2-9b-it,mistral-7b-instruct-2,llama-3.1-tulu-8b,zephyr-7b-beta,vicuna-7b"

python automatic_arena.py 