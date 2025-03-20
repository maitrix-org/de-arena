export OPENAI_API=""  
export OVERALL_IDS="81-102,103-121"
export SAVE_OUTPUT_FILE_PATH="mt_bench ranking result.txt"
export JUDGE_OPEN_MODEL=""  
export JUDGE_API_MODEL="o1-mini,o1-preview,ChatGPT-4o-latest,gpt-4o-2024-05-13,gpt-4o-2024-08-06,gpt-4-1106-preview,gpt-4-turbo-2024-04-09,gpt-3.5-turbo-0125,gpt-4o-mini-2024-07-18"
export BASE_MODEL_LIST="o1-mini,gpt-4o-mini-2024-07-18,gpt-3.5-turbo-0125"
export SORT_MODEL_LIST="o1-mini,o1-preview,ChatGPT-4o-latest,gpt-4o-2024-05-13,gpt-4o-2024-08-06,gpt-4-1106-preview,gpt-4-turbo-2024-04-09,gpt-3.5-turbo-0125,gpt-4o-mini-2024-07-18"

python automatic_arena.py 