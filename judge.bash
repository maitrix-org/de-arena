export CUDA_VISIBLE_DEVICES=6,7
python judge_responses_new.py --path "mt_bench_questions.jsonl" --model_name "llama2-13b-chat" --tensor_parallel_size 2