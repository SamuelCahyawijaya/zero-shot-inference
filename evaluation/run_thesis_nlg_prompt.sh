# NLG Zero-Shot mT0 Eng
CUDA_VISIBLE_DEVICES=3 python main_nlg_prompt_batch.py eng bigscience/mt0-small 0 1024
CUDA_VISIBLE_DEVICES=3 python main_nlg_prompt_batch.py eng bigscience/mt0-base 0 512
CUDA_VISIBLE_DEVICES=3 python main_nlg_prompt_batch.py eng bigscience/mt0-large 0 256
CUDA_VISIBLE_DEVICES=3 python main_nlg_prompt_batch.py eng bigscience/mt0-xl 0 128
CUDA_VISIBLE_DEVICES=3 python main_nlg_prompt_batch.py eng bigscience/mt0-xxl 0 64

# NLG Zero-Shot BLOOMZ Eng
CUDA_VISIBLE_DEVICES=3 python main_nlg_prompt_batch.py eng bigscience/bloomz-560m 0 1024
CUDA_VISIBLE_DEVICES=3 python main_nlg_prompt_batch.py eng bigscience/bloomz-1b1 0 512
CUDA_VISIBLE_DEVICES=3 python main_nlg_prompt_batch.py eng bigscience/bloomz-1b7 0 256
CUDA_VISIBLE_DEVICES=3 python main_nlg_prompt_batch.py eng bigscience/bloomz-3b 0 128
CUDA_VISIBLE_DEVICES=3 python main_nlg_prompt_batch.py eng bigscience/bloomz-7b1 0 64

# NLG Zero-Shot Aya-101 Eng
CUDA_VISIBLE_DEVICES=3 python main_nlg_prompt_batch.py eng CohereForAI/aya-101 0 64

# NLG Zero-Shot MALA-500 Eng
CUDA_VISIBLE_DEVICES=3 python main_nlg_prompt_batch.py eng MaLA-LM/mala-500-10b-v2 0 64