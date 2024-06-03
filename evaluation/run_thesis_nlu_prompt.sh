# NLU Zero-Shot mT0 Eng
CUDA_VISIBLE_DEVICES=1 python main_nlu_prompt_batch.py eng bigscience/mt0-small 2048
CUDA_VISIBLE_DEVICES=1 python main_nlu_prompt_batch.py eng bigscience/mt0-base 1024
CUDA_VISIBLE_DEVICES=1 python main_nlu_prompt_batch.py eng bigscience/mt0-large 512
CUDA_VISIBLE_DEVICES=1 python main_nlu_prompt_batch.py eng bigscience/mt0-xl 256
CUDA_VISIBLE_DEVICES=1 python main_nlu_prompt_batch.py eng bigscience/mt0-xxl 128

# NLU Zero-Shot BLOOMZ Eng
CUDA_VISIBLE_DEVICES=1 python main_nlu_prompt_batch.py eng bigscience/bloomz-560m 2048
CUDA_VISIBLE_DEVICES=1 python main_nlu_prompt_batch.py eng bigscience/bloomz-1b1 1024
CUDA_VISIBLE_DEVICES=1 python main_nlu_prompt_batch.py eng bigscience/bloomz-1b7 512
CUDA_VISIBLE_DEVICES=1 python main_nlu_prompt_batch.py eng bigscience/bloomz-3b 256
CUDA_VISIBLE_DEVICES=1 python main_nlu_prompt_batch.py eng bigscience/bloomz-7b1 128

# NLU Zero-Shot Aya-101 Eng
CUDA_VISIBLE_DEVICES=1 python main_nlu_prompt_batch.py eng CohereForAI/aya-101 128

# NLU Zero-Shot MALA-500 Eng
CUDA_VISIBLE_DEVICES=1 python main_nlu_prompt_batch.py eng MaLA-LM/mala-500-10b-v2 128