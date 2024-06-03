import os, sys
import csv
from os.path import exists

import pandas as pd
from tqdm import tqdm
from prompt_utils import get_prompt, get_lang_name
from data_utils import load_nlg_datasets

from nusacrowd.utils.constants import Tasks

from sacremoses import MosesTokenizer
import datasets
from anyascii import anyascii
import openai
import cohere
from retry import retry

openai.api_key = ""

DEBUG=True

""" Generation metrics """
bleu = datasets.load_metric('bleu')
rouge = datasets.load_metric('rouge')
sacrebleu = datasets.load_metric('sacrebleu')
chrf = datasets.load_metric('chrf')
squad_v2_metric = datasets.load_metric('squad_v2')
mt = MosesTokenizer(lang='id')

def generation_metrics_fn(list_hyp, list_label):
    # hyp and label are both list of string
    list_hyp_bleu = list(map(lambda x: mt.tokenize(x), list_hyp))
    list_label_bleu = list(map(lambda x: [mt.tokenize(x)], list_label))    
    list_label_sacrebleu = list(map(lambda x: [x], list_label))
    
    metrics = {}
    metrics["BLEU"] = bleu._compute(list_hyp_bleu, list_label_bleu)['bleu'] * 100
    metrics["SacreBLEU"] = sacrebleu._compute(list_hyp, list_label_sacrebleu)['score']
    metrics["chrF++"] = chrf._compute(list_hyp, list_label_sacrebleu)['score']
    
    rouge_score = rouge._compute(list_hyp, list_label)
    metrics["ROUGE1"] = rouge_score['rouge1'].mid.fmeasure * 100
    metrics["ROUGE2"] = rouge_score['rouge2'].mid.fmeasure * 100
    metrics["ROUGEL"] = rouge_score['rougeL'].mid.fmeasure * 100
    metrics["ROUGELsum"] = rouge_score['rougeLsum'].mid.fmeasure * 100
    
    return metrics

def to_prompt(input, prompt, prompt_lang, task_name, task_type, with_label=False):
    if '[INPUT]' in prompt:
        prompt = prompt.replace('[INPUT]', input['text_1'])

    if task_type == Tasks.MACHINE_TRANSLATION.value:
        # Extract src and tgt based on nusantara config name
        task_names = task_name.split('_')
        src_lang = task_names[-4]
        tgt_lang = task_names[-3]

        # Replace src and tgt lang name
        prompt = prompt.replace('[SOURCE]', get_lang_name(prompt_lang, src_lang))
        prompt = prompt.replace('[TARGET]', get_lang_name(prompt_lang, tgt_lang))
    
    if task_type == Tasks.QUESTION_ANSWERING.value:
        prompt = prompt.replace('[CONTEXT]', input['context'])
        prompt = prompt.replace('[QUESTION]', input['question'])
    
    if with_label:
        if task_type == Tasks.QUESTION_ANSWERING.value:
            prompt += " " + input['answer'][0]
        else:
            prompt += " " + input['text_2']
    
    return prompt

###
# API
###
def openai_api(openai_client, input_text, system_text = '', model_name = 'gpt-3.5-turbo', max_tokens=128):
    completion = openai_client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system", 
                "content": system_text,
                "role": "user",
                "content": input_text,
            },
        ],
        max_tokens=max_tokens
    )
    return completion.choices[0].message.content

def cohere_api(cohere_client, input_text, system_text = '', model_name='command-r', max_tokens=128):
    response = cohere_client.chat(
        model=model_name,
        chat_history=[],
        message=input_text,
        preamble=system_text,
        connectors=[],
        max_tokens=max_tokens
    )
    return response.text
    
if __name__ == '__main__':
    openai_client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    cohere_client = cohere.Client(os.environ['COHERE_API_KEY'])
    if len(sys.argv) != 5:
        raise ValueError('main_nlg_prompt.py <prompt_lang> <model_path_or_name> <n_shot> <n_batch>')

    # TODO: reduce hardcoded vars
    out_dir = './outputs_nlg'
    metric_dir = './metrics_nlg'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)

    prompt_lang = sys.argv[1]
    MODEL = sys.argv[2]
    N_SHOT = int(sys.argv[3])
    N_BATCH = int(sys.argv[4])
    SAVE_EVERY = 10
    
    if 'gpt' in MODEL:
        CLIENT = 'openai'
    else:
        CLIENT = 'cohere'

    # Load prompt
    prompt_templates = get_prompt(prompt_lang)

    if os.path.exists(f'{metric_dir}/nlg_results_{prompt_lang}_{N_SHOT}_{MODEL.split("/")[-1]}.csv'):
        print(f'Skipping {metric_dir}/nlg_results_{prompt_lang}_{N_SHOT}_{MODEL.split("/")[-1]}.csv')    
        sys.exit(0)
    
    # Load Dataset
    print('Load NLG Datasets...')
    nlg_datasets = load_nlg_datasets()

    print(f'Loaded {len(nlg_datasets)} NLG datasets')
    for i, dset_subset in enumerate(nlg_datasets.keys()):
        print(f'{i} {dset_subset}')

    metrics = {'dataset': []}
    for i, dset_subset in enumerate(nlg_datasets.keys()):
        nlg_dset, task_type = nlg_datasets[dset_subset]
        print(f"{i} {dset_subset} {task_type}")
        
        if task_type.value not in prompt_templates or nlg_dset is None:
            continue

        if 'test' in nlg_dset.keys():
            data = nlg_dset['test']
        elif 'validation' in nlg_dset.keys():
            data = nlg_dset['validation']
        else:
            data = nlg_dset['train']
        few_shot_data = nlg_dset['train']

        for prompt_id, prompt_template in enumerate(prompt_templates[task_type.value]):
            if len(prompt_id) > 0:
                break
            inputs = []
            preds = []
            preds_latin = []
            golds = []  
            print(f"PROMPT ID: {prompt_id}")
            print(f"SAMPLE PROMPT: {to_prompt(data[0], prompt_template, prompt_lang, dset_subset, task_type.value)}")

            few_shot_text_list = []
            if N_SHOT > 0:
                for sample in tqdm(few_shot_data):
                    # Skip shot examples
                    if task_type != Tasks.QUESTION_ANSWERING and len(sample['text_1']) < 20:
                        continue
                    few_shot_text_list.append(
                        to_prompt(sample, prompt_template, dset_subset, task_type.value, with_label=True)
                    )
                    if len(few_shot_text_list) == N_SHOT:
                        break
            print(f'FEW SHOT SAMPLES: {few_shot_text_list}')
            
            # Zero-shot inference
            if exists(f'{out_dir}/{dset_subset}_{prompt_lang}_{prompt_id}_{N_SHOT}_{MODEL.split("/")[-1]}.csv'):        
                print("Output exist, use existing log instead")
                with open(f'{out_dir}/{dset_subset}_{prompt_lang}_{prompt_id}_{N_SHOT}_{MODEL.split("/")[-1]}.csv') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        inputs.append(row["Input"])
                        preds.append(row["Pred"])
                        preds_latin.append(row["Pred_Latin"])
                        golds.append(row["Gold"])
                print(f"Skipping until {len(preds)}")

            # If incomplete, continue
            if len(preds) < len(data):
                count = 0
                for e, sample in enumerate(tqdm(data)):
                    if e < len(preds):
                        continue

                    # Buffer
                    prompt_text = to_prompt(sample, prompt_template, prompt_lang, dset_subset, task_type.value)
                    label = sample['answer'][0] if task_type == Tasks.QUESTION_ANSWERING else sample['text_2']

                    # Single Instance inference
                    if CLIENT == 'cohere':
                        pred = cohere_api(cohere_client, prompt_text, system_text = '', model_name='command-r', max_tokens=128)
                    else: # client == 'openai'
                        pred = openai_api(openai_client, prompt_text, system_text = '', model_name = 'gpt-3.5-turbo', max_tokens=128)

                    inputs.append(prompt_text)
                    preds.append(pred)
                    preds_latin.append(anyascii(pred))
                    golds.append(label)
                    count += 1

                    if count == SAVE_EVERY:
                        # partial saving
                        inference_df = pd.DataFrame(list(zip(inputs, preds, preds_latin, golds)), columns=['Input', 'Pred', 'Pred_Latin', 'Gold'])
                        inference_df.to_csv(f'{out_dir}/{dset_subset}_{prompt_lang}_{prompt_id}_{N_SHOT}_{MODEL.split("/")[-1]}.csv', index=False)
                        count = 0
            
            # Final save
            inference_df = pd.DataFrame(list(zip(inputs, preds, preds_latin, golds)), columns=['Input', 'Pred', 'Pred_Latin', 'Gold'])
            inference_df.to_csv(f'{out_dir}/{dset_subset}_{prompt_lang}_{prompt_id}_{N_SHOT}_{MODEL.split("/")[-1]}.csv', index=False)

            # To accomodate old bug where list are not properly re-initiated
            inputs = inputs[-len(data):]
            preds = preds[-len(data):]
            preds_latin = preds_latin[-len(data):]
            golds = golds[-len(data):]

            eval_metric = generation_metrics_fn(preds, golds)
            eval_metric_latin = generation_metrics_fn(preds_latin, golds)
            for key, value in eval_metric_latin.items():
                eval_metric[f'{key}_latin'] = value

            print(f'== {dset_subset} == ')
            for k, v in eval_metric.items():
                print(k, v)            
            print("===\n\n")
            eval_metric['prompt_id'] = prompt_id

            metrics['dataset'].append(dset_subset)
            for k in eval_metric:
                if k not in metrics:
                    metrics[k] = []
                metrics[k].append(eval_metric[k])


    pd.DataFrame.from_dict(metrics).reset_index().to_csv(f'{metric_dir}/nlg_results_{prompt_lang}_{N_SHOT}_{MODEL.split("/")[-1]}.csv', index=False)
