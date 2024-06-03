"""nusacrowd zero-shot prompt.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ru8DyS2ALWfRdkjOPHj-KNjw6Pfa44Nd
"""
import os, sys
import csv
from os.path import exists

from numpy import argmax, stack
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, precision_recall_fscore_support

from nusacrowd import NusantaraConfigHelper
from nusacrowd.utils.constants import Tasks

from prompt_utils import get_prompt, get_label_mapping
from data_utils import load_nlu_datasets

import openai
import cohere

#!pip install git+https://github.com/IndoNLP/nusa-crowd.git@release_exp
#!pip install transformers
#!pip install sentencepiece

DEBUG=False

def to_prompt(input, prompt, labels, prompt_lang):
    # single label
    if 'text' in input:
        prompt = prompt.replace('[INPUT]', input['text'])
    else:
        prompt = prompt.replace('[INPUT_A]', input['text_1'])
        prompt = prompt.replace('[INPUT_B]', input['text_2'])

    # replace [OPTIONS] to A, B, or C
    if "[OPTIONS]" in prompt:
        new_labels = [f'{l}' for l in labels]
        new_labels[-1] = ("or " if 'eng' in prompt_lang else  "atau ") + new_labels[-1] 
        if len(new_labels) > 2:
            prompt = prompt.replace('[OPTIONS]', ', '.join(new_labels))
        else:
            prompt = prompt.replace('[OPTIONS]', ' '.join(new_labels))

    return prompt

###
# API
###
def openai_api(openai_client, input_text, system_text = '', model_name = 'gpt-3.5-turbo', max_tokens=16):
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

def cohere_api(cohere_client, input_text, system_text = '', model_name='command-r', max_tokens=16):
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
    
    if len(sys.argv) < 4:
        raise ValueError('main_nlu_prompt.py <prompt_lang> <model_path_or_name> <batch_size> <save_every (OPTIONAL)>')

    prompt_lang = sys.argv[1]
    MODEL = sys.argv[2]
    BATCH_SIZE = 1
    
    if 'gpt' in MODEL:
        CLIENT = 'openai'
    else:
        CLIENT = 'cohere'

    SAVE_EVERY = 10

    out_dir = './outputs_nlu'
    metric_dir = './metrics_nlu'
    os.makedirs(out_dir, exist_ok=True) 
    os.makedirs(metric_dir, exist_ok=True) 

    if os.path.exists(f'{metric_dir}/nlu_results_{prompt_lang}_{MODEL.split("/")[-1]}.csv'):
        print(f'Skipping {metric_dir}/nlu_results_{prompt_lang}_{MODEL.split("/")[-1]}.csv')
        sys.exit(0)

    # Load Prompt
    TASK_TYPE_TO_PROMPT = get_prompt(prompt_lang)

    # Load Dataset
    print('Load NLU Datasets...')
    nlu_datasets = load_nlu_datasets()

    print(f'Loaded {len(nlu_datasets)} NLU datasets')
    for i, dset_subset in enumerate(nlu_datasets.keys()):
        print(f'{i} {dset_subset}')

    metrics = []
    labels = []
    for i, dset_subset in enumerate(nlu_datasets.keys()):
        print(f'{i} {dset_subset}')
        nlu_dset, task_type = nlu_datasets[dset_subset]
        if task_type.value not in TASK_TYPE_TO_PROMPT:
            print(f'SKIPPING {dset_subset}')
            continue

        # Retrieve metadata
        split = 'test'
        if 'test' in nlu_dset.keys():
            test_dset = nlu_dset['test']
        else:
            test_dset = nlu_dset['train']
            split = 'train'
        print(f'Processing {dset_subset}')

        # Retrieve & preprocess labels
        try:
            label_names = test_dset.features['label'].names
        except:
            label_names = list(set(test_dset['label']))
            
        # normalize some labels for more natural prompt:
        label_mapping = get_label_mapping(dset_subset, prompt_lang)
        label_names = list(map(lambda x: label_mapping[x], label_mapping))

        label_to_id_dict = { l : i for i, l in enumerate(label_names)}
        
        for prompt_id, prompt_template in enumerate(TASK_TYPE_TO_PROMPT[task_type.value]):
            inputs, preds, outs, golds = [], [], [], []
            
            # Check saved data
            if exists(f'{out_dir}/{dset_subset}_{prompt_lang}_{prompt_id}_{MODEL.split("/")[-1]}.csv'):
                print("Output exist, use partial log instead")
                with open(f'{out_dir}/{dset_subset}_{prompt_lang}_{prompt_id}_{MODEL.split("/")[-1]}.csv') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        inputs.append(row["Input"])
                        preds.append(row["Pred"])
                        outs.append(row["Out"])
                        golds.append(row["Gold"])
                print(f"Skipping until {len(preds)}")

            # sample prompt
            print("= LABEL NAME =")
            print(label_names)
            print("= SAMPLE PROMPT =")
            
            print(to_prompt(test_dset[0], prompt_template, label_names, prompt_lang))
            print("\n")

            # zero-shot inference
            count = 0
            for e, sample in tqdm(enumerate(test_dset)):
                if e < len(preds):
                    continue

                # Single Instance Inference
                prompt_text = to_prompt(sample, prompt_template, label_names, prompt_lang)
                prompt_text = prompt_text.replace('[LABELS_CHOICE]', '')
                label = label_to_id_dict[sample['label']] if type(sample['label']) == str else sample['label']

                if CLIENT == 'cohere':
                    out = cohere_api(cohere_client, prompt_text, system_text = '', model_name='command-r', max_tokens=16)
                else: # client == 'openai'
                    out = openai_api(openai_client, prompt_text, system_text = '', model_name = 'gpt-3.5-turbo', max_tokens=16)

                pred = out
                for i, label in enumerate(label_names):
                    if label in out.lower():
                        pred = label
                        break

                inputs.append(prompt_text)
                preds.append(pred)
                outs.append(out)
                golds.append(label)
                count += 1

                if count == SAVE_EVERY:
                    # partial saving
                    inference_df = pd.DataFrame(list(zip(inputs, preds, outs, golds)), columns =["Input", 'Pred', 'Out', 'Gold'])
                    inference_df.to_csv(f'{out_dir}/{dset_subset}_{prompt_lang}_{prompt_id}_{MODEL.split("/")[-1]}.csv', index=False)
                    count = 0
                        
            # partial saving
            inference_df = pd.DataFrame(list(zip(inputs, preds, outs, golds)), columns =["Input", 'Pred', 'Out', 'Gold'])
            inference_df.to_csv(f'{out_dir}/{dset_subset}_{prompt_lang}_{prompt_id}_{MODEL.split("/")[-1]}.csv', index=False)

            cls_report = classification_report(golds, preds, output_dict=True)
            micro_f1, micro_prec, micro_rec, _ = precision_recall_fscore_support(golds, preds, average='micro')
            print(dset_subset)
            print('accuracy', cls_report['accuracy'])
            print('f1 micro', micro_f1)
            print('f1 macro', cls_report['macro avg']['f1-score'])
            print('f1 weighted', cls_report['weighted avg']['f1-score'])
            print("===\n\n")       

            metrics.append({
                'dataset': dset_subset,
                'prompt_id': prompt_id,
                'prompt_lang': prompt_lang,
                'accuracy': cls_report['accuracy'], 
                'micro_prec': micro_prec,
                'micro_rec': micro_rec,
                'micro_f1_score': micro_f1,
                'macro_prec': cls_report['macro avg']['precision'],
                'macro_rec': cls_report['macro avg']['recall'],
                'macro_f1_score': cls_report['macro avg']['f1-score'],
                'weighted_prec': cls_report['weighted avg']['precision'],
                'weighted_rec': cls_report['weighted avg']['recall'],
                'weighted_f1_score': cls_report['weighted avg']['f1-score'],
            })

    pd.DataFrame(metrics).reset_index().to_csv(f'{metric_dir}/nlu_results_{prompt_lang}_{MODEL.split("/")[-1]}.csv', index=False)
