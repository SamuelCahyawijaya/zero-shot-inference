from nusacrowd import NusantaraConfigHelper
from nusacrowd.utils.constants import Tasks
import pandas as pd
import datasets
from enum import Enum

NLU_TASK_LIST = [
    # Indo
    'emot',
    'wrete',
    'smsa',
    'nusax_senti_ind',
    'nusax_senti_eng',
    # Local
    'nusax_senti_ace', 'nusax_senti_ban', 'nusax_senti_bjn', 'nusax_senti_bbc', 'nusax_senti_bug', 
    'nusax_senti_jav', 'nusax_senti_mad', 'nusax_senti_min', 'nusax_senti_nij', 'nusax_senti_sun',
    'nusatranslation_senti_abs', 'nusatranslation_senti_btk', 'nusatranslation_senti_bew', 'nusatranslation_senti_bhp', 
    'nusatranslation_senti_jav', 'nusatranslation_senti_mad', 'nusatranslation_senti_mak', 'nusatranslation_senti_min', 
    'nusatranslation_senti_mui', 'nusatranslation_senti_rej', 'nusatranslation_senti_sun',
    'nusatranslation_emot_abd', 'nusatranslation_emot_btk', 'nusatranslation_emot_bew', 'nusatranslation_emot_bug',
    'nusatranslation_emot_jav', 'nusatranslation_emot_mad', 'nusatranslation_emot_mak', 'nusatranslation_emot_min', 
    'nusatranslation_emot_mui', 'nusatranslation_emot_rej', 'nusatranslation_emot_sun',
    'nusaparagraph_topic_btk', 'nusaparagraph_topic_bew', 'nusaparagraph_topic_bug', 'nusaparagraph_topic_jav', 'nusaparagraph_topic_mad',
    'nusaparagraph_topic_mak', 'nusaparagraph_topic_min', 'nusaparagraph_topic_mui', 'nusaparagraph_topic_rej', 'nusaparagraph_topic_sun',
    'nusaparagraph_rhetoric_btk', 'nusaparagraph_rhetoric_bew', 'nusaparagraph_rhetoric_bug', 'nusaparagraph_rhetoric_jav', 'nusaparagraph_rhetoric_mad',
    'nusaparagraph_rhetoric_mak', 'nusaparagraph_rhetoric_min', 'nusaparagraph_rhetoric_mui', 'nusaparagraph_rhetoric_rej', 'nusaparagraph_rhetoric_sun',
    'nusaparagraph_emot_btk', 'nusaparagraph_emot_bew', 'nusaparagraph_emot_bug', 'nusaparagraph_emot_jav', 'nusaparagraph_emot_mad',
    'nusaparagraph_emot_mak', 'nusaparagraph_emot_min', 'nusaparagraph_emot_mui', 'nusaparagraph_emot_rej', 'nusaparagraph_emot_sun',
]

NLU_TASK_LIST_EXTERNAL = [
    'MAPS',
    'haryoaw/COPAL',
    'MABL/id',
    'MABL/jv',
    'MABL/su',
    'IndoStoryCloze',
    'IndoMMLU',
    # 'MAPS/figurative',
    # 'MAPS/non_figurative',
]

NLG_TASK_LIST = [
    # Indo
    'ted_en_id',
    'nusax_mt_eng_ind',
    'nusax_mt_ind_eng',
    'liputan6_xtreme',
    'tydiqa_id',
    # Local
    'nusax_mt_ind_ace', 'nusax_mt_ind_ban', 'nusax_mt_ind_bjn', 'nusax_mt_ind_bbc', 'nusax_mt_ind_bug', 
    'nusax_mt_ind_jav', 'nusax_mt_ind_mad', 'nusax_mt_ind_min', 'nusax_mt_ind_nij', 'nusax_mt_ind_sun',
    'nusax_mt_ace_ind', 'nusax_mt_ban_ind', 'nusax_mt_bjn_ind', 'nusax_mt_bbc_ind', 'nusax_mt_bug_ind', 
    'nusax_mt_jav_ind', 'nusax_mt_mad_ind', 'nusax_mt_min_ind', 'nusax_mt_nij_ind', 'nusax_mt_sun_ind',
    'nusatranslation_mt_ind_abs', 'nusatranslation_mt_ind_btk', 'nusatranslation_mt_ind_bew', 'nusatranslation_mt_ind_bug', 
    'nusatranslation_mt_ind_jav', 'nusatranslation_mt_ind_mad', 'nusatranslation_mt_ind_mak', 'nusatranslation_mt_ind_min', 
    'nusatranslation_mt_ind_mui', 'nusatranslation_mt_ind_rej', 'nusatranslation_mt_ind_sun',
    'nusatranslation_mt_abs_ind', 'nusatranslation_mt_btk_ind', 'nusatranslation_mt_bew_ind', 'nusatranslation_mt_bug_ind',
    'nusatranslation_mt_jav_ind', 'nusatranslation_mt_mad_ind', 'nusatranslation_mt_mak_ind', 'nusatranslation_mt_min_ind', 
    'nusatranslation_mt_mui_ind', 'nusatranslation_mt_rej_ind', 'nusatranslation_mt_sun_ind',
]

FLORES200_TASK_LIST = [
    'flores200-sun_Latn-ind_Latn',
    'flores200-jav_Latn-ind_Latn',
    'flores200-bug_Latn-ind_Latn',
    'flores200-ace_Latn-ind_Latn',
    'flores200-bjn_Latn-ind_Latn',
    'flores200-ban_Latn-ind_Latn',
    'flores200-min_Latn-ind_Latn',
    'flores200-ind_Latn-sun_Latn',
    'flores200-ind_Latn-jav_Latn',
    'flores200-ind_Latn-bug_Latn',
    'flores200-ind_Latn-ace_Latn',
    'flores200-ind_Latn-bjn_Latn',
    'flores200-ind_Latn-ban_Latn',
    'flores200-ind_Latn-min_Latn',
]

def load_nlu_datasets():
    nc_conhelp = NusantaraConfigHelper()
    cfg_name_to_dset_map = {
        con.config.name: (con.load_dataset(), list(con.tasks)[0])
        for con in nc_conhelp.filtered(lambda x: x.config.name.replace(x.config.schema, '')[:-1] in NLU_TASK_LIST and 'nusantara_' in x.config.schema)
    } # {config_name: (datasets.Dataset, task_name)

    return cfg_name_to_dset_map

def load_external_nlu_datasets(lang='ind'):
    cfg_name_to_dset_map = {} # {config_name: (datasets.Dataset, task_name)

    # hack, add new Task
    class NewTasks(Enum):
        COPA = "COPA"
        MABL = "MABL"
        MAPS = "MAPS"
        IndoStoryCloze = "IndoStoryCloze"
        IndoMMLU = "IndoMMLU"

    for task in NLU_TASK_LIST_EXTERNAL:
        if 'COPAL' in task:
            dset = datasets.load_dataset(task)
            cfg_name_to_dset_map[task] = (dset, NewTasks.COPA)
        elif 'MABL' in task:
            mabl_path = './mabl_data'
            subset = task.split('/')[-1]
            
            df = pd.read_csv(f'{mabl_path}/{subset}.csv')
            dset = datasets.Dataset.from_pandas(
                df.rename({'startphrase': 'premise', 'ending1': 'choice1', 'ending2': 'choice2', 'labels': 'label'}, axis='columns')
            )
            cfg_name_to_dset_map[task] = (datasets.DatasetDict({'test': dset}), NewTasks.MABL)
        elif 'MAPS' in task:
            maps_path = './maps_data'
            df = pd.read_excel(f'{maps_path}/test_proverbs.xlsx')
            
            # Split by subset
            if '/' in task:
                subset = task.split('/')[-1]
                if subset =='figurative':
                    df = df.loc[df['is_figurative'] == 1,:]
                else: # non_figurative
                    df = df.loc[df['is_figurative'] == 0,:]

            dset = datasets.Dataset.from_pandas(
                df.rename({
                    'proverb': 'premise', 'conversation': 'context', 
                    'answer1': 'choice1', 'answer2': 'choice2', 'answer_key': 'label'
                }, axis='columns')
            )
            cfg_name_to_dset_map[task] = (datasets.DatasetDict({'test': dset}), NewTasks.MAPS)
        elif 'IndoStoryCloze' in task:
            df = datasets.load_dataset('indolem/indo_story_cloze')['test'].to_pandas()
            
            # Preprocess
            df['premise'] = df.apply(lambda x: '. '.join([
                x['sentence-1'], x['sentence-2'], x['sentence-3'], x['sentence-4']
            ]), axis='columns')
            df = df.rename({'correct_ending': 'choice1', 'incorrect_ending': 'choice2'}, axis='columns')
            df = df[['premise', 'choice1', 'choice2']]
            df['label'] = 0
            
            dset = datasets.Dataset.from_pandas(df)
            cfg_name_to_dset_map[task] = (datasets.DatasetDict({'test': dset}), NewTasks.IndoStoryCloze)
        elif 'IndoMMLU' in task:
            df = pd.read_csv('indommlu_data/IndoMMLU.csv')
            dset = datasets.Dataset.from_pandas(df.rename({'kunci': 'label'}, axis='columns'))
            cfg_name_to_dset_map[task] = (datasets.DatasetDict({'test': dset}), NewTasks.IndoMMLU)
    return cfg_name_to_dset_map

def load_nlg_datasets():
    nc_conhelp = NusantaraConfigHelper()
    cfg_name_to_dset_map = {
        con.config.name: (con.load_dataset(), list(con.tasks)[0])
        for con in nc_conhelp.filtered(lambda x: x.config.name.replace(x.config.schema, '')[:-1] in NLG_TASK_LIST and 'nusantara_' in x.config.schema)
    } # {config_name: (datasets.Dataset, task_name)

    # Hack
    for k, (dset, task) in cfg_name_to_dset_map.items():
        if k == 'ted_en_id':
            def swap_text(row):
                tmp, tmp_name = row['text_1'], row['text_1_name']
                row['text_1'], row['text_1_name'] = row['text_2'], row['text_2_name']
                row['text_2'], row['text_2_name'] = tmp, tmp_name
                return row
            rev_dset = dset.map(swap_text)
            cfg_name_to_dset_map['ted_id_en'] = rev_dset
            print(rev_dset['text'][0])
    return cfg_name_to_dset_map

def load_flores_datasets():
    dset_map = {}
    for task in FLORES200_TASK_LIST:
        subset = task.replace('flores200-', '')
        src_lang, tgt_lang = subset.split('-')
        dset = datasets.load_dataset('facebook/flores', subset)
        dset = dset.rename_columns({f'sentence_{src_lang}': 'text_1', f'sentence_{tgt_lang}': 'text_2'}).select_columns(['id', 'text_1', 'text_2'])
        dset_map[task] = (dset, Tasks.MACHINE_TRANSLATION)
    return dset_map

def load_truthfulqa_datasets():
    class NewTasks(Enum):
        TRUTHFULQA = "TRUTHFULQA"
    return {'truthfulqa': (datasets.load_from_disk('./truthfulqa_ind'), NewTasks.TRUTHFULQA)}