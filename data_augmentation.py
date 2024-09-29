from dataset_model import *
from translation_pipline import * 
from typing import List
from predictions import split_entry,get_text_splits,getBoudaryAnnotationsForRange
from tqdm import tqdm
import os

# This function translates the entry by sections.
# It returns t2o arrays: 
# 1) list of lists of indicies that correspond to id's for each section
# 2) list of translations for each section independently
# The section in the original entry can be split in several parts
# in case it is longer then the maximum length accepted by translation model used.
def translate_entry(entry:Entry):
    max_accepted_length = int(0.7 * pipe_es_en.tokenizer.model_max_length)
    section2indeces = {}
    entry_sections_texts = []
    next_index = 0
    for i,section in enumerate(entry.section_annotation.gold):
        tokenized_section = pipe_es_en.tokenizer(section.segment,truncation=False)
        if (len(tokenized_section['input_ids'])>max_accepted_length):
            rdy=False
            ba_splits = [getBoudaryAnnotationsForRange(entry.boundary_annotation.gold,section.start_offset,section.end_offset)]
            while not rdy:
                rdy,ba_splits = split_entry(ba_splits,pipe_es_en.tokenizer,max_accepted_length)
            text_splits = get_text_splits(entry,ba_splits)
            entry_sections_texts+=text_splits
            section2indeces[i] = [i for i in range(next_index,next_index + len(text_splits))]
            next_index += len(text_splits)
        else:
            entry_sections_texts.append(section.segment)
            section2indeces[i] = [next_index]
            next_index+=1
    return section2indeces,apply_translation_pipeline(entry_sections_texts)

# not used in production
# just for testing porposes  
def translate_sections(entry_sections_texts:List[str])->List[str]:
    res = apply_translation_pipeline(entry_sections_texts)
    return res
    
# This function is ment to be called from Google Colab platform
# It takes two paths: original dataset path and path for translated dataset to be saved at
# The format of the result will be ClinAISDataset
# It might take many hours to execute this funciton (12 hours aprox.),
# because of this, with the porpose of not loosing the work done in case of any
# exception or error douring the execution, this function saves periodicly the progress
# and in case of restart, starts from where it has left
def translate_dataset_and_save(dataset_path,translated_dataset_path):

    if(not os.path.isfile(translated_dataset_path)):
        print("No existing translated data file found. Creating new...")
        with open(translated_dataset_path,'w',encoding='utf-8') as f:
            f.write(ClinAISDataset(annotated_entries = {}).toJson())

    print("Loading dataset for translation...")
    with open(dataset_path,encoding='utf-8') as f:
        dataset : ClinAISDataset = ClinAISDataset(**json.load(f))

    print("Loading dataset of translated data...")
    with open(translated_dataset_path,encoding='utf-8') as f:
        translated_dataset : ClinAISDataset = ClinAISDataset(**json.load(f))
    print(f"Translated dataset has {len(translated_dataset.annotated_entries.keys())} entries")

    print("Translating dataset...")
    for index,key in tqdm(enumerate(list(dataset.annotated_entries.keys()))):
        if (key in translated_dataset.annotated_entries.keys()):
            continue
        entry = dataset.annotated_entries[key]
        try:
            sec2indeces,translations = translate_entry(entry)

            translated_entry = Entry(note_id=entry.note_id,note_text=" ".join(translations))

            section_offset = 0
            for section_idx in list(sec2indeces.keys()):
                start_idx = sec2indeces[section_idx][0]
                end_idx = sec2indeces[section_idx][-1]
                section_text = " ".join(translations[start_idx:end_idx+1])
                section = SectionAnnotation(
                    segment=section_text,
                    label=entry.section_annotation.gold[section_idx].label,
                    start_offset=section_offset,
                    end_offset=section_offset+len(section_text))
                translated_entry.section_annotation.gold.append(section)

                section_spans = section_text.split(" ")
                span_offset = section_offset
                for i,span in enumerate(section_spans):
                    ba = BoundaryAnnotation(
                        span=span,
                        boundary=entry.section_annotation.gold[section_idx].label if i==0 else None,
                        start_offset=span_offset,
                        end_offset=span_offset+len(span))
                    translated_entry.boundary_annotation.gold.append(ba)
                    span_offset += len(span)+1
                section_offset += len(section_text)+1

            translated_dataset.annotated_entries[translated_entry.note_id] = translated_entry
            if (index%10==0):
                with open(translated_dataset_path,'w',encoding='utf-8') as f:
                    f.write(translated_dataset.toJson())

        except:
            print(f"Error ocured while processing entry {key}")
            with open(translated_dataset_path,'w',encoding='utf-8') as f:
                f.write(translated_dataset.toJson())
            continue
        
    print("Done translating")
    print("Saving translated data")
    with open(translated_dataset_path,'w',encoding='utf-8') as f:
        f.write(translated_dataset.toJson())
    print("Done")

def create_augmented_dataset(train_set_path,translated_train_set_path,save_path):
    with open(train_set_path,encoding='utf-8') as f:
        train_set : ClinAISDataset = ClinAISDataset(**json.load(f))

    with open(translated_train_set_path,encoding='utf-8') as f:
        translated_train_set : ClinAISDataset = ClinAISDataset(**json.load(f))


    augmented_set = ClinAISDataset(annotated_entries = {})
    for key,entry in train_set.annotated_entries.items():
        augmented_set.annotated_entries[key] = entry

    for key,entry in translated_train_set.annotated_entries.items():
        entry.note_id += '_T'
        augmented_set.annotated_entries[key+'_T'] = entry

    with open(save_path,'w',encoding='utf-8') as f:
        f.write(augmented_set.toJson())

# translate_dataset_and_save("./ClinAIS_dataset/clinais.train.json","./ClinAIS_dataset/clinais.train.translated.json")
# create_augmented_dataset("./ClinAIS_dataset/clinais.train.json","./ClinAIS_dataset/clinais.train.translated.json","./ClinAIS_dataset/clinais.train.augmented.json")
