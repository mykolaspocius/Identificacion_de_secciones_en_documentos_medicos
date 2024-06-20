from typing import List
from dataset_model import *
from datasets import Dataset,DatasetDict
import pandas as pd
from WordListSplitter import WordListSplitter

def get_labelled_span_list(b_annotations : BoundaryAnnotations):
    cur_label = [None] 
    def get_label(boundary):
        if (boundary != None):
            cur_label[0] = boundary
        return cur_label
    return [(b_annotation.span, get_label(b_annotation.boundary)[0]) for b_annotation in b_annotations]

def create_dataset_object(data_set : ClinAISDataset,label2id : dict)->Dataset:
    spans_labels = []
    for _,entry in data_set.annotated_entries.items():
        labeled_spans = get_labelled_span_list(entry.boundary_annotation.gold)
        # asign id's to labels
        labeled_spans = [(span,label2id[label]) for span,label in labeled_spans]
        spans_labels.append(zip(*labeled_spans))

    return Dataset.from_pandas(pd.DataFrame(data=spans_labels,columns=['spans','labels']))

# Helper function used in split_entry
# Takes as input two lists:
# 1. target_list. The list of dimenssion 1
# 2. patern_list. The list of dimenssion 2
#       this second list is a list of lists of dimenssion 1 each with some defined length
# The second list, if flattened, needs to have same number of elements as target_list
# Returns a list containning elements of the target_list but reshaped into a dimnession 2 list with sublists of the same size as patern_list
def get_reshaped_list(target_list,patern_list)->List:
    result = []
    cur_pos = 0
    for sublist in patern_list:
        result.append(target_list[cur_pos:cur_pos+len(sublist)])
        cur_pos+=len(sublist)
    return result

def split_entry(entry : dict[List[str],List[int]],word_splitter:WordListSplitter):
    parts = word_splitter.split(entry['spans'])
    labels = get_reshaped_list(target_list=entry['labels'],patern_list=parts)   
    return {'spans':parts,'labels':labels}

def tokenize_split_align(batch,tokenizer)->dict:
    tokenized_inputs = tokenizer(batch['spans'],truncation=False,is_split_into_words=True) # tokenize batch
    splits_batch = {'spans':[],'labels':[],'input_ids':[],'attention_mask':[],'aligned_labels':[]}
    for i,input_ids in enumerate(tokenized_inputs['input_ids']): # for each entry in the batch:
        if (len(input_ids)>tokenizer.model_max_length): # check entry length
            max_length = int((tokenizer.model_max_length/len(input_ids))*len(batch['spans'][i])) # set reduction factor
            splits = split_entry({'spans':batch['spans'][i],'labels':batch['labels'][i]},WordListSplitter(max_size=max_length,min_size=3)) # split entry
            for spans,labels in zip(splits['spans'],splits['labels']): # add new splits to the dataset
                splits_batch['spans'].append(spans)
                splits_batch['labels'].append(labels)
                splits_batch['input_ids'].append([])
                splits_batch['attention_mask'].append([])
                splits_batch['aligned_labels'].append([])          
        else: # tokenize,align entry data
            splits_batch['spans'].append(batch['spans'][i])
            splits_batch['labels'].append(batch['labels'][i])
            splits_batch['input_ids'].append(input_ids)
            splits_batch['attention_mask'].append(tokenized_inputs['attention_mask'][i])
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(batch['labels'][i][word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            splits_batch['aligned_labels'].append(label_ids) 
    return splits_batch
 
   
def tokenize_dataset_dict(dataset_dict : DatasetDict,tokenizer)->DatasetDict:
    def process_batch(batch):
        return tokenize_split_align(batch,tokenizer)

    tokenized_dataset_dict = dataset_dict.map(process_batch, batched=True) # process batch

    for split in tokenized_dataset_dict: # for each split in a dataset_dict
        for example in tokenized_dataset_dict[split]['spans']: # for each example check the length and if too long make another pass
            if len(tokenizer.encode(example, truncation=False,is_split_into_words=True)) > tokenizer.model_max_length:
                return tokenize_dataset_dict(tokenized_dataset_dict, tokenizer)
     
    def rename_columns(entry)->dict:
        return {
            'input_ids':entry['input_ids'],
            'labels' : entry['aligned_labels'],
            'attention_mask' : entry['attention_mask']
        }
    return tokenized_dataset_dict.map(rename_columns,remove_columns=['aligned_labels','spans']) # rename columns and drop the unnecesary ones
