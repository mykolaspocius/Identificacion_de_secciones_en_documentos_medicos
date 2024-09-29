# This file contains function used to prepear a data corpus for passing it to Trainer.
# It takes the following steps:
# 1. Load the data into ClinAISDataset structures
# 2. Create Dataset objects for train and validation datasets and put them into DatasetDict object
#       in this step tokens are labbeled by using label2id function, which
#       asigns numeric id to each label used in the model.
# 3. Tokenize bouth datasets using tokenizer past as a parameter
#       in this step the spans contained in each example are tokenized into tokens used by a tokenizer.
#       Examples that are longer than max axepted length by the model are split into smaller ones.
#       This process is repeated several times, until all examples are of an acceptable length.
from dataset_model import *
from data_preparation import create_dataset_object,tokenize_dataset_dict
from datasets import Dataset,DatasetDict


def execute_data_preparation_pipeline(train_data_path,validation_data_path,tokenizer,label2id)->DatasetDict:
    # load train an validation data
    with open(train_data_path,encoding='utf-8') as f:
        train_data: ClinAISDataset = ClinAISDataset(**json.load(f))
    with open(validation_data_path,encoding='utf-8') as f:
        validation_data: ClinAISDataset = ClinAISDataset(**json.load(f))

    # create dataset dict for train and validation splits
    train_dataset : Dataset = create_dataset_object(train_data,label2id)
    validation_dataset : Dataset = create_dataset_object(validation_data,label2id)
    dataset_dict = DatasetDict(train=train_dataset,val=validation_dataset)

    # tokenize and adjust entry lengths in the dataset_dict
    tokenized_dataset_dict = tokenize_dataset_dict(dataset_dict=dataset_dict,tokenizer=tokenizer)

    return tokenized_dataset_dict

