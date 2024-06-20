from dataset_model import *
from data_preparation import create_dataset_object,tokenize_dataset_dict
from datasets import Dataset,DatasetDict


def execute_data_preparation_pipeline(train_data_path,validation_data_path,tokenizer,label2id)->DatasetDict:
    # load train an validation data
    with open(train_data_path) as f:
        train_data: ClinAISDataset = ClinAISDataset(**json.load(f))
    with open(validation_data_path) as f:
        validation_data: ClinAISDataset = ClinAISDataset(**json.load(f))

    # create dataset dict for train and validation splits
    train_dataset : Dataset = create_dataset_object(train_data,label2id)
    validation_dataset : Dataset = create_dataset_object(validation_data,label2id)
    dataset_dict = DatasetDict(train=train_dataset,val=validation_dataset)

    # tokenize and adjust entry lengths in the dataset_dict
    tokenized_dataset_dict = tokenize_dataset_dict(dataset_dict=dataset_dict,tokenizer=tokenizer)

    return tokenized_dataset_dict

