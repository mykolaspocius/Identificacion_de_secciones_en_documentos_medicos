from dataset_model import *
from data_preparation_pipeline import execute_data_preparation_pipeline
from misc import create_label_id_dictionaries
from metrics import get_seqeval_metrics
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    Trainer
)

def get_trainer(base_model_id:str,train_data_path:str,val_data_path:str,train_args:TrainingArguments)->Trainer:
    hf_token="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"
    # create label->number and number->label relations
    label2id,id2label = create_label_id_dictionaries(ClinicalSections.list())
    # get model tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=base_model_id,token=hf_token)
    #prepear the data
    dataset_dict = execute_data_preparation_pipeline(
        train_data_path=train_data_path,
        validation_data_path=val_data_path,
        tokenizer=tokenizer,
        label2id=label2id)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    model = AutoModelForTokenClassification.from_pretrained(base_model_id,num_labels=len(label2id),id2label=id2label,label2id=label2id,token=hf_token)

    return Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["val"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=get_seqeval_metrics(id2label)
    )


