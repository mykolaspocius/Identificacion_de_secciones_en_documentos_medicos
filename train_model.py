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
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType

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

def get_peft_trainer(base_model_id:str,train_data_path:str,val_data_path:str,train_args:TrainingArguments):
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
    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        inference_mode=False,
        r=16, 
        lora_alpha=16,
        lora_dropout=0.1,
        bias="all",
        target_modules=[
            "query",
            "key",
            "value",
            "query_global",
            "key_global",
            "value_global"
        ]
    )
    model = get_peft_model(model, peft_config)
    # model.print_trainable_parameters()
    return Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["val"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=get_seqeval_metrics(id2label)
    )


