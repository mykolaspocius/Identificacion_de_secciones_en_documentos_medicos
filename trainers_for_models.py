from train_model import get_trainer
from transformers import (
    TrainingArguments,
)

# This file contains functions to get Trainer objecto to execute train() for the 5 models
# It this done this way, so it is possible to call this funcitons on Google Colab and execute te trainning on GPU

def get_trainer_M1(
        train_data_path="./ClinAIS_dataset/clinais.train.json",
        val_data_path="./ClinAIS_dataset/clinais.dev.json",
        output_dir='./finetuned_models/model1'):
    return get_trainer(
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        base_model_id="PlanTL-GOB-ES/bsc-bio-ehr-es",
        train_args=TrainingArguments(
            output_dir=output_dir,
            learning_rate=1e-4,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=10,
            weight_decay=0,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
    )

def get_trainer_M2(
        train_data_path="./ClinAIS_dataset/clinais.train.json",
        val_data_path="./ClinAIS_dataset/clinais.dev.json",
        output_dir='./finetuned_models/model2'):
    return get_trainer(
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        base_model_id="PlanTL-GOB-ES/longformer-base-4096-biomedical-clinical-es",
        train_args=TrainingArguments(
            output_dir=output_dir,
            learning_rate=8.48e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=10,
            weight_decay=3.73e-03,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
    )

def get_trainer_M3(
        train_data_path="./ClinAIS_dataset/clinais.train.augmented.json",
        val_data_path="./ClinAIS_dataset/clinais.dev.json",
        output_dir='./finetuned_models/model3'):
    return get_trainer(
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        base_model_id="PlanTL-GOB-ES/bsc-bio-ehr-es",
        train_args=TrainingArguments(
            output_dir=output_dir,
            learning_rate=1e-4,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=10,
            weight_decay=0,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
    )

def get_trainer_M4(
        train_data_path="./ClinAIS_dataset/clinais.train.augmented.json",
        val_data_path="./ClinAIS_dataset/clinais.dev.json",
        output_dir='./finetuned_models/model4'):
    return get_trainer(
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        base_model_id="PlanTL-GOB-ES/longformer-base-4096-biomedical-clinical-es",
        train_args=TrainingArguments(
            output_dir=output_dir,
            learning_rate=8.48e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=10,
            weight_decay=3.73e-03,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
    )