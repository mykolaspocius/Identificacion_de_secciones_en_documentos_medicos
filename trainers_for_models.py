from train_model import get_trainer
from transformers import (
    TrainingArguments,
)
from peft import LoraConfig,TaskType

# This file contains functions to get Trainer objecto to execute train() for the 5 models
# It this done this way, so it is possible to call this funcitons on Google Colab and execute te trainning on GPU

def get_trainer_M1(
        train_data_path="./ClinAIS_dataset/clinais.train.json",
        val_data_path="./ClinAIS_dataset/clinais.dev.json",
        output_dir='./models/model1'):
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
        output_dir='./models/model2'):
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
        output_dir='./models/model3'):
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
        output_dir='./models/model4'):
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

def get_trainer_M5(
        train_data_path="./ClinAIS_dataset/clinais.train.augmented.json",
        val_data_path="./ClinAIS_dataset/clinais.dev.json",
        output_dir='./models/model5'):
    return get_trainer(
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        base_model_id="PlanTL-GOB-ES/longformer-base-4096-biomedical-clinical-es",
        train_args=TrainingArguments(
            output_dir=output_dir,
            learning_rate=8.48e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=20,
            weight_decay=3.73e-03,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        ),
        adapter= LoraConfig(
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
    )

def get_trainer_M6(
        train_data_path="./ClinAIS_dataset/clinais.train.augmented.json",
        val_data_path="./ClinAIS_dataset/clinais.dev.json",
        output_dir='./models/model6'):
    return get_trainer(
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        base_model_id="PlanTL-GOB-ES/longformer-base-4096-biomedical-clinical-es",
        train_args=TrainingArguments(
            output_dir=output_dir,
            learning_rate=8.48e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=12,
            weight_decay=3.73e-03,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        ),
        adapter= LoraConfig(
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
                "value_global",
                "dense"
            ]
        )
    )
def get_trainer_M7(
        train_data_path="./ClinAIS_dataset/clinais.train.augmented.json",
        val_data_path="./ClinAIS_dataset/clinais.dev.json",
        output_dir='./models/model7'):
    return get_trainer(
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        base_model_id="/content/drive/MyDrive/Identificacion_de_secciones_en_documentos_medicos/models/model4/checkpoint-3910",
        train_args=TrainingArguments(
            output_dir=output_dir,
            learning_rate=8.48e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=12,
            weight_decay=3.73e-03,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        ),
        adapter= LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            inference_mode=False,
            r=16, 
            lora_alpha=16,
            lora_dropout=0.1,
            bias="all",
            target_modules=[
                "query",
                "key",
                "value"
            ]
        )
    )

def get_trainer_M8(
        train_data_path="./ClinAIS_dataset/clinais.train.augmented.json",
        val_data_path="./ClinAIS_dataset/clinais.dev.json",
        output_dir='./models/model8'):
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
        ),
        adapter= None,
        freez=True
    )
