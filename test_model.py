from predictions import create_predictions_file
from transformers import pipeline,AutoModelForTokenClassification,AutoTokenizer
from pathlib import Path
import sys
sys.path.append("./evaluation")
from evaluation.metricas import score_predictions
from peft import LoraConfig, TaskType, get_peft_model
from misc import create_label_id_dictionaries
from dataset_model import *

def test(finetuned_model_path,test_dataset_path,save_predictions_path,save_evaluated_path,peft_config=None):
    print("Creating predictions for dataset...")
    if(peft_config!=None):
        label2id,id2label = create_label_id_dictionaries(ClinicalSections.list())
        base_model = AutoModelForTokenClassification.from_pretrained(finetuned_model_path,num_labels=7,id2label=id2label,label2id=label2id)
        tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)
        peft_model = get_peft_model(base_model, peft_config)
        modelo = pipeline(task="token-classification", model=peft_model, tokenizer=tokenizer,aggregation_strategy="simple")
    else:
        modelo = pipeline( "token-classification", model=finetuned_model_path, aggregation_strategy="simple")
    create_predictions_file(test_dataset_path,save_predictions_path,modelo)
    print("Predictions created and saved to file")


    print("Evaluating predictions...")
    prediction_file = Path(save_predictions_path)
    output_file = Path(save_evaluated_path)
    score_predictions(prediction_file=prediction_file,output_result_file=output_file)
    print("Evaluated and saved")



# test(
#     finetuned_model_path="./models/model1/checkpoint-355",
#     test_dataset_path="./ClinAIS_dataset/clinais.dev.json",
#     save_predictions_path="./finetuned_models/model1/predictions.json",
#     save_evaluated_path="./finetuned_models/model1/predictions_evaluated.json")

# test(
#     finetuned_model_path="./models/model2/checkpoint-784",
#     test_dataset_path="./ClinAIS_dataset/clinais.dev.json",
#     save_predictions_path="./finetuned_models/model2/predictions.json",
#     save_evaluated_path="./finetuned_models/model2/predictions_evaluated.json")

# test(
#     finetuned_model_path="./models/model3/checkpoint-1380",
#     test_dataset_path="./ClinAIS_dataset/clinais.dev.json",
#     save_predictions_path="./finetuned_models/model3/predictions.json",
#     save_evaluated_path="./finetuned_models/model3/predictions_evaluated.json")

# test(
#     finetuned_model_path="./models/model4/checkpoint-3910",
#     test_dataset_path="./ClinAIS_dataset/clinais.dev.json",
#     save_predictions_path="./finetuned_models/model4/predictions.json",
#     save_evaluated_path="./finetuned_models/model4/predictions_evaluated.json")


test(
    finetuned_model_path="./models/model5/checkpoint-3910",
    test_dataset_path="./ClinAIS_dataset/clinais.dev.json",
    save_predictions_path="./models/model5/predictions.json",
    save_evaluated_path="./models/model5/predictions_evaluated.json")