from predictions import create_predictions_file
from transformers import pipeline
from pathlib import Path
import sys
sys.path.append("./evaluation")
from evaluation.metricas import score_predictions

def test(finetuned_model_path,test_dataset_path,save_predictions_path,save_evaluated_path):
    print("Creating predictions for dataset...")
    modelo = pipeline( "token-classification", model=finetuned_model_path, aggregation_strategy="simple")
    create_predictions_file(test_dataset_path,save_predictions_path,modelo)
    print("Predictions created and saved to file")


    print("Evaluating predictions...")
    prediction_file = Path(save_predictions_path)
    output_file = Path(save_evaluated_path)
    score_predictions(prediction_file=prediction_file,output_result_file=output_file)
    print("Evaluated and saved")



# test(
#     finetuned_model_path="./finetuned_models/model1/checkpoint-355",
#     test_dataset_path="./ClinAIS_dataset/clinais.dev.json",
#     save_predictions_path="./finetuned_models/model1/predictions.json",
#     save_evaluated_path="./finetuned_models/model1/predictions_evaluated.json")

# test(
#     finetuned_model_path="./finetuned_models/model2/checkpoint-784",
#     test_dataset_path="./ClinAIS_dataset/clinais.dev.json",
#     save_predictions_path="./finetuned_models/model2/predictions.json",
#     save_evaluated_path="./finetuned_models/model2/predictions_evaluated.json")

# test(
#     finetuned_model_path="./finetuned_models/model3/checkpoint-1380",
#     test_dataset_path="./ClinAIS_dataset/clinais.dev.json",
#     save_predictions_path="./finetuned_models/model3/predictions.json",
#     save_evaluated_path="./finetuned_models/model3/predictions_evaluated.json")

# test(
#     finetuned_model_path="./finetuned_models/model4/checkpoint-3910",
#     test_dataset_path="./ClinAIS_dataset/clinais.dev.json",
#     save_predictions_path="./finetuned_models/model4/predictions.json",
#     save_evaluated_path="./finetuned_models/model4/predictions_evaluated.json")