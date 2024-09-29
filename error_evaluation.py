from dataset_model import *
import json
import numpy as np
import matplotlib.pyplot as plt 

# This code is used to create graph for showing type of errors in model predicionts

def evaluate_error_per_section(evaluated_path,evaluation_set_path):
    with open(evaluated_path,encoding='utf-8') as f:
        evaluated_data = json.load(f)

    with open(evaluation_set_path,encoding='utf-8') as f:
        evaluation_set = ClinAISDataset(**json.load(f))

    val_data_section_counts = {label : 0 for label in ClinicalSections.list()}
    for key,entry in evaluation_set.annotated_entries.items():
        for section in entry.section_annotation.gold:
            val_data_section_counts[section.label]+=1


    scores = evaluated_data["Scores per file"]
    errors_per_section_type = {section_type:0 for section_type in ClinicalSections.list()}
    for note_id,data in scores.items():
        stats = data["Statistics"]
        for section_type in stats['additions']:
            errors_per_section_type[section_type]+=1
        for section_type in stats['deletions']:
            errors_per_section_type[section_type]+=1
        for section_type in [s for substitution in stats['substitutions'] for s in substitution]:
            errors_per_section_type[section_type]+=1

    errors_per_section_normalized = dict(map(lambda kv:(kv[0],round(kv[1]/val_data_section_counts[kv[0]],2)*100),errors_per_section_type.items()))

    sections = list(errors_per_section_normalized.keys())
    errors = list(errors_per_section_normalized.values())
    
    X_axis = np.arange(len(sections)) 
    plt.bar(X_axis - 0.2, errors, 0.4) 

    plt.xticks(X_axis, sections) 
    plt.xlabel("Secciones")
    plt.ylabel("% del total de cada sección")
    plt.title("Errores por tipo de sección")
    plt.legend() 
    plt.show() 

evaluate_error_per_section("models/model4/predictions_evaluated.json","ClinAIS_dataset/clinais.dev.json")
