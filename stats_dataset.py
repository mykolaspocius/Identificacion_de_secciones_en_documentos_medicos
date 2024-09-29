# This file is used to create a plot that shows how different sections are
# distributed in train and dev datasets

import numpy as np
import matplotlib.pyplot as plt 
from dataset_model import *

sections = ClinicalSections.list()
with open('./ClinAIS_dataset/clinais.train.json',encoding='utf-8') as f:
    train_data: ClinAISDataset = ClinAISDataset(**json.load(f))
with open('./ClinAIS_dataset/clinais.dev.json',encoding='utf-8') as f:
    val_data: ClinAISDataset = ClinAISDataset(**json.load(f))

train_data_section_counts = {label : 0 for label in sections}
val_data_section_counts = {label : 0 for label in sections}
for key,entry in train_data.annotated_entries.items():
    for section in entry.section_annotation.gold:
        train_data_section_counts[section.label]+=1
for key,entry in val_data.annotated_entries.items():
    for section in entry.section_annotation.gold:
        val_data_section_counts[section.label]+=1


total_train_data_section_count = sum(train_data_section_counts.values())
total_val_data_section_count = sum(val_data_section_counts.values())

train_section_fractions = dict(map(lambda kv:(kv[0],round(kv[1]/total_train_data_section_count*100,1)),train_data_section_counts.items()))
val_section_fractions = dict(map(lambda kv:(kv[0],round(kv[1]/total_val_data_section_count*100,1)),val_data_section_counts.items()))
 
  


sections = list(train_section_fractions.keys())
train_counts = list(train_section_fractions.values())
val_counts = list(val_section_fractions.values())
  
X_axis = np.arange(len(sections)) 
plt.bar(X_axis - 0.2, train_counts, 0.4, label = 'train set') 
plt.bar(X_axis + 0.2, val_counts, 0.4, label = 'val set')  
plt.xticks(X_axis, sections) 
plt.xlabel("Secciones")
plt.ylabel("%")
plt.title("Distribución de secciones")
plt.legend() 
plt.show() 
