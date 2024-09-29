# This file contains the code used to create the graph
# of percentage of error categories per section type in Model 4 predictions

import matplotlib.pyplot as plt
import numpy as np
import json
from dataset_model import *

<<<<<<< HEAD
# Error categories
=======
# This code is used to create graphs for hilighting error in model predictions

results = {
    "PRESENT_ILLNESS": [10, 15, 17, 32],
    "DERIVED_FROM_TO": [26, 22, 29, 10],
    "PAST_MEDICAL_HISTORY": [35, 37, 7, 2],
    "FAMILY_HISTORY": [32, 11, 9, 15],
    "EXPLORATION": [21, 29, 5, 5],
    "TREATMENT": [8, 19, 5, 30]
}

>>>>>>> 79d9736731c1584e693b258b86ab4005b4bd33ec
category_names = ['Corecto', 'Adiciones','Eliminaciones', 'Substituciones']

with open("./models/model4/predictions_evaluated.json",encoding='utf-8') as f:
    evaluated_data = json.load(f)

# Calculate num errors of each category in each section.    
scores = evaluated_data["Scores per file"]
stats_per_section_type = {section_type:{stats_type:0 for stats_type in category_names} for section_type in ClinicalSections.list()}
for note_id,data in scores.items():
    stats = data["Statistics"]
    for section_type in stats['matches']:
        stats_per_section_type[section_type]['Corecto']+=1
    for section_type in stats['additions']:
        stats_per_section_type[section_type]['Adiciones']+=1
    for section_type in stats['deletions']:
        stats_per_section_type[section_type]['Eliminaciones']+=1
    for section_type in [s for substitution in stats['substitutions'] for s in substitution]:
        stats_per_section_type[section_type]['Substituciones']+=1
        
# print(stats_per_section_type)
results = {section_type : [] for section_type in ClinicalSections.list()}
for section_type,stats in stats_per_section_type.items():
    total = sum([num for stats_type,num in stats.items()])
    results[section_type] = [round(num/total,2)*100 for stats_type,num in stats.items()]
    
# print(results)
 
# Function used to plot the results       
def plot_error_per_section(results, category_names):

    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.colormaps['RdYlGn'](
        np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        ax.bar_label(rects, label_type='center', color=text_color)
    ax.legend(ncols=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    return fig, ax


plot_error_per_section(results, category_names)
plt.show()
<<<<<<< HEAD

# with open("./models/model4/predictions.json",encoding='utf-8') as f:
#     predictions : ClinAISDataset = ClinAISDataset(**json.load(f)) 

# "S0034-98872009000700011-1"  
    
# prediction = predictions.annotated_entries["S0034-98872012001000013-1"]
# print(prediction.section_annotation.gold)
# print()
# print(prediction.section_annotation.prediction)
=======
>>>>>>> 79d9736731c1584e693b258b86ab4005b4bd33ec
