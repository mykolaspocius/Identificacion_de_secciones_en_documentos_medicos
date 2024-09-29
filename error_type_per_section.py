import matplotlib.pyplot as plt
import numpy as np
import json
from dataset_model import *

# This code is used to create graphs for hilighting error in model predictions

results = {
    "PRESENT_ILLNESS": [10, 15, 17, 32],
    "DERIVED_FROM_TO": [26, 22, 29, 10],
    "PAST_MEDICAL_HISTORY": [35, 37, 7, 2],
    "FAMILY_HISTORY": [32, 11, 9, 15],
    "EXPLORATION": [21, 29, 5, 5],
    "TREATMENT": [8, 19, 5, 30]
}

category_names = ['Corecto', 'Adiciones','Eliminaciones', 'Substituciones']

with open("./models/model4/predictions_evaluated.json",encoding='utf-8') as f:
    evaluated_data = json.load(f)
    
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
        
        
def plot_error_per_section(results, category_names):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
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
