import sys
sys.path.append('.')
from dataset_model import *
import pylev
import statistics
from tqdm import tqdm

def levenshtein_distance(t1:str,t2:str):
    return pylev.levenshtein(t1.split(" "),t2.split(" "))

def evaluate_levinshtein_distance(original_set_path,translated_set_path):
    with open(original_set_path,encoding='utf-8') as f:
        original_data : ClinAISDataset = ClinAISDataset(**json.load(f))
    with open(translated_set_path,encoding='utf-8') as f:
        translated_data : ClinAISDataset = ClinAISDataset(**json.load(f))
    
    distances = []
    for key,entry in tqdm(original_data.annotated_entries.items()):
        if key in translated_data.annotated_entries.keys():
            distances.append(levenshtein_distance(entry.note_text,translated_data.annotated_entries[key].note_text))

    print(round(statistics.mean(distances),2),round(statistics.stdev(distances),2))

def get_dataset_avg_entry_len(dataset_path):
    with open(dataset_path,encoding='utf-8') as f:
        dataset : ClinAISDataset = ClinAISDataset(**json.load(f))

    lengths = []
    for _,entry in tqdm(dataset.annotated_entries.items()):
        lengths.append(len(entry.boundary_annotation.gold))

    print(round(statistics.mean(lengths)))

# evaluate_levinshtein_distance('ClinAIS_dataset/clinais.train.json','ClinAIS_dataset/clinais.train.translated.json')
# get_dataset_avg_entry_len('ClinAIS_dataset/clinais.train.json')

print(124.94/349)