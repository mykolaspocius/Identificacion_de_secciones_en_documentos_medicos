
def create_label_id_dictionaries(section_types):
    label2id = {}
    id2label = {}
    for id,label in enumerate(section_types):
        label2id[label] = id
        id2label[id] = label

    return label2id,id2label

import json
from pathlib import Path
import matplotlib.pyplot as plt

def graph_trainning_loss(log_path: Path):
    with open(log_path) as f:
        log = json.load(f)

    eval_loss = []
    train_loss = []
    for data in log:
        if('eval_loss' in data.keys()):
            eval_loss.append([data['epoch'],data['eval_loss']])
        elif('loss' in data.keys()):
            train_loss.append([data['epoch'],data['loss']])

    epochs1,e_loss = zip(*eval_loss)
    epochs2,t_loss = zip(*train_loss)
    plt.figure(figsize=(10,5))
    plt.title("Train and Validation loss")
    plt.plot(epochs2,t_loss,label="train loss")
    plt.plot(epochs1,e_loss,label="validation loss")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

