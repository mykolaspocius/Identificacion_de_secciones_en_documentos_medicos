from dataset_model import *
from WordListSplitter import EntrySplitter
from post_process import *
from typing import List
from tqdm import tqdm


def split_entry(b_annotations_list:List[List[BoundaryAnnotation]],tokenizer,max_length_preset=None)->List[List[BoundaryAnnotation]]:
    res = []
    rdy = True
    length_limit = tokenizer.model_max_length if max_length_preset==None else max_length_preset
    for b_annotations in b_annotations_list:
        tokenized = tokenizer.encode(" ".join([ba.span for ba in b_annotations]),truncation=False)
        if (len(tokenized)>length_limit):
            max_length = int((length_limit/len(tokenized))*len(b_annotations))
            entry_splitter = EntrySplitter(max_size=max_length,min_size=3)
            res += entry_splitter.split(b_annotations)
            rdy = False
        else:
            res.append(b_annotations)
    return rdy,res

def get_text_splits(entry:Entry,ba_splits:List[List[BoundaryAnnotation]])->List[str]:
    res = []
    for b_annotations in ba_splits:
        res.append(entry.note_text[b_annotations[0].start_offset:b_annotations[-1].end_offset+1])
    return res


def make_prediction(text_splits,model):
    prediction=Prediction()
    offset = 0
    for text_split in text_splits:
        prediction_part  = Prediction()
        prediction_part.sections = [PredictionSection(**section) for section in model(text_split)]
        for section in prediction_part.sections:
            section.start+=offset
            section.end+=offset
            prediction.sections.append(section)
        offset+=len(text_split)
    return prediction

def post_process_prediction(prediction:Prediction,verbuous=True):
    ppp=PredictionPostProcessor(prediction,verbuous)
    ppp.reasign_leading_punctuation_marks()
    ppp.merge_undersize_sections()
    ppp.merge_contiguous_equivalent_sections()


def getBoudaryAnnotationsForRange(bas:List[BoundaryAnnotation],start,end)->List[BoundaryAnnotation]:
    res = []
    idx = 0
    while idx<len(bas) and bas[idx].start_offset<end:
        if (bas[idx].start_offset>=start):
            res.append(bas[idx])
        idx+=1
    return res

def create_b_annotations(entry:Entry,prediction:Prediction):

    b_annotations_predicted = [BoundaryAnnotation(**ba.__dict__) for ba in entry.boundary_annotation.gold]
    for i,section in enumerate(prediction.sections):
        bas = getBoudaryAnnotationsForRange(b_annotations_predicted,section.start,section.end)
        if (len(bas)!=0):
            bas[0].boundary = section.entity_group
            for i in range(1,len(bas)):
                bas[i].boundary = None
        else:
            print(f"Section got no bas asigned. Section {i}, last={i==len(prediction.sections)-1} 'word'={section.word}")

    return b_annotations_predicted


def process_entry(entry:Entry,model):
    rdy=False
    ba_splits = [entry.boundary_annotation.gold]
    while not rdy:
        rdy,ba_splits = split_entry(ba_splits,model.tokenizer)

    text_splits = get_text_splits(entry,ba_splits)
    prediction : Prediction = make_prediction(text_splits,model)
    post_process_prediction(prediction,False)
    
    b_annotations_predicted = create_b_annotations(entry,prediction)

    for section in prediction.sections:
        sa = SectionAnnotation(segment=section.word,label=section.entity_group,start_offset=section.start,end_offset=section.end)
        entry.section_annotation.prediction.append(sa)
    entry.boundary_annotation.prediction = b_annotations_predicted



def create_predictions_file(dataset_path,save_predicted_path,model):
    with open(dataset_path,encoding='utf-8') as f:
        dataset: ClinAISDataset = ClinAISDataset(**json.load(f))  

    for _,entry in tqdm(dataset.annotated_entries.items()):
        process_entry(entry,model) 

    with open(save_predicted_path, 'w', encoding='utf-8') as f:
        f.write(dataset.toJson())


