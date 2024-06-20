from dataset_model import *
from tqdm import tqdm
from pydantic import BaseModel
from typing import List


class PredictionSection(BaseModel):
    entity_group : ClinicalSections
    score : float | None
    word : str
    start : int
    end : int

    class Config:
            use_enum_values = True

class Prediction(BaseModel):
    sections : List[PredictionSection] = []

class PredictionPostProcessor():
    def __init__(self,prediction:Prediction,verbuous=True):
        self.prediction = prediction
        self.min_section_size = 3
        self.punctuation_marks = [',','.',';',':',')',']','}','!','?']
        self.verbuous = verbuous

    def get_section_size(self,section:PredictionSection):
        return len(section.word.strip().split())
    
    def merge_undersize_sections(self):
        erase_idx = []
        prev_section_idx = None
        for idx,section in enumerate(self.prediction.sections):
            if (self.get_section_size(section)<self.min_section_size):
                if (prev_section_idx!=None):
                    if(self.verbuous):print(f"merging section {idx} into {prev_section_idx}")
                    self.prediction.sections[prev_section_idx].word += section.word
                    self.prediction.sections[prev_section_idx].end = section.end
                    erase_idx.append(idx)
                elif(idx<len(self.prediction.sections)-1):
                    if(self.verbuous):print(f"merging section {idx} into {idx+1}")
                    self.prediction.sections[idx+1].word = section.word + self.prediction.sections[idx+1].word
                    self.prediction.sections[idx+1].start = section.start
            else:
                prev_section_idx = idx
        if(len(erase_idx)>0):
            if(self.verbuous):print(f"erasing sections with indecies {erase_idx}")
            for idx in sorted(erase_idx,reverse=True):
                del self.prediction.sections[idx]
        return self.prediction
    
    def reasign_leading_punctuation_marks(self):
        for idx,section in enumerate(self.prediction.sections):
            if (section.word[0] in self.punctuation_marks):
                if (idx>0):
                    if(self.verbuous):print(f"caracter '{section.word[0]}' will be moved from section {idx} to section {idx-1}")
                    self.prediction.sections[idx-1].word += section.word[0]
                    self.prediction.sections[idx-1].end +=1
                    section.word = section.word[1:]
                    section.start += 1

    def merge_contiguous_equivalent_sections(self):
        erase_idx = []
        last_valid_section = None
        for idx,section in enumerate(self.prediction.sections):
            if (last_valid_section == None):
                last_valid_section = section
                continue
            else:
                if (section.entity_group==last_valid_section.entity_group):
                    last_valid_section.word+=section.word
                    last_valid_section.end = section.end
                    erase_idx.append(idx)
                else:
                    last_valid_section = section
        if(len(erase_idx)>0):
            if(self.verbuous):print(f"erasing contiguous sections with same label: {erase_idx}")
            for idx in sorted(erase_idx,reverse=True):
                del self.prediction.sections[idx]
        return self.prediction
    
class PostProcess():
    def __init__(self,entry:Entry):
        self.entry = entry
        self.min_section_size = 3
        self.punctuation_marks = [',','.',';',':',')',']','}','!','?']
        self.verbuous = True

    def getBannotationsForSannotation(self,s_annotation:SectionAnnotation):
        start_idx = None
        res = []
        for i,b_annotation in enumerate(self.entry.boundary_annotation.prediction):
            if (b_annotation.start_offset>=s_annotation.start_offset):
                start_idx = i
                res.append(b_annotation)
                break
        if (start_idx==None):
            raise Exception("Error, Din't find first starting boundary annotation.")
        
        for ba in self.entry.boundary_annotation.prediction[start_idx+1:]:
            if (ba.boundary==None):
                res.append(ba)
        
        return res
    
    def merge_undersize_sections(self):
        erase_idx = []
        prev_section_idx = None
        for idx,section in enumerate(self.entry.section_annotation.prediction):
            corresponding_bas_curent_section = self.getBannotationsForSannotation(section)
            if (len(corresponding_bas_curent_section)<self.min_section_size):
                if (prev_section_idx):
                    if(self.verbuous):print(f"merging section {idx} into {prev_section_idx}")
                    self.entry.section_annotation.prediction[prev_section_idx].segment += section.segment
                    self.entry.section_annotation.prediction[prev_section_idx].end_offset = section.end_offset
                    corresponding_bas_curent_section[0].boundary = None
                    erase_idx.append(idx)
                elif(idx<len(self.entry.section_annotation.prediction)-1):
                    if(self.verbuous):print(f"merging section {idx} into {idx+1}")
                    self.entry.section_annotation.prediction[idx+1].segment = section.segment + self.entry.section_annotation.prediction[idx+1].segment
                    self.entry.section_annotation.prediction[idx+1].start_offset = section.start_offset
                    corresponding_bas_curent_section[0].boundary = self.entry.section_annotation.prediction[idx+1].label
            else:
                prev_section_idx = idx
        if(len(erase_idx)>0):
            if(self.verbuous):print(f"erasing sections with indecies {erase_idx}")
            for idx in sorted(erase_idx,reverse=True):
                del self.entry.section_annotation.prediction[idx]
        return self.entry.section_annotation.prediction
    
    def reasign_leading_punctuation_marks(self):
        for idx,section in enumerate(self.entry.section_annotation.prediction):
            if (section.segment[0] in self.punctuation_marks):
                if (idx>0):
                    if(self.verbuous):print(f"caracter '{section.segment[0]}' will be moved from section {idx} to section {idx-1}")
                    self.entry.section_annotation.prediction[idx-1].segment += section.segment[0]
                    self.entry.section_annotation.prediction[idx-1].end_offset +=1
                    section.segment = section.segment[1:]
                section.start_offset += 1

    def merge_contiguous_equivalent_sections(self):
        erase_idx = []
        last_valid_section = None
        for idx,section in enumerate(self.entry.section_annotation.prediction):
            if (last_valid_section == None):
                last_valid_section = section
                continue
            else:
                if (section.label==last_valid_section.label):
                    last_valid_section.segment+=section.segment
                    last_valid_section.end_offset = section.end_offset
                    corresponding_bas_curent_section = self.getBannotationsForSannotation(section)
                    corresponding_bas_curent_section[0].boundary = None
                    erase_idx.append(idx)
                else:
                    last_valid_section = section
        if(len(erase_idx)>0):
            if(self.verbuous):print(f"erasing contiguous sections with same label: {erase_idx}")
            for idx in sorted(erase_idx,reverse=True):
                del self.entry.section_annotation.prediction[idx]
        return self.entry.section_annotation.prediction
    
    def do_all(self,verbuous=True):
        self.verbuous = verbuous
        self.merge_undersize_sections()
        self.reasign_leading_punctuation_marks()
        self.merge_contiguous_equivalent_sections()

def post_process_dataset(dataset_path,save_postprocessed_path):

    with open(dataset_path) as f:
        dataset: ClinAISDataset = ClinAISDataset(**json.load(f))  

    for _,entry in tqdm(dataset.annotated_entries.items()):
        PostProcess(entry).do_all()

    with open(save_postprocessed_path, 'w', encoding='utf-8') as f:
        f.write(dataset.toJson())
