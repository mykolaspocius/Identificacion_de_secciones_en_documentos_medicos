
from typing import List
from dataset_model import *

# This object is used to split a list of spans into smaller sublists that are less then max_size
# It uses a list of split_caracters which contains a list of caracters that can be used as splitting points
class WordListSplitter():
    def __init__(self,max_size : int,min_size : int,split_caracter_list=['.',',',';',':']):
        self.max_size = max_size
        self.min_size = min_size
        self.split_caracters = split_caracter_list
        
    # This function looks for spans ending with split caracter
    # checks if it is a final span and ads possible split position to positions
    # It returns a position more close to the center of the list of spans
    def get_split_index(self,spans:list[str],caracter='.'):
        positions = []
        for idx,span in enumerate(spans):
            if (len(span)>0 and span[-1] == caracter):
                if (idx<len(spans)-1 and len(spans[idx+1])>0 and spans[idx+1][0]=='\n'):
                    positions.append(idx+1)
                else:
                    positions.append(idx)
        if(len(positions)==0): return -1
        avg_of_positions = sum(positions) / len(positions)
        return min(positions, key=lambda x: abs(x - avg_of_positions))

    # This function splits in a recursive maner the list of spans given
    # It uses split_caracter_index for desiding the caracter to be used for desiding split position
    # If it runs out of caracters, the list is split in half
    def split(self,spans : List[str],split_caracter_index=0):
        if(len(spans)>self.max_size):
            spos = self.get_split_index(spans,self.split_caracters[split_caracter_index])
            if (spos <  self.min_size or spos > len(spans) - self.min_size):
                if (split_caracter_index<len(self.split_caracters)-1):
                    print("changing split caracter")
                    return [*self.split(spans,split_caracter_index+1)]
                else:
                    print("split caracter choices exhousted. Dividing in half...")
                    spos = len(spans) // 2

            l = spans[0:spos+1]
            r = spans[spos+1:]
            return [*self.split(l,split_caracter_index),*self.split(r,split_caracter_index)]
        else:
            return [spans]

# Similar to a class before.
# The only difference are the parameters in and out
# This object uses objects from ClinAISDataset class as parameters
class EntrySplitter():
    def __init__(self,max_size : int,min_size : int,split_caracter_list=['.',',',';',':']):
        self.max_size = max_size
        self.min_size = min_size
        self.split_caracters = split_caracter_list

    def get_split_index(self,b_annotations:list[BoundaryAnnotation],caracter='.'):
        positions = []
        for idx,ba in enumerate(b_annotations):
            if (len(ba.span)>0 and ba.span[-1] == caracter):
                if (idx<len(b_annotations)-1 and len(b_annotations[idx+1].span)>0 and b_annotations[idx+1].span[0]=='\n'):
                    positions.append(idx+1)
                else:
                    positions.append(idx)
        if(len(positions)==0): return -1
        avg_of_positions = sum(positions) / len(positions)
        return min(positions, key=lambda x: abs(x - avg_of_positions))
    
    def split(self,b_annotations:list[BoundaryAnnotation],split_caracter_index=0)->list[list[BoundaryAnnotation]]:
        if(len(b_annotations)>self.max_size):
            spos = self.get_split_index(b_annotations,self.split_caracters[split_caracter_index])
            if (spos <  self.min_size or spos > len(b_annotations) - self.min_size):
                if (split_caracter_index<len(self.split_caracters)-1):
                    print("changing split caracter")
                    return [*self.split(b_annotations,split_caracter_index+1)]
                else:
                    print("split caracter choices exhousted. Dividing in half...")
                    spos = len(b_annotations) // 2

            l = b_annotations[0:spos+1]
            r = b_annotations[spos+1:]
            return [*self.split(l,split_caracter_index),*self.split(r,split_caracter_index)]
        else:
            return [b_annotations]
