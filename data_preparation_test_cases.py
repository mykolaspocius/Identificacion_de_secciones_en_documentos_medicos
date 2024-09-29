# This file contains the test code that was used during the development

import unittest
from dataset_model import *
from datasets import Dataset
import sys
sys.path.append("./")
from data_preparation import get_labelled_span_list,create_dataset_object
from misc import create_label_id_dictionaries
from WordListSplitter import WordListSplitter



class TestDataPreparation(unittest.TestCase):
    
    def setUp(self):
        with open('./ClinAIS_dataset/clinais.train.json',encoding='utf-8') as f:
            self.dataset: ClinAISDataset = ClinAISDataset(**json.load(f))  
        self.label2id,self.id2label = create_label_id_dictionaries(ClinicalSections.list())
    
    def tearDown(self):
        # Code to clean up after tests
        pass
    
    def test_get_labbeled_span_list(self):
        for _,entry in self.dataset.annotated_entries.items():
            bas = entry.boundary_annotation.gold
            labelled_spans = get_labelled_span_list(bas)
            self.assertEqual(len(bas),len(labelled_spans))
            cur_label = None
            for i,ba in enumerate(bas):
                if (ba.boundary != None):
                    cur_label=ba.boundary
                self.assertEqual(cur_label,labelled_spans[i][1])

    def test_create_dataset_object(self):
        ds : Dataset = create_dataset_object(self.dataset,self.label2id)
        self.assertEqual(ds.num_rows,len(self.dataset.annotated_entries.keys()))
        for i,row in enumerate(ds):
            self.assertEqual(row['spans'],[ba.span for ba in self.dataset.getEntry(i).boundary_annotation.gold])

    def test_WordListSplitter(self):
        test_list = ['span.' if i % 45 == 0 else 'span' for i in range(789)]
        wls = WordListSplitter(150,3)
        res_list = wls.split(test_list)
        for el in res_list:
            self.assertTrue(len(el)<=150)


if __name__ == '__main__':
    unittest.main()