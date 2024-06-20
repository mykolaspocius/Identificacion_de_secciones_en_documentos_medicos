import unittest
from dataset_model import *
from transformers import pipeline
from predictions import split_entry,get_text_splits,make_prediction,create_b_annotations,getBoudaryAnnotationsForRange,post_process_prediction,process_entry




class TestPredictions(unittest.TestCase):
    
    def setUp(self):
        with open('./ClinAIS_dataset/clinais.dev.json',encoding='utf-8') as f:
            self.dataset: ClinAISDataset = ClinAISDataset(**json.load(f))  
        finetuned_model_path="./finetuned_models/model1/checkpoint-355"
        self.model = pipeline( "token-classification", model=finetuned_model_path, aggregation_strategy="simple")
    
    def tearDown(self):
        # Code to clean up after tests
        pass
    
    # def test_split_entry(self):
    #     print("Testing split_entry...")
    #     entries = [79,80,122]
    #     for entry_idx in entries:
    #         print(f"testing entry {entry_idx}")
    #         entry = self.dataset.getEntry(entry_idx)
    #         rdy,res = split_entry([entry.boundary_annotation.gold],self.model.tokenizer)
    #         self.assertTrue(len(entry.boundary_annotation.gold)==sum([len(bas) for bas in res]))

    # def test_get_text_splits(self):
    #     print("Testing get_text_splits...")
    #     entries = range(127)
    #     for entry_idx in entries:
    #         # print(f"testing entry {entry_idx}")
    #         entry = self.dataset.getEntry(entry_idx)
    #         rdy,bas_splits = split_entry([entry.boundary_annotation.gold],self.model.tokenizer)
    #         text_splits = get_text_splits(entry,bas_splits)
    #         self.assertTrue(len(entry.note_text)==sum([len(split) for split in text_splits]))

    # def test_make_prediction(self):
    #     print("Testing merge_prediction_parts...")
    #     entries = [79,80,122]
    #     for entry_idx in entries:
    #         entry = self.dataset.getEntry(entry_idx)
    #         rdy=False
    #         bas_splits=[entry.boundary_annotation.gold]
    #         while not rdy:
    #             rdy,bas_splits = split_entry(bas_splits,self.model.tokenizer)
    #         text_splits = get_text_splits(entry,bas_splits)
    #         prediction = make_prediction(text_splits,self.model)

    #         self.assertTrue(prediction.sections[-1].end==entry.boundary_annotation.gold[-1].end_offset)

    # def test_post_processor(self):
    #     entries = [21]
    #     for entry_idx in entries:
    #         entry = self.dataset.getEntry(entry_idx)
    #         rdy=False
    #         bas_splits=[entry.boundary_annotation.gold]
    #         while not rdy:
    #             rdy,bas_splits = split_entry(bas_splits,self.model.tokenizer)
    #         text_splits = get_text_splits(entry,bas_splits)
    #         prediction = make_prediction(text_splits,self.model)
    #         post_process_prediction(prediction)



    # def test_getBoudaryAnnotationsForRange(self):
    #     entries = [122]
    #     for entry_idx in entries:
    #         entry = self.dataset.getEntry(entry_idx)
    #         bas = entry.boundary_annotation.gold
    #         r1 = getBoudaryAnnotationsForRange(bas,bas[0].start_offset,bas[3].end_offset)
    #         self.assertTrue(len(r1)==4 and r1[0].start_offset==bas[0].start_offset and r1[-1].end_offset==bas[3].end_offset)
    #         r2 = getBoudaryAnnotationsForRange(bas,bas[1].start_offset-2,bas[3].end_offset)
    #         self.assertTrue(len(r2)==3 and r2[0].start_offset==bas[1].start_offset and r2[-1].end_offset==bas[3].end_offset)


    def test_create_b_annotations(self):
        entries = [83]
        for entry_idx in entries:
            entry = self.dataset.getEntry(entry_idx)
            rdy=False
            bas_splits=[entry.boundary_annotation.gold]
            while not rdy:
                rdy,bas_splits = split_entry(bas_splits,self.model.tokenizer)
            text_splits = get_text_splits(entry,bas_splits)
            prediction = make_prediction(text_splits,self.model)
            post_process_prediction(prediction,False)
            bas = create_b_annotations(entry,prediction)

    # def test_process_entry(self):
    #     entries = [79,80,122]
    #     for entry_idx in entries:
    #         entry = self.dataset.getEntry(entry_idx)
    #         process_entry(entry,self.model)



if __name__ == '__main__':
    unittest.main()