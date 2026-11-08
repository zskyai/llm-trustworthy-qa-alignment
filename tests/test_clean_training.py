import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import torch
from torch import nn
from src.clean_pubmedqa import load_clean_records, sha256, adapter_state, load_adapter_strict, read_jsonl
from src.pubmedqa_protocol import reconstruct_pqal
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from real_pipeline import inject_lora, save_adapter
from train_clean_pubmedqa import encode_complete
from calibrate_generated_refusal import calibration_inputs

class CleanTrainingTests(unittest.TestCase):
    def test_jsonl_preserves_unicode_line_and_paragraph_separators(self):
        records=[{'document_id':'one','answer':'First\u2028second\u2029third'}, {'document_id':'two','answer':'Last'}]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'records.jsonl'
            with path.open('w',encoding='utf8') as stream:
                for record in records:
                    stream.write(json.dumps(record,ensure_ascii=False)+'\n')
            self.assertEqual(read_jsonl(path),records)

    def test_calibration_rejects_test_ids_duplicates_and_invalid_confidence(self):
        manifest={'train_ids':['train'],'validation_ids':['dev1','dev2'],'test_ids':['test']}
        rows=[{'document_id':'dev1','split':'validation','gold_label':'yes','classification_label':'yes','classification_confidence':.9}, {'document_id':'dev2','split':'validation','gold_label':'maybe','classification_label':'no','classification_confidence':.8}]
        self.assertEqual(calibration_inputs(rows,manifest),([.9,.8],[True,False]))
        mutations=[{'document_id':'test'}, {'split':'test'}, {'classification_confidence':float('nan')}, {'classification_confidence':True}, {'classification_confidence':1.1}]
        for mutation in mutations:
            changed=[dict(rows[0],**mutation),rows[1]]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                calibration_inputs(changed,manifest)
        with self.assertRaises(ValueError):calibration_inputs([rows[0],rows[0]],manifest)
        with self.assertRaises(ValueError):calibration_inputs(rows[:1],manifest)

    def test_full_source_positions_and_test_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source.json'; manifest_path=Path(directory)/'manifest.json'
            data={str(i):{'final_decision':('yes' if i<552 else 'no' if i<890 else 'maybe'),'CONTEXTS':['x'*1500,'second full context'],'QUESTION':'Question?','LONG_ANSWER':'Gold answer.'} for i in range(1000)}
            source.write_text(json.dumps(data),encoding='utf8')
            manifest=reconstruct_pqal(data);manifest['source_sha256']=sha256(source);manifest_path.write_text(json.dumps(manifest),encoding='utf8')
            train,dev,_=load_clean_records(source,manifest_path)
            self.assertEqual((len(train),len(dev)),(450,50))
            self.assertFalse({r['record_id'] for r in train+dev}&set(manifest['test_ids']))
            self.assertEqual(len(train[0]['evidence'][0]['text']),1500)
            self.assertEqual(train[0]['evidence'][0]['position']['char_end'],1500)
            manifest['train_ids'][0]=manifest['train_ids'][1];manifest_path.write_text(json.dumps(manifest),encoding='utf8')
            with self.assertRaises(ValueError):load_clean_records(source,manifest_path)

    def test_only_lora_parameters_train_and_save(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__();self.q_proj=nn.Linear(3,3);self.v_proj=nn.Linear(3,3);self.other=nn.Linear(3,3)
            def forward(self,x):return self.other(self.q_proj(x)+self.v_proj(x))
        model=Model();original={k:v.clone() for k,v in model.state_dict().items()};inject_lora(model)
        trainable={k for k,p in model.named_parameters() if p.requires_grad}
        self.assertEqual(trainable,{'q_proj.a','q_proj.b','v_proj.a','v_proj.b'})
        optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.1)
        model(torch.ones(1,3)).sum().backward();optimizer.step()
        self.assertTrue(torch.equal(original['other.weight'],model.other.weight))
        self.assertTrue(torch.equal(original['q_proj.weight'],model.q_proj.base.weight))
        class Tokenizer:
            def save_pretrained(self,path):pass
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);save_adapter(model,Tokenizer(),path)
            state=torch.load(path/'adapter_model.pt',weights_only=True)
            self.assertEqual(set(state),trainable)
            load_adapter_strict(model,path/'adapter_model.pt')
            state['q_proj.base.weight']=model.q_proj.base.weight;torch.save(state,path/'bad.pt')
            with self.assertRaises(ValueError):load_adapter_strict(model,path/'bad.pt')

    def test_complete_encoding_never_silently_truncates(self):
        class Tokenizer:
            eos_token=' END'
            def apply_chat_template(self,messages,**kw):return messages[0]['content']+' ANSWER'
            def __call__(self,text,**kw):return {'input_ids':list(range(len(text.split())))}
        row={'prompt':'one two three','sft_target':'four five six','record_id':'r'}
        ids,labels=encode_complete(Tokenizer(),row,8)
        self.assertEqual(ids.shape[1],8);self.assertEqual(labels.ne(-100).sum().item(),4)
        with self.assertRaises(ValueError):encode_complete(Tokenizer(),row,7)

if __name__=='__main__':unittest.main()
