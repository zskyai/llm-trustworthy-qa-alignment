"""LED evidence scaffold interface for long-context Qasper generation.

The scaffold is extractive and source preserving: LED receives selected
paragraphs, while paragraph and sentence offsets remain available for citation
validation. It does not claim a trained LED checkpoint until fine-tuning is
run on the official Qasper training papers.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import torch
from transformers import AutoTokenizer, LEDForConditionalGeneration

@dataclass(frozen=True)
class ScaffoldItem:
    evidence_id: str
    paragraph_id: str
    text: str
    sentence_ids: tuple[int, ...]

class LEDEvidenceScaffold:
    def __init__(self, checkpoint, *, max_input_tokens=16384, device='cpu'):
        if max_input_tokens < 128 or max_input_tokens > 16384: raise ValueError('LED input budget must be 128..16384')
        self.checkpoint=str(checkpoint);self.max_input_tokens=max_input_tokens;self.device=device
        self.tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True)
        self.model=LEDForConditionalGeneration.from_pretrained(checkpoint,local_files_only=True).to(device).eval()
        if self.model.config.max_encoder_position_embeddings < max_input_tokens: raise ValueError('checkpoint cannot cover configured LED input budget')

    def build(self, question: str, selected: Sequence[dict]) -> tuple[str, list[ScaffoldItem]]:
        if not question.strip(): raise ValueError('question must not be empty')
        items=[];parts=['Question: '+question,'Evidence:']
        for i,row in enumerate(selected,1):
            pid=str(row['paragraph_id'] if 'paragraph_id' in row else row['id'])
            text=str(row['text']).strip()
            if not text: continue
            sid=tuple(int(x) for x in row.get('sentence_ids',()))
            evidence_id='E'+str(i);items.append(ScaffoldItem(evidence_id,pid,text,sid));parts.append(f'[{evidence_id}] {text}')
        prompt='\n'.join(parts)
        tokenized=self.tokenizer(prompt,add_special_tokens=True,truncation=False)
        if len(tokenized['input_ids'])>self.max_input_tokens: raise ValueError('selected evidence exceeds LED budget; retrieve fewer paragraphs')
        return prompt,items

    @torch.inference_mode()
    def generate(self, question: str, selected: Sequence[dict], *, max_new_tokens=256) -> dict:
        if max_new_tokens<1 or max_new_tokens>1024: raise ValueError('invalid generation budget')
        prompt,items=self.build(question,selected);encoded=self.tokenizer(prompt,return_tensors='pt',truncation=False).to(self.device)
        global_mask=torch.zeros_like(encoded['input_ids']);global_mask[:,0]=1
        output=self.model.generate(**encoded,global_attention_mask=global_mask,max_new_tokens=max_new_tokens,do_sample=False)
        text=self.tokenizer.decode(output[0],skip_special_tokens=True)
        return {'text':text,'evidence': [item.__dict__ for item in items], 'input_tokens':encoded['input_ids'].shape[1], 'generated_tokens':output.shape[1]}
