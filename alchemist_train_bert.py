import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from pathlib import Path
import numpy as np

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
import evaluate

MODEL_NAME = "dmis-lab/biobert-v1.1"
BIORED_TRAIN = Path("BioRED/Train.BioC.JSON")
OUTPUT_DIR = Path("biobert_ner")

BIORED_ENTITY_TYPES = [
    "GeneOrGeneProduct", "DiseaseOrPhenotypicFeature", "ChemicalEntity",
    "SequenceVariant", "CellLine", "OrganismTaxon"
]

LABELS = ["O"]
for t in BIORED_ENTITY_TYPES:
    LABELS.extend([f"B-{t}", f"I-{t}"])

LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

def load_and_tokenize_biored(path: Path, tokenizer):
    """
    Parses BioC JSON and aligns text annotations to BERT tokens.
    Note: In a full production script, you'd calculate strict character offsets.
    This function uses a word-split approximation for demonstration.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    
    collection = raw if isinstance(raw, list) else raw.get("documents", [])
    
    words_list = []
    labels_list = []
    
    for doc in collection:
        for passage in doc.get("passages", []):
            text = passage.get("text", "")
            annotations = passage.get("annotations", [])
            
            words = text.split()
            labels = ["O"] * len(words)
            
            for ann in annotations:
                ann_text = ann.get("text", "")
                ann_type = ann["infons"].get("type", "")
                if ann_type not in BIORED_ENTITY_TYPES:
                    continue
                
                ann_words = ann_text.split()
                for i in range(len(words) - len(ann_words) + 1):
                    if words[i:i+len(ann_words)] == ann_words:
                        labels[i] = f"B-{ann_type}"
                        for j in range(1, len(ann_words)):
                            labels[i+j] = f"I-{ann_type}"
            
            words_list.append(words)
            labels_list.append(labels)
            
    return Dataset.from_dict({"tokens": words_list, "ner_tags": [[LABEL2ID[l] for l in tags] for tags in labels_list]})

def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, padding=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100) 
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def run_training():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID)
    
    dataset = load_and_tokenize_biored(BIORED_TRAIN, tokenizer)
    dataset = dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)
    
    dataset = dataset.train_test_split(test_size=0.1)
    
    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )
    
    trainer = Trainer(model=model, args=args, train_dataset=dataset["train"], eval_dataset=dataset["test"], data_collator=DataCollatorForTokenClassification(tokenizer))
    trainer.train()

if __name__ == "__main__":
    run_training()