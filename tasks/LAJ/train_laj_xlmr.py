import numpy as np
import json
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score


dataset = load_dataset("json", data_files="data/laj.jsonl")

# split
dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)


# label mapping
label_map = {"acceptable": 0, "unacceptable": 1}

def preprocess(example):
    example["label"] = label_map[example["label"]]
    return example

dataset = dataset.map(preprocess)

# load xlmr
MODEL_NAME = "xlm-roberta-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(example):
    return tokenizer(
        example["sentence"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

dataset = dataset.map(tokenize, batched=True)


model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)


# compute metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }


# training setup
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,  # we can reduce it to 1 here
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="no"
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics
)

trainer.train()


# evaluate
results = trainer.evaluate()
print("\nFinal Results:", results)

# save file
os.makedirs("results/xlmr", exist_ok=True)
output_path = "results/xlmr/laj_xlmr_results.json"

experiment = {
    "model": "xlmr",
    "model_name": "xlm-roberta-base",
    "task": "LAJ",
    "epochs": 2,
    "batch_size": 16,
    "results": results
}

with open(output_path, "w") as f:
    json.dump(experiment, f, indent=4)

print(f"\nSaved results to {output_path}")