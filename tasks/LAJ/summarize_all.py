import json
from sklearn.metrics import accuracy_score, f1_score

models = ["chatgpt", "gemini", "deepseek", "qwen"]

def load_preds(model):
    path = f"../results/{model}/laj_predictions.jsonl"
    preds, golds = [], []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            obj = json.loads(line)
            preds.append(obj["pred"])
            golds.append(obj["gold"])
    return preds, golds

print("MODEL PERFORMANCE SUMMARY (LAJ)")
print("-----------------------------------------")
print("| Model     | Accuracy | Macro-F1 |")
print("-----------------------------------------")

for m in models:
    preds, golds = load_preds(m)
    acc = accuracy_score(golds, preds)
    f1 = f1_score(golds, preds, average="macro")
    print(f"| {m:9} | {acc:.4f}   | {f1:.4f}   |")

print("-----------------------------------------")
