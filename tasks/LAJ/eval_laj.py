import json
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

MODEL = "chatgpt"  # change later to gemini / deepseek / qwen
INPUT = f"../../results/{MODEL}/laj_predictions.jsonl"

def main():
    preds, golds = [], []

    with open(INPUT, "r", encoding="utf8") as f:
        for line in f:
            obj = json.loads(line)
            preds.append(obj["pred"])
            golds.append(obj["gold"])

    print("Model:", MODEL)
    print("Accuracy:", accuracy_score(golds, preds))
    print("Macro-F1:", f1_score(golds, preds, average="macro"))
    print("\nConfusion matrix:")
    print(confusion_matrix(golds, preds))

if __name__ == "__main__":
    main()
