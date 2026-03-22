import json, argparse, os
from tqdm import tqdm

PROMPT = """You are evaluating Cantonese grammatical acceptability.
Answer ONLY with: acceptable or unacceptable.

Sentence: {sentence}
"""
def normalize_pred(text):
    t = text.strip().lower()

    # remove punctuation
    t = t.replace(".", "").replace("!", "").replace("?", "").strip()

    # direct exact match
    if t == "acceptable":
        return "acceptable"
    if t == "unacceptable":
        return "unacceptable"

    # substring matches
    if "unacceptable" in t:
        return "unacceptable"
    if "acceptable" in t:
        return "acceptable"

    # numeric cases
    if t in ["1", "yes", "true"]:
        return "acceptable"
    if t in ["0", "no", "false"]:
        return "unacceptable"

    # fallback
    return "acceptable"



def get_prediction(model_name, sentence):
    prompt = PROMPT.format(sentence=sentence)

    if model_name == "chatgpt":
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip().lower()

    elif model_name == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip().lower()





    elif model_name == "deepseek":
        from openai import OpenAI
        client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1"
        )
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip().lower()

    elif model_name == "qwen":
        from openai import OpenAI
        client = OpenAI(
            api_key=os.getenv("QWEN_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        resp = client.chat.completions.create(
            model="qwen-max",
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip().lower()

    else:
        raise ValueError("Unknown model name")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["chatgpt", "gemini", "deepseek", "qwen"])
    args = parser.parse_args()

    model_name = args.model

    input_path = "../../data/laj.jsonl"
    output_path = f"../../results/{model_name}/laj_predictions.jsonl"

    os.makedirs(f"../../results/{model_name}", exist_ok=True)

    data = [json.loads(line) for line in open(input_path, "r", encoding="utf8")]

    outputs = []
    for obj in tqdm(data):
        raw_pred = get_prediction(model_name, obj["sentence"])
        pred = normalize_pred(raw_pred)

        outputs.append({
            "id": obj["id"],
            "sentence": obj["sentence"],
            "gold": obj["label"],
            "pred": pred
        })

    with open(output_path, "w", encoding="utf8") as f:
        for row in outputs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Saved predictions →", output_path)


if __name__ == "__main__":
    main()
