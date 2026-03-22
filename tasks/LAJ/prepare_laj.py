import json

INPUT = "../../data/full_annotated.json"
OUTPUT = "../../data/laj.jsonl"

semantic_types = {"mistranslation", "omission", "addition", "grammar", "untranslated"}


def get_spans_and_source(entry):
    ann = entry.get("annotations", {})

    if "finalized_annotations" in ann:
        return ann["finalized_annotations"].get("annotatedSpans", []), "finalized"
    elif "wingspecialist_annotations" in ann:
        return ann["wingspecialist_annotations"].get("annotatedSpans", []), "wingspecialist"
    elif "loka9_annotations" in ann:
        return ann["loka9_annotations"].get("annotatedSpans", []), "loka9"
    elif "york_annotations" in ann:
        return ann["york_annotations"].get("annotatedSpans", []), "york"
    elif "Phantom65536_annotations" in ann:
        return ann["Phantom65536_annotations"].get("annotatedSpans", []), "Phantom65536"
    
    return [], None


def extract_label(entry):
    spans, source = get_spans_and_source(entry)

    if source is None:
        return "acceptable", None

    major_errors = 0
    semantic_errors = 0
    total_errors = len(spans)

    for span in spans:
        if span.get("error_severity", "").lower() == "major":
            major_errors += 1
        if span.get("error_type", "").lower() in semantic_types:
            semantic_errors += 1

    if major_errors >= 2:
        return "unacceptable", source
    if semantic_errors >= 3:
        return "unacceptable", source
    if total_errors >= 7:
        return "unacceptable", source

    return "acceptable", source


def main():
    with open(INPUT, "r", encoding="utf8") as f:
        data = json.load(f)

    out = []

    for entry in data:
        spans, source = get_spans_and_source(entry)

        # skip if no usable annotations
        if source is None:
            continue

        sentence = entry.get("mt", "").strip()
        label, source = extract_label(entry)

        out.append({
            "id": entry.get("id"),
            "sentence": sentence,
            "label": label,
            "source": source   
        })

    with open(OUTPUT, "w", encoding="utf8") as f:
        for item in out:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(out)} LAJ examples → {OUTPUT}")


if __name__ == "__main__":
    main()