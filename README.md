# LLM Cantonese NLU Benchmark

## Overview
This project benchmarks both **multilingual pretrained models** and **large language models (LLMs)** on a diverse set of **Cantonese Natural Language Understanding (NLU)** tasks.

The goal is to systematically evaluate how well different model families handle **low-resource Sinitic languages**, with a focus on **robustness, generalization, and task transferability**.


## Phase 1: Task Coverage

We evaluate models across **7 core Cantonese NLU tasks**:

- **LAJ Grammar Acceptability**  
- **OpenRice Sentiment Analysis**  
- **Dialogue Act Classification**  
- **Emotion Classification**  
- **Topic Classification**  
- **Spam Detection**  
- **Semantic Similarity**

These tasks span multiple dimensions:
- Syntax (grammar)
- Semantics (similarity, topic)
- Pragmatics (dialogue acts)
- Real-world applications (sentiment, spam)

## Models

### Multilingual Pretrained Models (Completed)
- mBERT (`bert-base-multilingual-cased`)
- XLM-R (`xlm-roberta-base`)

**Status:**  
- Data preprocessing pipeline implemented  
- Label mapping and tokenization completed  
- Fine-tuning + evaluation completed on LAJ  
- Metrics: Accuracy, F1-score  

These serve as **strong baseline models** for comparison with LLMs.

### Large Language Models (In Progress)
- GPT-5  
- Gemini  
- Qwen  
- DeepSeek  

These models will be evaluated in a **zero-shot / few-shot prompting setting** to compare against fine-tuned baselines.


### Cantonese-Specific Model
- YueTung-7B  

This model is expected to provide insights into the benefits of **language-specific pretraining**.

## Current Progress

### Completed
- Converted annotated dataset -> `laj.jsonl`
- Implemented preprocessing pipeline:
  - Annotation selection hierarchy  
    (`finalized -> wingspecialist -> loka9 -> york -> Phantom`)
  - Binary label mapping (`acceptable` vs `unacceptable`)
- Fine-tuned:
  - mBERT
  - XLM-R
- Evaluated on **LAJ Grammar Task**

## Next Steps

### 1. Evaluation Harness
- Standardize evaluation across:
  - Fine-tuned models (mBERT, XLM-R)
  - Prompt-based LLMs
- Unified metrics:
  - Accuracy
  - F1-score
  - (Optional) Precision / Recall


### 2. LLM Benchmarking (Starting with LAJ)
- Implement prompting pipeline
- Compare:
  - Zero-shot vs few-shot
  - Instruction formatting effects
- Analyze:
  - Error patterns (e.g., grammar vs semantics confusion)
  - Sensitivity to Cantonese-specific structures


### 3. Expand to All Tasks
- Extend evaluation to remaining 6 tasks
- Ensure consistent data format across datasets


## Research Goals
- Compare **fine-tuned multilingual models vs prompting-based LLMs**
- Understand limitations of LLMs in **low-resource languages**
- Provide insights for:
  - Cantonese NLP development
  - Cross-lingual transfer learning
  - Future benchmark design


## Notes
- Current focus: **LAJ Grammar Task (pilot benchmark)**
- Future work includes:
  - More robust evaluation protocols
  - Cross-task generalization analysis
  - Incorporating additional Cantonese datasets