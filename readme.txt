CoalBench: A Bilingual Dataset Suite for Coal-Domain LLM Post-Training
=====================================================================

Language: English (EN) and Chinese (ZH)

Overview
--------
CoalBench is a bilingual synthetic dataset suite designed for large language model (LLM) post-training in the coal mining domain. The dataset is generated through a unified pipeline that constructs three aligned dataset formats:

1. CoalBench-SFT  : Supervised Fine-Tuning dataset
2. CoalBench-DPO  : Direct Preference Optimization dataset
3. CoalBench-RaR  : Rubric-as-Reward dataset

The English and Chinese subsets are generated using the same workflow but from language-specific source materials and prompts.

Dataset Files
-------------

Raw Datasets
~~~~~~~~~~~~
CoalBench-EN.json
    Complete English dataset containing:
    - id
    - class
    - question
    - answer1
    - answer2
    - rubrics

CoalBench-ZH.json
    Complete Chinese dataset containing:
    - id
    - class
    - question
    - answer1
    - answer2
    - rubrics


SFT Datasets
~~~~~~~~~~~~
CoalBench-SFT-EN.json
CoalBench-SFT-ZH.json

Schema:
{
  "id": int,
  "class": str,
  "question": str,
  "answer": str
}

Description:
Question-answer pairs for supervised fine-tuning. The answer field correspond to answer2 generated under weak supervision using class and reference-answer guidance.


DPO Datasets
~~~~~~~~~~~~
CoalBench-DPO-EN.json
CoalBench-DPO-ZH.json

Schema:
{
  "id": int,
  "class": str,
  "question": str,
  "chosen": str,
  "rejected": str
}

Description:
Preference pairs for Direct Preference Optimization (DPO). The chosen response corresponds to answer2, while the rejected response corresponds to answer1.


RaR Datasets
~~~~~~~~~~~~
CoalBench-RaR-EN.json
CoalBench-RaR-ZH.json

Schema:
{
  "id": int,
  "class": str,
  "question": str,
  "rubrics": [...]
}

Description:
Rubric-as-Reward (RaR) dataset containing structured evaluation criteria associated with each question.


Rubric Structure
----------------
Each rubric item contains:

{
  "rubric_id": int,
  "title": str,
  "description": str,
  "weight": int
}

CoalBench follows a four-category rubric design:

Essential
    Critical domain facts whose omission invalidates the response.
    Weight range: 3–5

Important
    Key reasoning steps or completeness requirements.
    Weight range: 2–3

Optional
    Additional details, explanations, or clarity improvements.
    Weight range: 1–2

Pitfall
    Common mistakes, misconceptions, or missing assumptions.
    Weight range: -1 to -2


Dataset Statistics
------------------

English Subset (CoalBench-EN)
    Instances:                742
    Classes:                  26
    Total Rubrics:            8,404
    Avg Rubrics / Instance:   11

Chinese Subset (CoalBench-ZH)
    Instances:                526
    Classes:                  17
    Total Rubrics:            8,240
    Avg Rubrics / Instance:   16


Generation Workflow
-------------------
The dataset is generated using a Qwen3-32B-based pipeline inspired by On-Policy Self-Distillation (OPSD).

Stage 1:
    Generate answer1 using only the question.

Stage 2:
    Generate answer2 using the question together with the class label and reference answer as weak supervision.

Stage 3:
    Generate structured rubrics from the question-answer pair using prompts adapted from OpenAI HealthBench.

