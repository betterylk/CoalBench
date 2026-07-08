CoalBench: A Bilingual Coal-Domain Dataset Suite for LLM Post-Training
======================================================================

This folder contains the CoalBench release package: bilingual coal-mining
datasets, prompt templates, post-training/evaluation code, and the main
post-training learning curves exported from SwanLab.

Languages
---------
- EN: English
- ZH: Chinese

Directory Layout
----------------
CoalBench/
  CoalBench/
    Final dataset files for full data, SFT, DPO, and RaR training.

  Code/
    Data construction, post-training, evaluation, plotting, and reward code.

  Prompt/
    Prompt templates used for answer generation, rubric generation, and
    GPT-as-judge evaluation.

  Post-training results/
    SwanLab CSV exports and rendered learning-curve figures.


1. Dataset Files
----------------
All dataset files are JSON arrays. Each record is indexed by "id" and includes
a coal-domain "class" label and "question". The derived training formats keep
the same sample order and identifiers as the corresponding full dataset.

1.1 Full CoalBench Files
~~~~~~~~~~~~~~~~~~~~~~~~
Files:
- CoalBench/CoalBench-EN.json
- CoalBench/CoalBench-ZH.json

Schema:
{
  "id": int,
  "class": str,
  "question": str,
  "answer1": str,
  "answer2": str,
  "rubrics": [rubric]
}

Description:
- answer1 is generated from the question only.
- answer2 is generated from the question plus weak supervision from the class
  label and reference answer.
- rubrics are structured evaluation criteria generated from the question and
  reference-guided answer.

1.2 Supervised Fine-Tuning Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Files:
- CoalBench/CoalBench-SFT-EN.json
- CoalBench/CoalBench-SFT-ZH.json

Schema:
{
  "id": int,
  "class": str,
  "question": str,
  "answer": str
}

Description:
Question-answer pairs for supervised fine-tuning. The "answer" field is derived
from answer2 in the full dataset.

1.3 Direct Preference Optimization Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Files:
- CoalBench/CoalBench-DPO-EN.json
- CoalBench/CoalBench-DPO-ZH.json

Schema:
{
  "id": int,
  "class": str,
  "question": str,
  "chosen": str,
  "rejected": str
}

Description:
Preference pairs for DPO. "chosen" corresponds to answer2 and "rejected"
corresponds to answer1.

1.4 Rubric-as-Reward Files
~~~~~~~~~~~~~~~~~~~~~~~~~~
Files:
- CoalBench/CoalBench-RaR-EN.json
- CoalBench/CoalBench-RaR-ZH.json

Schema:
{
  "id": int,
  "class": str,
  "question": str,
  "rubrics": [rubric]
}

Description:
Rubric-as-Reward (RaR) data for reward-model-free reinforcement learning. Each
question is paired with a set of structured rubrics adapted from the HealthBench
rubric style.


2. Rubric Format
----------------
Each rubric item has the following schema:

{
  "rubric_id": int,
  "title": str,
  "description": str,
  "weight": int
}

Rubric categories are encoded at the beginning of "description":
- Essential: critical facts or conclusions required for a valid answer.
  Typical positive weight: 3 to 5.
- Important: key reasoning steps, technical details, or completeness criteria.
  Typical positive weight: 2 to 3.
- Optional: additional detail, clarity, or explanatory depth.
  Typical positive weight: 1 to 2.
- Pitfall: common misconceptions, unsafe guidance, or missing assumptions.
  Typical negative weight: -1 to -2.


3. Dataset Statistics
---------------------
Statistics computed from the JSON files in CoalBench/:

English subset:
- Full file: CoalBench-EN.json
- Instances: 742
- Classes: 26
- Total rubrics: 8,387
- Average rubrics per instance: 11.30

Chinese subset:
- Full file: CoalBench-ZH.json
- Instances: 526
- Classes: 17
- Total rubrics: 8,240
- Average rubrics per instance: 15.67


4. Prompt Files
---------------
Files in Prompt/ document the main prompt templates used in the pipeline:

- Question only.txt
  Generates answer1 from the question alone.

- Question with reference.txt
  Generates answer2 from the question, class label, and reference answer.

- PromptHealthBenchEn.txt
  English rubric-generation prompt adapted from the HealthBench rubric style.

- PromptHealthBenchCn.txt
  Chinese rubric-generation prompt adapted from the same rubric style.

- GPT-as-judge.txt
  Integer 1-10 answer-quality judging prompt using question, reference answer,
  and model answer as inputs.

Note: some prompt/code files may display mojibake if opened with a mismatched
encoding. Use UTF-8 where possible.


5. Code Files
-------------
Files in Code/ are research scripts rather than a single packaged command-line
tool. Paths and device settings are environment-specific and should be updated
before rerunning.

- runEn.py
  English data construction pipeline using Qwen3-32B. Stages include input
  cleaning/reindexing, question-only answer generation, reference-guided answer
  generation, rubric generation, empty-rubric repair, final filtering/reindexing,
  and class mismatch processing.

- runCn.py
  Chinese data construction pipeline using Qwen3-32B. Stages include source
  conversion, answer generation, rubric generation, rubric repair, and final
  dataset construction.

- runQwen2.5_0.5B.py
  Post-training and evaluation script for Qwen2.5-0.5B-Instruct with ms-swift.
  It includes stages for:
  - SFT LoRA training, inference, BLEU-1/ROUGE-L evaluation, and plotting.
  - DPO LoRA training, inference, GPT-as-judge scoring, and plotting.
  - RaR/GRPO LoRA training with rubric reward, inference, reward comparison,
    and plotting.
  - SwanLab CSV learning-curve plotting.
  - Dataset validation and class/rubric distribution analysis.

- rubric_Reward2.py
  External ms-swift ORM reward function for RaR/GRPO. It embeds model responses
  and rubric text with BAAI/bge-m3, computes cosine similarity, applies a
  floor-scaled continuous reward, adds positive rubric weights, and subtracts
  matching pitfall weights.


6. Post-Training Results
------------------------
Post-training results/ contains SwanLab exports and rendered figures:

CSV files:
- coalbench-sft-lora-2026-6-25_15_29_18.csv
  SFT training and evaluation loss over training steps.

- coalbench-dpo-2026-6-25_15_29_08.csv
  DPO chosen/rejected rewards for training and evaluation, plus reward margins.

- coalbench-grpo-rubric-lora-2026-6-25_15_28_14.csv
  RaR/GRPO rubric reward curves for training and evaluation.

- coalbench-grpo-rubric-lora-2026-6-25_15_29_00.csv
  RaR/GRPO KL curves for training and evaluation.

Figures:
- Post-training results/figures/SwanLabSFT.png
- Post-training results/figures/SwanLabDPO.png
- Post-training results/figures/SwanLabRaR.png

These figures are generated from the CSV files by Stage 4 of
Code/runQwen2.5_0.5B.py.


7. Reproducibility Notes
------------------------
- The scripts assume local model paths such as Qwen3-32B, Qwen2.5-0.5B-Instruct,
  and BAAI/bge-m3. Update these paths for a new environment.
- The scripts use Huawei Ascend device environment variables in their current
  form. Adjust device settings for CUDA, CPU, or other accelerators.
- The post-training script depends on ms-swift, PyTorch, pandas, NumPy,
  matplotlib, and related evaluation utilities.
- SwanLab logging is enabled in the training commands; configure or disable it
  as needed.
- The released dataset files in CoalBench/ are the primary artifacts. The code
  records the construction and evaluation workflow used to produce and analyze
  them.
