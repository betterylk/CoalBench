import os
import subprocess
import json
import re
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math
import numpy as np
import pandas as pd
import argparse
        
import torch
import torch.nn.functional as F
from swift import get_model_processor

os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"
env = os.environ.copy()
env["NPROC_PER_NODE"] = "7"

# os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "1"
# env = os.environ.copy()
# env["NPROC_PER_NODE"] = "1"

Question_DATA = "/root/Lingkai/CoalBench/DataEn/CoalBench-Question-EN.json"
MODEL_DIR = "/root/Lingkai/LLMBias/Models/Qwen2.5-0.5B-Instruct"

STAGE = 1 # SFT (Base and SFT infer/eval; bleu and rouge measures), Plot
STAGE = 2 # DPO
STAGE = 3 # Rubric
STAGE = 4 # Plot learning curves using data from Swanlab CSV
STAGE = 5 # Print an example from validation set
STAGE = 6 # Plot for class distribution
STAGE = 7 # Technique validation, part 1.

Base_folder = "Qwen2.5-0.5B/"
os.makedirs(Base_folder, exist_ok=True)
for STAGE in [7]:
    if STAGE == 1:
        STAGE1_Steps = [5]
        TRAIN_DATA = "/root/Lingkai/CoalBench/DataEn/CoalBench-SFT-EN.json"
        OUTPUT_DIR = f"Models/{Base_folder}/CoalBench-SFT-0.5B"
        SFT_ADAPTER = f"{OUTPUT_DIR}/v11-20260616-103320/checkpoint-400"

        BASE_PRED = "Models/Qwen2.5-0.5B/CoalBench-SFT-0.5B/base_predictions.jsonl"
        SFT_PRED = "Models/Qwen2.5-0.5B/CoalBench-SFT-0.5B/sft_predictions.jsonl"
        val_datasets = "Models/Qwen2.5-0.5B/CoalBench-SFT-0.5B/v11-20260616-103320/val_dataset.jsonl"
        OUTPUT = "Models/Qwen2.5-0.5B/CoalBench-SFT-0.5B/evaluation_results.json"

        # STEP = 1 # SFT training
        # STEP = 2 # infer for base
        # STEP = 3 # infer for SFT model
        # STEP = 4 # bleu and rouge
        # STEP = 5 # plot
        for STEP in STAGE1_Steps:
            if STEP == 1:
                # --------------------------------------------------
                # Step 1. SFT Training
                # --------------------------------------------------
                cmd = [
                    "swift", "sft",
                    "--tuner_type", "lora",
                    "--model", MODEL_DIR,
                    "--optim", "adamw_torch",

                    "--dataset", TRAIN_DATA,
                    "--split_dataset_ratio", "0.15",

                    "--template", "qwen2_5",
                    "--model_type", "qwen2",
                    "--columns", '{"query": "question", "response": "answer"}',

                    "--learning_rate", "5e-5",
                    "--weight_decay", "0.01",
                    "--warmup_ratio", "0.05",

                    "--per_device_train_batch_size", "4",
                    "--per_device_eval_batch_size", "4",
                    "--gradient_accumulation_steps", "8",

                    "--num_train_epochs", "10",
                    "--max_length", "2048",

                    "--lora_rank", "8",
                    "--lora_alpha", "16",
                    "--lora_dropout", "0.1",

                    "--do_eval", "true",
                    "--eval_strategy", "steps",
                    "--eval_steps", "10",

                    "--save_strategy", "steps",
                    "--save_steps", "10",
                    "--save_total_limit", "3",

                    "--metric_for_best_model", "eval_loss",
                    "--load_best_model_at_end", "true",
                    "--greater_is_better", "false",

                    "--output_dir", OUTPUT_DIR,
                    "--overwrite_output_dir", "true",

                    "--report_to", "swanlab",
                    "--swanlab_mode", "cloud",
                    "--swanlab_project", "ms-swift",
                    "--swanlab_exp_name", "coalbench-sft-lora",

                    "--dataloader_num_workers", "0",
                    "--save_only_model", "true",
                ]

                subprocess.run(cmd, env=env, check=True)

            if STEP == 2:
                # --------------------------------------------------
                # Step 2. Base Model Evaluation
                # --------------------------------------------------

                subprocess.run([
                    "swift", "infer",
                    "--model", MODEL_DIR,
                    "--dataset", TRAIN_DATA,
                    "--split_dataset_ratio", "0.15",
                    "--template", "qwen2_5",
                    "--model_type", "qwen2",

                    "--max_new_tokens", "512",
                    "--temperature", "0",

                    "--result_path",
                    f"{OUTPUT_DIR}/base_predictions.jsonl"
                ], env=env, check=True)

            if STEP == 3:
                # --------------------------------------------------
                # Step 3. SFT Model Evaluation
                # --------------------------------------------------
                subprocess.run([
                    "swift", "infer",
                    "--model", MODEL_DIR,
                    "--adapters", SFT_ADAPTER,

                    "--dataset", TRAIN_DATA,
                    "--split_dataset_ratio", "0.15",

                    "--template", "qwen2_5",
                    "--model_type", "qwen2",

                    "--max_new_tokens", "512",
                    "--temperature", "0",

                    "--result_path",
                    f"{OUTPUT_DIR}/sft_predictions.jsonl"
                ], env=env, check=True)

            if STEP == 4:
                # --------------------------------------------------
                # Step 4. Compute Metrics
                # --------------------------------------------------
                def load_jsonl(path):
                    rows = []
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                rows.append(json.loads(line))
                    return rows

                def load_eval_data(path):
                    rows = []

                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                rows.append(json.loads(line))

                    return rows

                def normalize(text):
                    text = str(text).lower()
                    text = re.sub(r"\s+", " ", text)
                    return text.strip()

                def tokenize(text):
                    text = normalize(text)
                    return re.findall(r"\b\w+\b", text)

                def get_response(row):
                    return row.get("response", "")

                def lcs_length(pred_tokens, ref_tokens):
                    n = len(ref_tokens)
                    dp = [0] * (n + 1)

                    for x in pred_tokens:
                        prev = 0
                        for j, y in enumerate(ref_tokens, start=1):
                            tmp = dp[j]
                            if x == y:
                                dp[j] = prev + 1
                            else:
                                dp[j] = max(dp[j], dp[j - 1])
                            prev = tmp

                    return dp[n]

                def rouge_l(pred, ref):
                    pred_tokens = tokenize(pred)
                    ref_tokens = tokenize(ref)

                    if not pred_tokens or not ref_tokens:
                        return 0.0

                    lcs = lcs_length(pred_tokens, ref_tokens)

                    precision = lcs / len(pred_tokens)
                    recall = lcs / len(ref_tokens)

                    if precision + recall == 0:
                        return 0.0
                    return 2 * precision * recall / (precision + recall)

                def bleu_1(pred, ref):
                    pred_tokens = tokenize(pred)
                    ref_tokens = tokenize(ref)

                    if not pred_tokens or not ref_tokens:
                        return 0.0

                    pred_counts = Counter(pred_tokens)
                    ref_counts = Counter(ref_tokens)

                    overlap = sum(
                        min(pred_counts[token], ref_counts[token])
                        for token in pred_counts
                    )

                    precision = overlap / len(pred_tokens)

                    pred_len = len(pred_tokens)
                    ref_len = len(ref_tokens)

                    brevity_penalty = 1.0 if pred_len > ref_len else math.exp(1 - ref_len / pred_len)
                    return brevity_penalty * precision

                def get_ref_answer(ref_row):
                    # Format 1: {"answer": "..."}
                    if "answer" in ref_row:
                        return ref_row.get("answer", "")

                    # Format 2: {"messages": [{"role": "user"}, {"role": "assistant"}]}
                    for msg in ref_row.get("messages", []):
                        if msg.get("role") == "assistant":
                            return msg.get("content", "")

                    return ""


                def get_ref_question(ref_row):
                    # Format 1: {"question": "..."}
                    if "question" in ref_row:
                        return ref_row.get("question", "")

                    # Format 2: {"messages": [{"role": "user"}, {"role": "assistant"}]}
                    for msg in ref_row.get("messages", []):
                        if msg.get("role") == "user":
                            return msg.get("content", "")

                    return ""


                def evaluate(pred_rows, ref_rows):
                    rouge_scores = []
                    bleu_scores = []
                    details = []

                    n = min(len(pred_rows), len(ref_rows))

                    for i in range(n):
                        pred_row = pred_rows[i]
                        ref_row = ref_rows[i]

                        pred = get_response(pred_row)
                        ref = get_ref_answer(ref_row)

                        question = get_ref_question(ref_row)
                        sample_id = ref_row.get("id", i)

                        r = rouge_l(pred, ref)
                        b = bleu_1(pred, ref)

                        rouge_scores.append(r)
                        bleu_scores.append(b)

                        details.append({
                            "index": i,
                            "id": sample_id,
                            "question": question,
                            "prediction": pred,
                            "reference": ref,
                            "rouge_l": r,
                            "bleu_1": b,
                        })

                    return {
                        "num_samples": n,
                        "rouge_l": sum(rouge_scores) / n if n > 0 else 0.0,
                        "bleu_1": sum(bleu_scores) / n if n > 0 else 0.0,
                        "details": details,
                    }
                def main():
                    base_rows = load_jsonl(BASE_PRED)
                    sft_rows = load_jsonl(SFT_PRED)
                    ref_rows = load_eval_data(val_datasets)

                    n = min(len(base_rows), len(sft_rows), len(ref_rows))
                    base_rows = base_rows[:n]
                    sft_rows = sft_rows[:n]
                    ref_rows = ref_rows[:n]

                    base_result = evaluate(base_rows, ref_rows)
                    sft_result = evaluate(sft_rows, ref_rows)

                    result = {
                        "base": {
                            "num_samples": base_result["num_samples"],
                            "rouge_l": base_result["rouge_l"],
                            "bleu_1": base_result["bleu_1"],
                        },
                        "sft": {
                            "num_samples": sft_result["num_samples"],
                            "rouge_l": sft_result["rouge_l"],
                            "bleu_1": sft_result["bleu_1"],
                        },
                        "improvement": {
                            "rouge_l_abs": sft_result["rouge_l"] - base_result["rouge_l"],
                            "bleu_1_abs": sft_result["bleu_1"] - base_result["bleu_1"],
                            "rouge_l_rel_percent": (
                                (sft_result["rouge_l"] - base_result["rouge_l"])
                                / base_result["rouge_l"] * 100
                                if base_result["rouge_l"] > 0 else None
                            ),
                            "bleu_1_rel_percent": (
                                (sft_result["bleu_1"] - base_result["bleu_1"])
                                / base_result["bleu_1"] * 100
                                if base_result["bleu_1"] > 0 else None
                            ),
                        },
                        "details": {
                            "base": base_result["details"],
                            "sft": sft_result["details"],
                        },
                    }

                    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
                    with open(OUTPUT, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)

                    print("\nBase Model:")
                    print(json.dumps(result["base"], indent=2))

                    print("\nSFT Model:")
                    print(json.dumps(result["sft"], indent=2))

                    print("\nImprovement:")
                    print(json.dumps(result["improvement"], indent=2))

                    print(f"\nSaved results to: {OUTPUT}")

                if __name__ == "__main__":
                    main()
            
            if STEP == 5:
                # =========================
                # File paths
                # =========================
                EVAL_JSON = "Models/Qwen2.5-0.5B/CoalBench-SFT-0.5B/evaluation_results.json"
                OUT_DIR = Path("Models/Qwen2.5-0.5B/CoalBench-SFT-0.5B/figures")
                OUT_DIR.mkdir(parents=True, exist_ok=True)

                # =========================
                # Global style
                # =========================
                plt.rcParams.update({
                    "font.family": "Arial",
                    "font.size": 9,
                    "axes.labelsize": 9,
                    "axes.titlesize": 10,
                    "legend.fontsize": 8,
                    "xtick.labelsize": 8,
                    "ytick.labelsize": 8,
                    "figure.dpi": 300,
                    "savefig.dpi": 600,
                    "axes.linewidth": 0.8,
                    "lines.linewidth": 1.5,
                })


                # =========================
                # Helpers
                # =========================
                def load_results(path):
                    with open(path, "r", encoding="utf-8") as f:
                        return json.load(f)


                def extract_scores(results, metric):
                    base = results["details"]["base"]
                    sft = results["details"]["sft"]

                    n = min(len(base), len(sft))
                    base_scores = np.array([base[i][metric] for i in range(n)], dtype=float)
                    sft_scores = np.array([sft[i][metric] for i in range(n)], dtype=float)

                    return base_scores, sft_scores


                def savefig(fig, name):
                    fig.tight_layout()
                    fig.savefig(OUT_DIR / f"{name}.png", bbox_inches="tight")
                    plt.close(fig)


                def format_axis(ax):
                    ax.grid(True, linewidth=0.4, alpha=0.4)

                    for spine in ["top", "right"]:
                        ax.spines[spine].set_visible(False)


                def plot_metric_scatter(ax, base_scores, sft_scores, metric_name):
                    improved = sft_scores > base_scores
                    degraded = sft_scores < base_scores
                    unchanged = ~(improved | degraded)

                    max_v = max(base_scores.max(), sft_scores.max()) * 1.08
                    max_v = max(max_v, 0.05)

                    ax.scatter(
                        base_scores[degraded],
                        sft_scores[degraded],
                        s=22,
                        alpha=0.75,
                        label="Base better",
                    )

                    ax.scatter(
                        base_scores[improved],
                        sft_scores[improved],
                        s=22,
                        alpha=0.75,
                        label="SFT better",
                    )

                    if unchanged.any():
                        ax.scatter(
                            base_scores[unchanged],
                            sft_scores[unchanged],
                            s=22,
                            alpha=0.75,
                            label="Unchanged",
                        )

                    ax.plot(
                        [0, max_v],
                        [0, max_v],
                        linestyle="--",
                        linewidth=1.0,
                        label="Equal performance",
                    )

                    improved_ratio = improved.sum() / len(base_scores) * 100

                    ax.text(
                        0.04,
                        0.96,
                        f"Samples: {len(base_scores)}\nImproved: {improved.sum()} ({improved_ratio:.1f}%)",
                        transform=ax.transAxes,
                        fontsize=8,
                        va="top",
                        bbox=dict(
                            boxstyle="round,pad=0.30",
                            facecolor="white",
                            edgecolor="0.7",
                            alpha=0.9,
                            linewidth=0.6,
                        ),
                    )

                    ax.set_xlim(0, max_v)
                    ax.set_ylim(0, max_v)
                    ax.set_title(metric_name)
                    ax.set_xlabel(f"Base model")
                    ax.set_ylabel(f"SFT model")
                    ax.legend(frameon=False, loc="lower right")
                    format_axis(ax)


                # =========================
                # Main figure
                # =========================
                def main():
                    results = load_results(EVAL_JSON)

                    metrics = [
                        ("bleu_1", "BLEU-1"),
                        ("rouge_l", "ROUGE-L"),
                    ]

                    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))

                    for ax, (metric, metric_name) in zip(axes, metrics):
                        base_scores, sft_scores = extract_scores(results, metric)
                        plot_metric_scatter(ax, base_scores, sft_scores, metric_name)

                        deltas = sft_scores - base_scores
                        print(f"\n{metric_name}")
                        print(f"Samples: {len(deltas)}")
                        print(f"Improved: {(deltas > 0).sum()}")
                        print(f"Unchanged: {(deltas == 0).sum()}")
                        print(f"Degraded: {(deltas < 0).sum()}")
                        print(f"Improved ratio: {(deltas > 0).mean():.4f}")
                        print(f"Mean delta: {deltas.mean():.4f}")

                    savefig(fig, "ScatterSFT")
                    print(f"\nFigure saved to: {OUT_DIR.resolve()}")


                if __name__ == "__main__":
                    main()

    if STAGE == 2:
        STAGE2_Steps = [5]
        # STEP = 1 # DPO training
        # STEP = 2 # infer for base
        # STEP = 3 # infer for DPO model
        # STEP = 4 # GPT judge
        # STEP = 5 # plot
        for STEP in STAGE2_Steps:
            SRC_DATA = "/root/Lingkai/CoalBench/DataEn/CoalBench-DPO-EN.json"
            TRAIN_DATA = "/root/Lingkai/CoalBench/DataEn/CoalBench-DPO-EN-swift.json"
            Path_GPTJudge = "/root/Lingkai/CoalBench/DataEn/GPTjudged.json"

            OUTPUT_DIR = f"Models/{Base_folder}/CoalBench-DPO-0.5B"
            DPO_ADAPTER = f"{OUTPUT_DIR}/v0-20260617-174058/checkpoint-400"  

            with open(SRC_DATA, "r", encoding="utf-8") as f:
                data = json.load(f)

            swift_data = []
            for x in data:
                swift_data.append({
                    "messages": [
                        {"role": "user", "content": x["question"]},
                        {"role": "assistant", "content": x["chosen"]}
                    ],
                    "rejected_response": x["rejected"]
                })

            with open(TRAIN_DATA, "w", encoding="utf-8") as f:
                json.dump(swift_data, f, ensure_ascii=False, indent=2)

            assert os.path.exists(TRAIN_DATA)
            print("converted:", TRAIN_DATA, len(swift_data))
            print(swift_data[0])

            if STEP == 1:        
                cmd = [
                    "swift", "rlhf",
                    "--rlhf_type", "dpo",
                    "--tuner_type", "lora",
                    "--model", MODEL_DIR,
                    "--dataset", TRAIN_DATA,
                    "--split_dataset_ratio", "0.15",   # 15% validation
                    "--template", "qwen2_5",
                    "--model_type", "qwen2",
                    "--beta", "0.08",
                    "--num_train_epochs", "10",  
                    "--learning_rate", "2e-6",
                    "--optim", "adamw_torch",
                    "--per_device_train_batch_size", "2",
                    "--per_device_eval_batch_size", "2",
                    "--gradient_accumulation_steps", "8",
                    "--max_length", "1024",
                    "--lora_rank", "8",
                    "--lora_alpha", "16",
                    "--lora_dropout", "0.05",
                    "--do_eval", "true",
                    "--eval_strategy", "steps",
                    "--eval_steps", "10",
                    "--save_strategy", "steps",
                    "--save_steps", "100",
                    "--save_total_limit", "3",
                    "--output_dir", OUTPUT_DIR,
                    "--overwrite_output_dir", "true",
                    "--report_to", "swanlab",
                    "--swanlab_mode", "cloud",
                    "--swanlab_project", "ms-swift",
                    "--swanlab_exp_name", "coalbench-dpo",
                    "--logging_steps", "10",
                    "--save_only_model", "true",
                ]
                subprocess.run(cmd, env=env, check=True)

            if STEP == 2:
                # --------------------------------------------------
                # Step 2. Base Model Evaluation
                # --------------------------------------------------

                subprocess.run([
                    "swift", "infer",
                    "--model", MODEL_DIR,
                    "--dataset", Question_DATA,
                    "--split_dataset_ratio", "0.15",
                    "--template", "qwen2_5",
                    "--model_type", "qwen2",

                    "--max_new_tokens", "512",
                    "--temperature", "0",

                    "--result_path",
                    f"{OUTPUT_DIR}/base_predictions.jsonl"
                ], env=env, check=True)

            if STEP == 3:
                # --------------------------------------------------
                # Step 3. DPO Model Evaluation
                # --------------------------------------------------
                subprocess.run([
                    "swift", "infer",
                    "--model", MODEL_DIR,
                    "--adapters", DPO_ADAPTER,

                    "--dataset", Question_DATA,
                    "--split_dataset_ratio", "0.15",

                    "--template", "qwen2_5",
                    "--model_type", "qwen2",

                    "--max_new_tokens", "512",
                    "--temperature", "0",

                    "--result_path",
                    f"{OUTPUT_DIR}/dpo_predictions.jsonl"
                ], env=env, check=True)

            if STEP == 4:
                pass
                # using gpt 5.5 compare results from the base and the DPO result
            
            if STEP == 5:
                # =========================
                # File paths
                # =========================
                OUT_DIR = "Models/Qwen2.5-0.5B/CoalBench-DPO-0.5B/figures"
                BASE_SCORE_KEY = "base_score"
                DPO_SCORE_KEY = "DPO_score"


                # =========================
                # Global style
                # =========================
                plt.rcParams.update({
                    "font.family": "Arial",
                    "font.size": 9,
                    "axes.labelsize": 9,
                    "axes.titlesize": 10,
                    "legend.fontsize": 8,
                    "xtick.labelsize": 8,
                    "ytick.labelsize": 8,
                    "figure.dpi": 300,
                    "savefig.dpi": 600,
                    "axes.linewidth": 0.8,
                    "lines.linewidth": 1.5,
                })
                # =========================
                # Helpers
                # =========================
                def load_json(path):
                    with open(path, "r", encoding="utf-8") as f:
                        return json.load(f)


                def savefig(fig, out_dir, name):
                    png_path = out_dir / f"{name}.png"

                    fig.tight_layout()
                    fig.savefig(png_path, dpi=600, bbox_inches="tight")
                    plt.close(fig)

                    print(f"Saved: {png_path}")


                def format_axis(ax):
                    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.35)
                    for spine in ["top", "right"]:
                        ax.spines[spine].set_visible(False)


                def plot_score_scatter(data, out_dir):
                    out_dir = Path(out_dir)
                    out_dir.mkdir(parents=True, exist_ok=True)

                    base_scores = np.array([float(x[BASE_SCORE_KEY]) for x in data], dtype=float)
                    dpo_scores = np.array([float(x[DPO_SCORE_KEY]) for x in data], dtype=float)

                    n = len(base_scores)

                    improved = dpo_scores > base_scores
                    degraded = dpo_scores < base_scores
                    tied = dpo_scores == base_scores

                    base_avg = base_scores.mean()
                    dpo_avg = dpo_scores.mean()

                    pair_counts = Counter(zip(base_scores, dpo_scores))

                    fig, ax = plt.subplots(figsize=(5.4, 3.2))

                    colors = {
                        "DPO better": "tab:green",
                        "Tie": "tab:orange",
                        "Base better": "tab:blue",
                    }

                    labels_used = set()

                    for (base_score, dpo_score), count in sorted(pair_counts.items()):
                        if dpo_score > base_score:
                            label = "DPO better"
                        elif dpo_score < base_score:
                            label = "Base better"
                        else:
                            label = "Tie"

                        size = 38 + 42 * np.sqrt(count)

                        ax.scatter(
                            base_score,
                            dpo_score,
                            s=size,
                            color=colors[label],
                            alpha=0.78,
                            edgecolors="black",
                            linewidths=0.45,
                            label=label if label not in labels_used else None,
                            zorder=3,
                        )

                        ax.text(
                            base_score,
                            dpo_score,
                            str(count),
                            ha="center",
                            va="center",
                            fontsize=6.5,
                            fontweight="bold",
                            color="black",
                            zorder=4,
                        )

                        labels_used.add(label)

                    min_v = min(base_scores.min(), dpo_scores.min()) - 0.5
                    max_v = max(base_scores.max(), dpo_scores.max()) + 0.5

                    ax.plot(
                        [min_v, max_v],
                        [min_v, max_v],
                        linestyle="--",
                        linewidth=1.1,
                        color="tab:blue",
                        label="Equal score",
                        zorder=2,
                    )

                    improved_ratio = improved.sum() / n * 100
                    tied_ratio = tied.sum() / n * 100
                    degraded_ratio = degraded.sum() / n * 100

                    summary_text = (
                        f"Samples: {n}\n"
                        f"DPO better: {improved.sum()} ({improved_ratio:.1f}%)\n"
                        f"Tie: {tied.sum()} ({tied_ratio:.1f}%)\n"
                        f"Base better: {degraded.sum()} ({degraded_ratio:.1f}%)\n"
                        f"DPO avg. score: {dpo_avg:.2f}\n"
                        f"Base avg. score: {base_avg:.2f}"
                    )

                    ax.text(
                        0.04,
                        0.96,
                        summary_text,
                        transform=ax.transAxes,
                        fontsize=7.2,
                        va="top",
                        linespacing=1.1,
                        bbox=dict(
                            boxstyle="round,pad=0.25",
                            facecolor="white",
                            edgecolor="0.7",
                            alpha=0.90,
                            linewidth=0.6,
                        ),
                        zorder=5,
                    )

                    ax.set_xlim(min_v, max_v)
                    ax.set_ylim(min_v, max_v)
                    ax.set_xlabel("Base model score")
                    ax.set_ylabel("DPO model score")

                    handles, labels = ax.get_legend_handles_labels()
                    order = ["Equal score", "DPO better", "Tie", "Base better"]
                    ordered = [(h, l) for l in order for h, lab in zip(handles, labels) if lab == l]
                    if ordered:
                        handles, labels = zip(*ordered)
                        ax.legend(handles, labels, frameon=False, loc="lower right")

                    format_axis(ax)
                    savefig(fig, out_dir, "ScatterDPO")

                    print("\nSummary")
                    print(f"Samples: {n}")
                    print(f"DPO better: {improved.sum()} ({improved_ratio:.1f}%)")
                    print(f"Tie: {tied.sum()} ({tied_ratio:.1f}%)")
                    print(f"Base better: {degraded.sum()} ({degraded_ratio:.1f}%)")
                    print(f"DPO average score: {dpo_avg:.2f}")
                    print(f"Base average score: {base_avg:.2f}")


                if __name__ == "__main__":
                    data = load_json(Path_GPTJudge)
                    plot_score_scatter(data, OUT_DIR)

    if STAGE == 3:
        STAGE3_Steps = [5]
        # STEP = 1 # Rubric split to Bank and Question
        # STEP = 2 # Train as RaR
        # STEP = 3 # infer for base
        # STEP = 4 # infer for RaR model
        # STEP = 5 # Rubric-level model comparison/plot

        INPUT_PATH = "/root/Lingkai/CoalBench/DataEn/CoalBench-RaR-EN.json"
        QUESTION_PATH = "/root/Lingkai/CoalBench/DataEn/RubricQuestion.json"
        RUBRIC_PATH = "/root/Lingkai/CoalBench/DataEn/RubricBank.json"

        OUTPUT_DIR = f"Models/{Base_folder}/CoalBench-RaR-0.5B"
        RaR_ADAPTER = f"{OUTPUT_DIR}/v3-20260625-083624/checkpoint-180"  

        for STEP in STAGE3_Steps:
            if STEP == 1:
                def split_rubric_data(input_path, question_path, rubric_path):
                    input_path = Path(input_path)
                    question_path = Path(question_path)
                    rubric_path = Path(rubric_path)

                    with input_path.open("r", encoding="utf-8") as f:
                        records = json.load(f)

                    questions = []
                    rubric_bank = {}

                    for record in records:
                        sample_id = int(record["id"])
                        question = record["question"].strip()
                        rubrics = record["rubrics"]

                        questions.append(
                            {
                                "id": sample_id,
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": f"ID:{sample_id}\n{question}",
                                    }
                                ],
                            }
                        )

                        rubric_bank[str(sample_id)] = [
                            {
                                "description": rubric["description"],
                                "weight": float(rubric["weight"]),
                            }
                            for rubric in rubrics
                        ]

                    question_path.parent.mkdir(parents=True, exist_ok=True)
                    rubric_path.parent.mkdir(parents=True, exist_ok=True)

                    with question_path.open("w", encoding="utf-8") as f:
                        json.dump(questions, f, ensure_ascii=False, indent=2)

                    with rubric_path.open("w", encoding="utf-8") as f:
                        json.dump(rubric_bank, f, ensure_ascii=False, indent=2)

                    print(f"Wrote {len(questions)} questions to {question_path}")
                    print(f"Wrote {len(rubric_bank)} rubric entries to {rubric_path}")

                split_rubric_data(INPUT_PATH, QUESTION_PATH, RUBRIC_PATH)

            if STEP == 2:
                cmd = [
                    "swift", "rlhf",
                    "--rlhf_type", "grpo",
                    "--tuner_type", "lora",
                    "--model", MODEL_DIR,
                    "--dataset", QUESTION_PATH,
                    "--split_dataset_ratio", "0.15",   # 15% validation
                    "--model_type", "qwen2",
                    "--template", "qwen2_5",
                    "--external_plugins", "rubric_Reward2.py",
                    # "--reward_funcs", "rubric", "repetition", "soft_overlong",
                    # "--reward_weights", "0.6", "0.2", "0.2",
                    "--reward_funcs", "rubric",
                    "--reward_weights", "1.0",
                    "--beta", "0.05",
                    "--num_generations", "4",
                    "--num_train_epochs", "2",
                    "--learning_rate", "1e-5",
                    "--optim", "adamw_torch",
                    "--per_device_train_batch_size", "4",
                    "--per_device_eval_batch_size", "4",
                    "--gradient_accumulation_steps", "
                    "--torch_dtype", "bfloat16",
                    "--attn_impl", "eager",
                    "--temperature", "1.0",
                    "--top_p", "0.9",
                    "--top_k", "50",
                    "--max_length", "512",
                    "--max_new_tokens", "512",
                    "--max_completion_length", "512",
                    "--response_length", "512",
                    "--soft_cache_length", "96",
                    "--lora_rank", "8",
                    "--lora_alpha", "16",
                    "--lora_dropout", "0.05",
                    "--do_eval", "true",
                    "--eval_strategy", "steps",
                    "--eval_steps", "10",
                    "--save_strategy", "steps",
                    "--save_steps", "100",
                    "--save_total_limit", "3",
                    "--save_only_model", "true",
                    "--output_dir", f"Models/{Base_folder}/CoalBench-RaR-0.5B",
                    "--overwrite_output_dir", "true",
                    "--report_to", "swanlab",
                    "--swanlab_mode", "cloud",
                    "--swanlab_project", "ms-swift",
                    "--swanlab_exp_name", "coalbench-grpo-rubric-lora",
                    "--logging_steps", "1",
                    "--logging_first_step", "true",
                    "--log_completions", "true",
                    "--logging_dir", f"Models/{Base_folder}/CoalBench-RaR-0.5B/logs",
                    "--log_level", "info",
                ]

                subprocess.run(cmd, env=env, check=True)

            if STEP == 3:
                # --------------------------------------------------
                subprocess.run([
                    "swift", "infer",
                    "--model", MODEL_DIR,
                    "--dataset", Question_DATA,
                    "--split_dataset_ratio", "0.15",
                    "--template", "qwen2_5",
                    "--model_type", "qwen2",

                    "--max_new_tokens", "512",
                    "--temperature", "0",

                    "--result_path",
                    f"{OUTPUT_DIR}/base_predictions.jsonl"
                ], env=env, check=True)

            if STEP == 4:
                # --------------------------------------------------
                subprocess.run([
                    "swift", "infer",
                    "--model", MODEL_DIR,
                    "--adapters", RaR_ADAPTER,

                    "--dataset", Question_DATA,
                    "--split_dataset_ratio", "0.15",

                    "--template", "qwen2_5",
                    "--model_type", "qwen2",

                    "--max_new_tokens", "512",
                    "--temperature", "0",

                    "--result_path",
                    f"{OUTPUT_DIR}/RaR_predictions.jsonl"
                ], env=env, check=True)

            if STEP == 5:
                # ============================================================
                # Paths
                # ============================================================
                MODEL_PATH = os.path.expanduser("~/.cache/modelscope/hub/models/BAAI/bge-m3")

                RUN_DIR = Path("/root/Lingkai/CoalBench/Models/Qwen2.5-0.5B/CoalBench-RaR-0.5B")
                VAL_DIR = RUN_DIR / "v3-20260625-083624"

                BASE_PRED_PATH = RUN_DIR / "base_predictions.jsonl"
                RAR_PRED_PATH = RUN_DIR / "RaR_predictions.jsonl"
                VAL_DATASET_PATH = VAL_DIR / "val_dataset.jsonl"

                RAR_DATA_PATH = Path("/root/Lingkai/CoalBench/DataEn/CoalBench-RaR-EN.json")

                OUT_DIR = RUN_DIR / "figures"
                OUT_DIR.mkdir(parents=True, exist_ok=True)

                OUT_JSON = OUT_DIR / "rar_reward_comparison.json"
                OUT_JSONL = OUT_DIR / "rar_reward_comparison.jsonl"


                # ============================================================
                # Config
                # ============================================================
                SIM_FLOOR = 0.55
                PASS_LOG_THRESHOLD = 0.70

                BASE_SCORE_KEY = "base_score"
                RAR_SCORE_KEY = "RaR_score"


                # ============================================================
                # Matplotlib style
                # ============================================================
                plt.rcParams.update({
                    "font.family": "Arial",
                    "font.size": 9,
                    "axes.labelsize": 9,
                    "axes.titlesize": 10,
                    "legend.fontsize": 8,
                    "xtick.labelsize": 8,
                    "ytick.labelsize": 8,
                    "figure.dpi": 300,
                    "savefig.dpi": 600,
                    "axes.linewidth": 0.8,
                    "lines.linewidth": 1.2,
                })


                # ============================================================
                # IO utilities
                # ============================================================
                def load_json(path):
                    path = Path(path)
                    with path.open("r", encoding="utf-8") as f:
                        return json.load(f)


                def load_jsonl(path):
                    path = Path(path)
                    rows = []
                    with path.open("r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                rows.append(json.loads(line))
                    return rows


                def extract_response(obj):
                    """
                    Extract model response from Swift prediction jsonl row.
                    Expected:
                        {"response": "..."}
                    Fallbacks are included for robustness.
                    """
                    if "response" in obj:
                        return obj["response"]

                    if "predict" in obj:
                        return obj["predict"]

                    if "prediction" in obj:
                        return obj["prediction"]

                    if "generated_text" in obj:
                        return obj["generated_text"]

                    raise KeyError(f"Cannot find response field in prediction row. Keys: {list(obj.keys())}")


                def extract_id_from_val_row(obj):
                    """
                    Extract ID from val_dataset.jsonl row:
                    {
                    "id": 685,
                    "messages": [
                        {
                        "role": "user",
                        "content": "ID:685\nQuestion..."
                        }
                    ]
                    }
                    """
                    if "id" in obj:
                        return int(obj["id"])

                    messages = obj.get("messages", [])
                    if messages:
                        content = messages[0].get("content", "")
                        match = re.search(r"ID\s*:\s*(\d+)", content)
                        if match:
                            return int(match.group(1))

                    raise ValueError(f"Cannot extract ID from validation row: {obj}")


                def extract_question_from_val_row(obj):
                    messages = obj.get("messages", [])
                    if not messages:
                        return ""

                    content = messages[0].get("content", "")
                    content = re.sub(r"^\s*ID\s*:\s*\d+\s*", "", content).strip()
                    return content


                # ============================================================
                # Load embedding model
                # ============================================================
                print("Loading embedding model...")
                model, tokenizer = get_model_processor(MODEL_PATH)
                model.eval()
                device = next(model.parameters()).device
                print(f"Embedding model loaded on device: {device}")


                # ============================================================
                # Encoder and reward functions
                # ============================================================
                @torch.no_grad()
                def encode_texts(texts, batch_size=32):
                    """
                    Encode a list of texts with mean pooling and L2 normalization.
                    """
                    all_embs = []

                    for start in range(0, len(texts), batch_size):
                        batch = texts[start:start + batch_size]

                        inputs = tokenizer(
                            batch,
                            return_tensors="pt",
                            truncation=True,
                            max_length=512,
                            padding=True,
                        )
                        inputs = {k: v.to(device) for k, v in inputs.items()}

                        out = model(**inputs, output_hidden_states=True)
                        hidden = out.hidden_states[-1]
                        mask = inputs["attention_mask"].unsqueeze(-1).float()

                        emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
                        emb = F.normalize(emb, p=2, dim=1)

                        all_embs.append(emb.detach().cpu())

                    return torch.cat(all_embs, dim=0)


                def sim_to_partial_score(sim, floor=SIM_FLOOR):
                    if sim <= floor:
                        return 0.0
                    return (sim - floor) / (1.0 - floor)


                def compute_reward_from_embeddings(response_emb, rubric_embs, rubrics):
                    """
                    Same logic as rubric_Reward2.py:

                    score = sum_i w_i * partial_i
                    denominator = sum positive weights only
                    reward = clip(score / denominator, 0, 1)
                    """
                    sims = (rubric_embs @ response_emb.T).squeeze(-1).numpy()

                    score = 0.0
                    total_positive_weight = 0.0
                    rubric_logs = []

                    for sim, r in zip(sims, rubrics):
                        title = r.get("title", "")
                        desc = r.get("description", "")
                        weight = float(r.get("weight", 1.0))

                        partial = sim_to_partial_score(float(sim), SIM_FLOOR)

                        score += weight * partial

                        if weight > 0:
                            total_positive_weight += weight

                        rubric_logs.append({
                            "rubric_id": r.get("rubric_id", None),
                            "title": title,
                            "description": desc,
                            "weight": weight,
                            "similarity": float(sim),
                            "partial_score": float(partial),
                            "passed_for_log": bool(sim > PASS_LOG_THRESHOLD),
                        })

                    if total_positive_weight > 0:
                        final_reward = score / total_positive_weight
                    else:
                        final_reward = 0.0

                    final_reward = max(0.0, min(1.0, final_reward))

                    return final_reward, score, total_positive_weight, rubric_logs


                # ============================================================
                # Plotting
                # ============================================================
                def format_axis(ax):
                    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.35)
                    for spine in ["top", "right"]:
                        ax.spines[spine].set_visible(False)


                def plot_rar_scatter(results, out_dir):
                    base_scores = np.array([float(x[BASE_SCORE_KEY]) for x in results])
                    rar_scores = np.array([float(x[RAR_SCORE_KEY]) for x in results])

                    n = len(base_scores)

                    improved = rar_scores > base_scores
                    degraded = rar_scores < base_scores
                    tied = rar_scores == base_scores

                    base_avg = base_scores.mean()
                    rar_avg = rar_scores.mean()
                    mean_gain = (rar_scores - base_scores).mean()

                    fig, ax = plt.subplots(figsize=(5.4, 3.2))

                    colors = {
                        "RaR better": "tab:green",
                        "Tie": "tab:orange",
                        "Base better": "tab:blue",
                    }

                    # Rounded score pairs for bubble counting
                    rounded_pairs = [
                        (round(float(b), 3), round(float(r), 3))
                        for b, r in zip(base_scores, rar_scores)
                    ]

                    from collections import Counter
                    pair_counts = Counter(rounded_pairs)

                    labels_used = set()

                    for (base_score, rar_score), count in sorted(pair_counts.items()):
                        if rar_score > base_score:
                            label = "RaR better"
                        elif rar_score < base_score:
                            label = "Base better"
                        else:
                            label = "Tie"

                        size = 28 + 45 * np.sqrt(count)

                        ax.scatter(
                            base_score,
                            rar_score,
                            s=size,
                            color=colors[label],
                            alpha=0.78,
                            edgecolors="black",
                            linewidths=0.45,
                            label=label if label not in labels_used else None,
                            zorder=3,
                        )

                        if count > 1:
                            ax.text(
                                base_score,
                                rar_score,
                                str(count),
                                ha="center",
                                va="center",
                                fontsize=6.5,
                                fontweight="bold",
                                color="black",
                                zorder=4,
                            )

                        labels_used.add(label)

                    min_v = max(0.0, min(base_scores.min(), rar_scores.min()) - 0.03)
                    max_v = min(1.0, max(base_scores.max(), rar_scores.max()) + 0.03)

                    ax.plot(
                        [min_v, max_v],
                        [min_v, max_v],
                        linestyle="--",
                        linewidth=1.1,
                        color="tab:blue",
                        label="Equal reward",
                        zorder=2,
                    )

                    improved_ratio = improved.sum() / n * 100
                    tied_ratio = tied.sum() / n * 100
                    degraded_ratio = degraded.sum() / n * 100

                    summary_text = (
                        f"n = {n}\n"
                        f"Improved: {improved_ratio:.1f}%\n"
                        f"Base: {base_avg:.3f}\n"
                        f"RaR : {rar_avg:.3f}"
                    )

                    ax.text(
                        0.04,
                        0.96,
                        summary_text,
                        transform=ax.transAxes,
                        fontsize=7.2,
                        va="top",
                        linespacing=1.1,
                        bbox=dict(
                            boxstyle="round,pad=0.25",
                            facecolor="white",
                            edgecolor="0.7",
                            alpha=0.90,
                            linewidth=0.6,
                        ),
                        zorder=5,
                    )

                    ax.set_xlim(min_v, max_v)
                    ax.set_ylim(min_v, max_v)
                    ax.set_xlabel("Base model rubric reward")
                    ax.set_ylabel("RaR model rubric reward")

                    handles, labels = ax.get_legend_handles_labels()

                    order = ["RaR better", "Tie", "Base better"]

                    ordered = [
                        (h, l)
                        for l in order
                        for h, lab in zip(handles, labels)
                        if lab == l
                    ]

                    if ordered:
                        handles, labels = zip(*ordered)
                        ax.legend(
                            handles,
                            labels,
                            frameon=False,
                            loc="lower right",
                        )

                    format_axis(ax)

                    png_path = out_dir / "ScatterRaR.png"
                    fig.tight_layout()
                    fig.savefig(png_path, dpi=600, bbox_inches="tight")
                    plt.close(fig)
                    print(f"Saved: {png_path}")

                # ============================================================
                # Main
                # ============================================================
                def main():
                    print("Loading files...")

                    base_rows = load_jsonl(BASE_PRED_PATH)
                    rar_rows = load_jsonl(RAR_PRED_PATH)
                    val_rows = load_jsonl(VAL_DATASET_PATH)
                    rar_data = load_json(RAR_DATA_PATH)

                    base_outputs = [extract_response(x) for x in base_rows]
                    rar_outputs = [extract_response(x) for x in rar_rows]
                    sample_ids = [extract_id_from_val_row(x) for x in val_rows]
                    questions = [extract_question_from_val_row(x) for x in val_rows]

                    n = min(len(base_outputs), len(rar_outputs), len(sample_ids))

                    base_outputs = base_outputs[:n]
                    rar_outputs = rar_outputs[:n]
                    sample_ids = sample_ids[:n]
                    questions = questions[:n]

                    print(f"Base predictions: {len(base_rows)}")
                    print(f"RaR predictions:  {len(rar_rows)}")
                    print(f"Validation rows:  {len(val_rows)}")
                    print(f"Matched samples:  {n}")

                    rubric_bank = {
                        int(item["id"]): item.get("rubrics", [])
                        for item in rar_data
                    }

                    missing = [sid for sid in sample_ids if sid not in rubric_bank]
                    if missing:
                        print(f"[WARN] Missing rubric IDs: {missing[:10]} ... total={len(missing)}")

                    print("Scoring responses...")

                    results = []

                    for idx, (sid, question, base_resp, rar_resp) in enumerate(
                        zip(sample_ids, questions, base_outputs, rar_outputs)
                    ):
                        rubrics = rubric_bank.get(sid, [])

                        if not rubrics:
                            print(f"[WARN] No rubrics found for id={sid}")
                            continue

                        rubric_texts = [
                            f"{r.get('title', '')}. {r.get('description', '')}".strip()
                            for r in rubrics
                            if r.get("title", "") or r.get("description", "")
                        ]

                        valid_rubrics = [
                            r for r in rubrics
                            if r.get("title", "") or r.get("description", "")
                        ]

                        # Encode rubrics + two responses
                        texts_to_encode = rubric_texts + [base_resp, rar_resp]
                        embs = encode_texts(texts_to_encode, batch_size=32)

                        rubric_embs = embs[:len(rubric_texts)]
                        base_emb = embs[len(rubric_texts):len(rubric_texts) + 1]
                        rar_emb = embs[len(rubric_texts) + 1:len(rubric_texts) + 2]

                        base_reward, base_raw, base_pos_w, base_logs = compute_reward_from_embeddings(
                            base_emb, rubric_embs, valid_rubrics
                        )
                        rar_reward, rar_raw, rar_pos_w, rar_logs = compute_reward_from_embeddings(
                            rar_emb, rubric_embs, valid_rubrics
                        )

                        results.append({
                            "index": idx,
                            "id": sid,
                            "question": question,

                            "base_response": base_resp,
                            "RaR_response": rar_resp,

                            "base_score": base_reward,
                            "RaR_score": rar_reward,
                            "delta": rar_reward - base_reward,

                            "base_score_raw": base_raw,
                            "RaR_score_raw": rar_raw,
                            "total_positive_weight": base_pos_w,

                            "base_rubric_logs": base_logs,
                            "RaR_rubric_logs": rar_logs,
                        })

                        if (idx + 1) % 10 == 0 or idx == 0:
                            print(
                                f"[{idx + 1}/{n}] id={sid} "
                                f"base={base_reward:.3f} rar={rar_reward:.3f} "
                                f"delta={rar_reward - base_reward:.3f}"
                            )

                    # Save JSON and JSONL
                    with OUT_JSON.open("w", encoding="utf-8") as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)

                    with OUT_JSONL.open("w", encoding="utf-8") as f:
                        for row in results:
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")

                    print(f"Saved: {OUT_JSON}")
                    print(f"Saved: {OUT_JSONL}")

                    # Summary
                    base_scores = np.array([x["base_score"] for x in results])
                    rar_scores = np.array([x["RaR_score"] for x in results])
                    deltas = rar_scores - base_scores

                    improved = np.sum(deltas > 0)
                    tied = np.sum(deltas == 0)
                    degraded = np.sum(deltas < 0)

                    print("\n========== Summary ==========")
                    print(f"Samples: {len(results)}")
                    print(f"RaR better: {improved} ({improved / len(results) * 100:.1f}%)")
                    print(f"Tie: {tied} ({tied / len(results) * 100:.1f}%)")
                    print(f"Base better: {degraded} ({degraded / len(results) * 100:.1f}%)")
                    print(f"Base avg. reward: {base_scores.mean():.4f}")
                    print(f"RaR avg. reward:  {rar_scores.mean():.4f}")
                    print(f"Mean gain:        {deltas.mean():.4f}")
                    print(f"Median gain:      {np.median(deltas):.4f}")

                    # Plot
                    plot_rar_scatter(results, OUT_DIR)

                if __name__ == "__main__":
                    main()

    if STAGE == 4:
        # =========================
        # File paths
        # =========================
        Folder = "Models/Qwen2.5-0.5B/LearningCurve/"
        SFT_CSV = Folder + "coalbench-sft-lora-2026-6-25_15_29_18.csv"
        DPO_CSV = Folder + "coalbench-dpo-2026-6-25_15_29_08.csv"
        RAR_REWARD_CSV = Folder + "coalbench-grpo-rubric-lora-2026-6-25_15_28_14.csv"
        RAR_KL_CSV = Folder + "coalbench-grpo-rubric-lora-2026-6-25_15_29_00.csv"

        OUT_DIR = Path(Folder + "figures")
        OUT_DIR.mkdir(exist_ok=True)

        # =========================
        # Global style
        # =========================
        plt.rcParams.update({
            "font.family": "Arial",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
        })

        def savefig(name):
            plt.tight_layout()
            plt.savefig(OUT_DIR / f"{name}.png", bbox_inches="tight")
            plt.close()

        def plot_series(ax, df, x_col, y_col, label, linestyle="-"):
            if y_col not in df.columns:
                print(f"[WARN] Column not found: {y_col}")
                return
            sub = df[[x_col, y_col]].dropna()
            ax.plot(sub[x_col], sub[y_col], label=label, linestyle=linestyle)

        # =========================
        # Figure 1: SFT loss
        # =========================
        sft = pd.read_csv(SFT_CSV)

        fig, ax = plt.subplots(figsize=(3.5, 2.6))

        plot_series(
            ax, sft, "step",
            "coalbench-sft-lora-train/loss_step",
            "Training",
            "-"
        )

        plot_series(
            ax, sft, "step",
            "coalbench-sft-lora-eval/loss_step",
            "Validation",
            "--"
        )

        ax.set_title("SFT Loss")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Loss")
        ax.legend(frameon=False)
        ax.grid(True, linewidth=0.4, alpha=0.4)

        savefig("SwanLabSFT")

        # =========================
        # Figure 2: DPO, two-panel version
        # =========================
        dpo = pd.read_csv(DPO_CSV)

        dpo["train_margin"] = (
            dpo["coalbench-dpo-train/rewards/chosen_step"]
            - dpo["coalbench-dpo-train/rewards/rejected_step"]
        )

        dpo["eval_margin"] = (
            dpo["coalbench-dpo-eval/rewards/chosen_step"]
            - dpo["coalbench-dpo-eval/rewards/rejected_step"]
        )

        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))

        # ---- Panel (a): chosen/rejected rewards
        ax = axes[0]

        plot_series(
            ax, dpo, "step",
            "coalbench-dpo-train/rewards/chosen_step",
            "Training chosen",
            "-"
        )

        plot_series(
            ax, dpo, "step",
            "coalbench-dpo-train/rewards/rejected_step",
            "Training rejected",
            "--"
        )

        plot_series(
            ax, dpo, "step",
            "coalbench-dpo-eval/rewards/chosen_step",
            "Validation chosen",
            "-."
        )

        plot_series(
            ax, dpo, "step",
            "coalbench-dpo-eval/rewards/rejected_step",
            "Validation rejected",
            ":"
        )

        ax.set_title("Preference Rewards")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Reward")
        ax.legend(frameon=False, ncol=1)
        ax.grid(True, linewidth=0.4, alpha=0.4)

        # ---- Panel (b): preference margin
        ax = axes[1]

        plot_series(ax, dpo, "step", "train_margin", "Training", "-")
        plot_series(ax, dpo, "step", "eval_margin", "Validation", "--")

        ax.axhline(0, linewidth=0.8, linestyle=":")
        ax.set_title("Preference Margin")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Chosen $-$ rejected reward")
        ax.legend(frameon=False)
        ax.grid(True, linewidth=0.4, alpha=0.4)

        savefig("SwanLabDPO")

        # =========================
        # Figure 3: RaR / GRPO, two-panel version
        # =========================
        rar_reward = pd.read_csv(RAR_REWARD_CSV)
        rar_kl = pd.read_csv(RAR_KL_CSV)

        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))

        # ---- Panel (a): reward
        ax = axes[0]

        plot_series(
            ax, rar_reward, "step",
            "coalbench-grpo-rubric-lora-train/reward_step",
            "Training",
            "-"
        )

        plot_series(
            ax, rar_reward, "step",
            "coalbench-grpo-rubric-lora-eval/reward_step",
            "Validation",
            "--"
        )

        ax.set_title("Reward")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Reward")
        ax.legend(frameon=False)
        ax.grid(True, linewidth=0.4, alpha=0.4)

        # ---- Panel (b): KL divergence
        ax = axes[1]

        plot_series(
            ax, rar_kl, "step",
            "coalbench-grpo-rubric-lora-train/kl_step",
            "Training",
            "-"
        )

        plot_series(
            ax, rar_kl, "step",
            "coalbench-grpo-rubric-lora-eval/kl_step",
            "Validation",
            "--"
        )

        ax.set_title("KL Divergence")
        ax.set_xlabel("Training step")
        ax.set_ylabel("KL divergence")
        ax.legend(frameon=False)
        ax.grid(True, linewidth=0.4, alpha=0.4)

        savefig("SwanLabRaR")

        print(f"Figures saved to: {OUT_DIR.resolve()}")

    if STAGE == 5:
        BASE = Path("/root/Lingkai/CoalBench/Models/Qwen2.5-0.5B")

        files = {
            "Base": BASE / "CoalBench-SFT-0.5B/base_predictions.jsonl",
            "SFT":  BASE / "CoalBench-SFT-0.5B/sft_predictions.jsonl",
            "DPO":  BASE / "CoalBench-DPO-0.5B/dpo_predictions.jsonl",
            "RaR":  BASE / "CoalBench-RaR-0.5B/RaR_predictions.jsonl",
        }

        OUTPUT_PATH = Path("comparison_all.jsonl")


        def read_jsonl(path):
            data = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            return data


        def extract_question(item):
            if "messages" in item and len(item["messages"]) > 0:
                return item["messages"][0].get("content", "")
            return item.get("query", item.get("question", ""))


        # ==========================================
        # Load all predictions
        # ==========================================
        base_data = read_jsonl(files["Base"])
        sft_data  = read_jsonl(files["SFT"])
        dpo_data  = read_jsonl(files["DPO"])
        rar_data  = read_jsonl(files["RaR"])

        n = min(len(base_data), len(sft_data), len(dpo_data), len(rar_data))

        print(f"Base instances: {len(base_data)}")
        print(f"SFT instances:  {len(sft_data)}")
        print(f"DPO instances:  {len(dpo_data)}")
        print(f"RaR instances:  {len(rar_data)}")
        print(f"Saving aligned instances: {n}")

        # ==========================================
        # Save all comparisons
        # ==========================================
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            for i in range(n):
                comparison = {
                    "instance": i + 1,
                    "question": extract_question(base_data[i]),
                    "label": base_data[i].get("labels", ""),
                    "base": base_data[i].get("response", ""),
                    "sft": sft_data[i].get("response", ""),
                    "dpo": dpo_data[i].get("response", ""),
                    "rar": rar_data[i].get("response", ""),
                }

                f.write(json.dumps(comparison, ensure_ascii=False) + "\n")

        print(f"Saved to: {OUTPUT_PATH.resolve()}")

    if STAGE == 6:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # =========================
        # File paths
        # =========================
        DATASETS = {
            "CoalBench-EN": Path("DataEn/CoalBench-EN.json"),
            "CoalBench-ZH": Path("DataCn/CoalBench-ZH.json"),
        }

        Folder = "Models/Qwen2.5-0.5B/ClassDistibution/"
        OUT_DIR = Path(Folder + "figures")
        OUT_DIR.mkdir(exist_ok=True)

        # =========================
        # Global style, same as DPO figures
        # =========================
        plt.rcParams.update({
            "font.family": "Arial",
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 8,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
        })


        # =========================
        # Chinese to English mapping
        # =========================
        ZH_TO_EN = {
            "不安全行为": "Unsafe Behavior",
            "主运输": "Main Transportation",
            "井工开采": "Underground Mining",
            "供水": "Water Supply",
            "供配电": "Power Supply and Distribution",
            "储装运": "Storage, Loading and Transport",
            "冲击地压防治": "Rockburst Prevention and Control",
            "地质": "Geology",
            "安全培训": "Safety Training",
            "安全监控": "Safety Monitoring",
            "应急救援": "Emergency Rescue",
            "排水": "Mine Drainage",
            "掘进与巷道工程": "Excavation and Roadway Engineering",
            "提升": "Mine Hoisting",
            "支护技术与设备": "Support Technology and Equipment",
            "智能矿山": "Intelligent Mine",
            "机电": "Mechanical and Electrical Engineering",
            "水害防治": "Mine Water Control",
            "洗选": "Coal Preparation",
            "火灾防治": "Mine Fire Prevention and Control",
            "热害治理": "Thermal Hazard Control",
            "煤化工": "Coal Chemical Engineering",
            "煤基材料": "Coal-Based Materials",
            "爆破": "Blasting",
            "环保与生态恢复": "Environmental and Ecological Restoration",
            "瓦斯防治": "Methane Control in Mining",
            "矿工": "Miners",
            "碳减排与循环利用": "Carbon Reduction and Circular Utilization",
            "粉尘防治": "Mine Dust Control",
            "职业健康": "Occupational Health",
            "调度": "Dispatching",
            "辅助运输": "Auxiliary Transportation",
            "通讯联络": "Communication and Liaison",
            "通风": "Mine Ventilation",
            "降温制冷": "Cooling and Refrigeration",
            "露天开采": "Surface Mining",
            "顶板": "Ground Control",
            "其它": "Others",
        }

        # -----------------------------
        # Text length functions
        # -----------------------------
        def en_word_count(text: str) -> int:
            return len(text.split())


        def zh_char_count(text: str) -> int:
            return len(text.replace(" ", ""))


        # -----------------------------
        # Storage
        # -----------------------------
        summary = {}
        class_distributions = {}

        # -----------------------------
        # Compute stats
        # -----------------------------
        for name, path in DATASETS.items():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # unify classes
            if name == "CoalBench-ZH":
                classes = [
                    ZH_TO_EN.get(x.get("class", "UNKNOWN"), x.get("class", "UNKNOWN"))
                    for x in data
                ]
            else:
                classes = [x.get("class", "UNKNOWN") for x in data]

            class_counter = Counter(classes)
            class_distributions[name] = class_counter

            rubric_counts = [len(x.get("rubrics", [])) for x in data]

            # lengths
            if "EN" in name:
                q_len = [en_word_count(x.get("question", "")) for x in data]
                a1_len = [en_word_count(x.get("answer1", "")) for x in data]
                a2_len = [en_word_count(x.get("answer2", "")) for x in data]
            else:
                q_len = [zh_char_count(x.get("question", "")) for x in data]
                a1_len = [zh_char_count(x.get("answer1", "")) for x in data]
                a2_len = [zh_char_count(x.get("answer2", "")) for x in data]

            summary[name] = {
                "instances": len(data),
                "num_classes": len(class_counter),
                "total_rubrics": sum(rubric_counts),

                # integer-only averages
                "avg_rubrics": int(round(sum(rubric_counts) / len(data))),
                "avg_q": int(round(sum(q_len) / len(data))),
                "avg_a1": int(round(sum(a1_len) / len(data))),
                "avg_a2": int(round(sum(a2_len) / len(data))),
            }

        # -----------------------------
        # PRINT TABLE
        # -----------------------------
        print("\n" + "=" * 110)
        print("CoalBench Dataset Summary (Integer-only, ZH→EN unified classes)")
        print("=" * 110)

        metrics = [
            "Instances",
            "Number of Classes",
            "Total Rubrics",
            "Avg Rubrics",
            "Avg Question Length",
            "Avg Answer1 Length",
            "Avg Answer2 Length",
        ]

        print(f"{'Metric':<30}{'CoalBench-EN':<25}{'CoalBench-ZH':<25}")
        print("-" * 80)

        for m in metrics:
            en = summary["CoalBench-EN"]
            zh = summary["CoalBench-ZH"]

            if m == "Instances":
                en_val, zh_val = en["instances"], zh["instances"]
            elif m == "Number of Classes":
                en_val, zh_val = en["num_classes"], zh["num_classes"]
            elif m == "Total Rubrics":
                en_val, zh_val = en["total_rubrics"], zh["total_rubrics"]
            elif m == "Avg Rubrics":
                en_val, zh_val = en["avg_rubrics"], zh["avg_rubrics"]
            elif m == "Avg Question Length":
                en_val, zh_val = en["avg_q"], zh["avg_q"]
            elif m == "Avg Answer1 Length":
                en_val, zh_val = en["avg_a1"], zh["avg_a1"]
            elif m == "Avg Answer2 Length":
                en_val, zh_val = en["avg_a2"], zh["avg_a2"]

            print(f"{m:<30}{en_val:<25}{zh_val:<25}")

        # -----------------------------
        # BAR PLOT FUNCTION (SORT + COLOR + LABELS)
        # -----------------------------
        def plot_bar(counter, title, filename):
            items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
            labels = [x[0] for x in items]
            values = [x[1] for x in items]

            colors = []
            for v in values:
                if v > 50:
                    colors.append("red")
                elif 20 <= v <= 50:
                    colors.append("blue")
                else:
                    colors.append("green")

            plt.figure(figsize=(14, 6))
            bars = plt.bar(labels, values, color=colors)

            ax = plt.gca()
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ax.spines["left"].set_visible(False)
            ax.grid(axis="y", linestyle="--", alpha=0.4)

            for bar, v in zip(bars, values):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    str(v),
                    ha="center",
                    va="bottom"
                )

            plt.xticks(rotation=45, ha="right")
            plt.title(title)
            plt.xlabel("Class")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(filename, dpi=300)
            plt.close()


        # -----------------------------
        # PLOTS
        # -----------------------------
        plot_bar(
            class_distributions["CoalBench-EN"],
            "CoalBench-EN Class Distribution",
            OUT_DIR / "CoalBenchENclassdistribution.png"
        )

        plot_bar(
            class_distributions["CoalBench-ZH"],
            "CoalBench-ZH Class Distribution",
            OUT_DIR / "CoalBenchZHclassdistribution.png"
        )

        print(class_distributions["CoalBench-EN"])

    if STAGE == 7:
        DATASETS = {
            "English": {
                "folder": Path("DataEn"),
                "files": {
                    "Full": "CoalBench-EN.json",
                    "SFT": "CoalBench-SFT-EN.json",
                    "DPO": "CoalBench-DPO-EN.json",
                    "RaR": "CoalBench-RaR-EN.json",
                },
            },
            "Chinese": {
                "folder": Path("DataCn"),
                "files": {
                    "Full": "CoalBench-ZH.json",
                    "SFT": "CoalBench-SFT-ZH.json",
                    "DPO": "CoalBench-DPO-ZH.json",
                    "RaR": "CoalBench-RaR-ZH.json",
                },
            },
        }

        REQUIRED_FIELDS = {
            "Full": ["id", "class", "question", "answer1", "answer2", "rubrics"],
            "SFT": ["id", "class", "question", "answer"],
            "DPO": ["id", "class", "question", "chosen", "rejected"],
            "RaR": ["id", "class", "question", "rubrics"],
        }

        REQUIRED_RUBRIC_FIELDS = ["rubric_id", "title", "description", "weight"]


        def load_json(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)


        def count_duplicate_ids(ids):
            return sum(count > 1 for count in Counter(ids).values())


        def validate_required_fields(data, required_fields):
            missing = 0

            for item in data:
                for field in required_fields:
                    if field not in item:
                        missing += 1

            return missing


        def validate_rubrics(data):
            rubric_count = 0
            rubric_errors = 0

            for item in data:
                rubrics = item.get("rubrics", [])

                if not isinstance(rubrics, list):
                    rubric_errors += 1
                    continue

                rubric_count += len(rubrics)

                for rubric in rubrics:
                    if not isinstance(rubric, dict):
                        rubric_errors += 1
                        continue

                    for field in REQUIRED_RUBRIC_FIELDS:
                        if field not in rubric:
                            rubric_errors += 1

                    if "weight" in rubric:
                        try:
                            float(rubric["weight"])
                        except Exception:
                            rubric_errors += 1

            return rubric_count, rubric_errors


        print("=" * 110)
        print(
            f"{'File':30} {'Records':>8} {'Parse':>8} "
            f"{'Missing':>10} {'DupIDs':>8} {'Align':>8} {'RubricErr':>10}"
        )
        print("=" * 110)

        for lang, info in DATASETS.items():
            folder = info["folder"]
            files = info["files"]

            full_path = folder / files["Full"]

            try:
                full_data = load_json(full_path)
            except Exception as e:
                print(f"{files['Full']:30} {'-':>8} {'FAIL':>8}  {e}")
                continue

            full_ids = {item.get("id") for item in full_data}

            for task, filename in files.items():
                path = folder / filename

                try:
                    data = load_json(path)
                    parse_status = "OK"
                except Exception as e:
                    print(f"{filename:30} {'-':>8} {'FAIL':>8}  {e}")
                    continue

                ids = [item.get("id") for item in data]

                missing = validate_required_fields(data, REQUIRED_FIELDS[task])
                duplicate_ids = count_duplicate_ids(ids)

                if task == "Full":
                    alignment_errors = "--"
                else:
                    alignment_errors = len(set(ids) - full_ids)

                if task in ["Full", "RaR"]:
                    rubric_count, rubric_errors = validate_rubrics(data)
                else:
                    rubric_count, rubric_errors = None, "--"

                records = rubric_count if task == "RaR" else len(data)

                print(
                    f"{filename:30}"
                    f"{records:8,d}"
                    f"{parse_status:>8}"
                    f"{missing:10}"
                    f"{duplicate_ids:8}"
                    f"{str(alignment_errors):>8}"
                    f"{str(rubric_errors):>10}"
                )

        print("=" * 110)

        FILES = [
            Path("DataEn/CoalBench-EN.json"),
            Path("DataEn/CoalBench-SFT-EN.json"),
            Path("DataEn/CoalBench-DPO-EN.json"),
            Path("DataEn/CoalBench-RaR-EN.json"),
            Path("DataCn/CoalBench-ZH.json"),
            Path("DataCn/CoalBench-SFT-ZH.json"),
            Path("DataCn/CoalBench-DPO-ZH.json"),
            Path("DataCn/CoalBench-RaR-ZH.json"),
        ]

        print("=" * 50)
        print(f"{'File':30} {'Lines':>10}")
        print("=" * 50)

        for file in FILES:
            with open(file, "r", encoding="utf-8") as f:
                num_lines = sum(1 for _ in f)

            print(f"{file.name:30} {num_lines:10,}")

        print("=" * 50)