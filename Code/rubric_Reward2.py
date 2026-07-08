import os
import json
import time
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from swift import get_model_processor
from swift.rewards import ORM, orms

# =========================
# CONFIG
# =========================
MODEL_PATH = os.path.expanduser("~/.cache/modelscope/hub/models/BAAI/bge-m3")

DEBUG = False

# Best recommendation:
# continuous floor-scaled reward
SIM_FLOOR = 0.55

# only for logging, not used in reward
PASS_LOG_THRESHOLD = 0.70

LOG_PATH = Path("reward_logs.jsonl")

RUBRIC_PATH = "/root/Lingkai/CoalBench/DataEn/RubricBank.json"

# =========================
# RUBRIC BANK
# =========================
with open(RUBRIC_PATH, "r", encoding="utf-8") as f:
    raw_rubric_bank = json.load(f)

rubric_bank = {int(k): v for k, v in raw_rubric_bank.items()}

# =========================
# MODEL
# =========================
model, tokenizer = get_model_processor(MODEL_PATH)
model.eval()
device = next(model.parameters()).device

# =========================
# LOGGER
# =========================
def init_log():
    with LOG_PATH.open("w", encoding="utf-8") as f:
        pass


def log_json(data):
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


init_log()
GLOBAL_STEP = 0

# =========================
# ENCODER
# =========================
def encode(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    hidden = out.hidden_states[-1]
    mask = inputs["attention_mask"].unsqueeze(-1).float()

    emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
    emb = F.normalize(emb, p=2, dim=1)

    return emb


def similarity(a, b):
    return (encode(a) @ encode(b).T).item()


def sim_to_partial_score(sim, floor=SIM_FLOOR):
    """
    Floor-scaled continuous reward.

    sim <= floor -> 0
    sim > floor  -> linearly scaled to [0, 1]

    Example with floor=0.55:
      sim=0.55 -> 0.00
      sim=0.65 -> 0.22
      sim=0.75 -> 0.44
      sim=0.85 -> 0.67
    """
    if sim <= floor:
        return 0.0
    return (sim - floor) / (1.0 - floor)


# =========================
# REWARD FUNCTION
# =========================
class RubricRewardFunction(ORM):
    def __call__(self, completions, **kwargs):
        global GLOBAL_STEP
        GLOBAL_STEP += 1

        start_time = time.time()
        messages = kwargs.get("messages", [])

        rewards = []

        for i, c in enumerate(completions):
            text = c.get("content", "") if isinstance(c, dict) else str(c)

            # -------------------------
            # Extract sample ID
            # Expected user message:
            # ID:549
            # question...
            # -------------------------
            try:
                user_text = messages[i][1]["content"]
                match = re.search(r"ID\s*:\s*(\d+)", user_text)
                sample_id = int(match.group(1)) if match else i
            except Exception:
                user_text = ""
                sample_id = i

            rubrics = rubric_bank.get(sample_id, [])

            if sample_id not in rubric_bank:
                print(f"[WARN] sample_id={sample_id} not in rubric_bank keys")

            if DEBUG:
                print("=" * 80)
                print("sample_id:", sample_id)
                print("user_text:", user_text)
                print("num_rubrics:", len(rubrics))
                print("completion:", text)

            score = 0.0
            total_positive_weight = 0.0
            rubric_logs = []

            # -------------------------
            # Evaluate rubrics
            # -------------------------
            for r in rubrics:
                if not isinstance(r, dict):
                    continue

                title = r.get("title", "")
                desc = r.get("description", "")
                weight = float(r.get("weight", 1.0))

                if not title and not desc:
                    continue

                # Use both title and description for stronger semantic signal
                rubric_text = f"{title}. {desc}".strip()

                sim = similarity(rubric_text, text)
                partial = sim_to_partial_score(sim, SIM_FLOOR)

                # Positive rubric: add reward
                if weight > 0:
                    score += weight * partial
                    total_positive_weight += weight

                # Negative rubric / pitfall: subtract reward if similar
                else:
                    score += weight * partial

                passed_for_log = sim > PASS_LOG_THRESHOLD

                rubric_logs.append({
                    "title": title,
                    "description": desc,
                    "weight": weight,
                    "similarity": sim,
                    "partial_score": partial,
                    "passed_for_log": passed_for_log,
                })

                if DEBUG:
                    print(
                        f"weight={weight:.2f}, sim={sim:.3f}, "
                        f"partial={partial:.3f}, passed_log={passed_for_log}"
                    )
                    print("rubric:", rubric_text)

            # Normalize by positive rubric weights only
            if total_positive_weight > 0:
                final_reward = score / total_positive_weight
            else:
                final_reward = 0.0

            # Keep reward in [0, 1] for stable GRPO
            final_reward = max(0.0, min(1.0, final_reward))

            rewards.append(final_reward)

            log_json({
                "step": GLOBAL_STEP,
                "sample_index": i,
                "sample_id": sample_id,
                "timestamp": time.time(),

                "input_messages": messages[i] if i < len(messages) else None,
                "completion": text,
                "completion_length": len(text),

                "rubrics": rubric_logs,

                "score_raw": score,
                "score_total_positive_weight": total_positive_weight,
                "final_reward": final_reward,
            })

            print(f"[STEP {GLOBAL_STEP}] sample_id={sample_id} reward={final_reward:.3f}")

        elapsed = time.time() - start_time

        log_json({
            "step": GLOBAL_STEP,
            "type": "step_summary",
            "batch_size": len(completions),
            "avg_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "latency_sec": elapsed,
        })

        return rewards


# =========================
# REGISTER
# =========================
orms["rubric"] = RubricRewardFunction