import json
from pathlib import Path
import shutil
from tqdm import tqdm
import re
import copy

import os
os.environ["ASCEND_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"
from swift.infer_engine import TransformersEngine, RequestConfig, InferRequest

BATCH_SIZE = 24
Path_model = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-32B"

# STAGE = 1   # Pure id, class, q and reference data
# STAGE = 2   # Qwen32B for generation (DPO) question-only
# STAGE = 3   # Qwen32B for DPO question-class-reference as input
# STAGE = 4   # Qwen32B for Rubric GPRO (given question and answer2)
# STAGE = 5   # regenerate rubrics for some empty (given question and answer2)
# STAGE = 6   # remove samples with empty rubrics and finalize dataset, re-id

PATH_in = Path("DataCn/Input.json")
PATH_DPO = Path("DataCn/CoalBench_CN.json")

for STAGE in [1,2,3,4,5,6]:
    if STAGE == 1:
        RUBRICS_PATH = Path("DataCn/Rubrics.json")
        BACKUP_PATH = Path("DataCn/Input.json")

        def build_input_dataset():
            with open(RUBRICS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)

            output = []

            for new_id, item in enumerate(data, start=1):
                question = ""

                for msg in item.get("messages", []):
                    if msg.get("role") == "user":
                        question = msg.get("content", "")
                        break

                output.append({
                    "id": new_id,  # reindex from 1..N
                    "class": item.get("class", ""),
                    "question": question,
                    "answer_reference": item.get("answer_reference", "")
                })

            with open(BACKUP_PATH, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

            print(f"Saved {len(output)} samples to {BACKUP_PATH}")


        if __name__ == "__main__":
            build_input_dataset()

    if STAGE == 2:
        with open(PATH_in, "r", encoding="utf-8") as f:
            data = json.load(f)

        engine = TransformersEngine(Path_model, max_batch_size=BATCH_SIZE)

        request_config = RequestConfig(
            max_tokens=512,
            temperature=0
        )

        infer_requests = [
            InferRequest(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是一名煤矿科学与工程领域的专业问答助手。\n"
                            "请直接回答用户问题。\n"
                            "不要输出任何推理过程、思考步骤或分析内容。\n"
                            "不要输出 <think> 标签。\n"
                            "只输出最终答案，不要添加多余解释。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": x["question"]
                    }
                ]
            )
            for x in tqdm(data, desc="Preparing requests")
        ]

        resp_list = engine.infer(infer_requests, request_config)

        output_data = []
        sample_question = None
        sample_answer = None

        for x, r in zip(data, resp_list):
            answer = r.choices[0].message.content

            # Remove any leaked thinking blocks
            answer = re.sub(
                r"<think>.*?</think>",
                "",
                answer,
                flags=re.DOTALL | re.IGNORECASE
            ).strip()

            output_data.append(
                {
                    "id": x["id"],
                    "class": x["class"],
                    "question": x["question"],
                    "answer1": answer
                }
            )

        with open(PATH_DPO, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(output_data)} samples -> {PATH_DPO}")

    if STAGE == 3:
        with open(PATH_in, "r", encoding="utf-8") as f:
            ref_data = {x["id"]: x for x in json.load(f)}

        with open(PATH_DPO, "r", encoding="utf-8") as f:
            dpo_data = json.load(f)

        engine = TransformersEngine(Path_model, max_batch_size=BATCH_SIZE)
        request_config = RequestConfig(
            max_tokens=512,
            temperature=0
        )

        def build_prompt(x, ref):
            return f"""
                输出格式:
                只输出最终答案，不要添加多余解释。

                Question:
                {x['question']}

                Class: {x['class']}

                Reference answer:
                {ref['answer_reference']}
                """

        infer_requests = [
            InferRequest(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是一名煤矿科学与工程领域的专业问答助手。\n"
                            "请直接回答用户问题。\n"
                            "不要输出任何推理过程、思考步骤或分析内容。\n"
                            "不要输出 <think> 标签。\n"
                            "只输出最终答案，不要添加多余解释。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": build_prompt(x, ref_data[x["id"]])
                    }
                ]
            )
            for x in tqdm(dpo_data, desc="Preparing requests")
        ]
        resp_list = engine.infer(infer_requests, request_config)

        output_data = []
        for x, r in zip(dpo_data, resp_list):
            answer = r.choices[0].message.content

            # Remove leaked reasoning blocks
            answer = re.sub(
                r"<think>.*?</think>",
                "",
                answer,
                flags=re.DOTALL | re.IGNORECASE
            ).strip()

            output_data.append({
                **x,
                "answer2": answer
            })

        with open(PATH_DPO, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"Updated {len(output_data)} samples with answer2 -> {PATH_DPO}")

    if STAGE == 4:
        with open(PATH_DPO, "r", encoding="utf-8") as f:
            data = json.load(f)

        engine = TransformersEngine(Path_model, max_batch_size=BATCH_SIZE)
        request_config = RequestConfig(
            max_tokens=1024,
            temperature=0
        )

        # Load rubric template
        with open("PromptHealthBenchCn.txt", "r", encoding="utf-8") as f:
            RUBRIC_TEMPLATE = f.read()

        def build_prompt(x):
            return f"""{RUBRIC_TEMPLATE}

            Question:
            {x['question']}

            Reference Answer:
            {x['answer2']}
            """

        # -------------------------
        # Build requests
        # -------------------------
        infer_requests = [
            InferRequest(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是一名严格的JSON生成器。"
                            "只输出有效的rubric对象数组。"
                            "每个对象必须包含：title, description, weight。"
                            "不要输出任何解释。不要输出 <think> 标签。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": build_prompt(x)
                    }
                ]
            )
            for x in tqdm(data, desc="Preparing Stage 4")
        ]

        # -------------------------
        # Inference
        resp_list = engine.infer(infer_requests, request_config)

        updated_data = []
        for x, r in zip(data, resp_list):
            text = r.choices[0].message.content

            # remove thinking traces
            text = re.sub(r"<think>.*?</think>", "", text,
                        flags=re.DOTALL | re.IGNORECASE).strip()

            # -------------------------
            # parse JSON list
            # -------------------------
            try:
                raw_rubrics = json.loads(text)
            except Exception:
                raw_rubrics = []

            # -------------------------
            # normalize to required format
            # -------------------------
            rubrics = []
            for i, r_item in enumerate(raw_rubrics):
                if not isinstance(r_item, dict):
                    continue

                rubrics.append({
                    "rubric_id": i + 1,
                    "title": r_item.get("title", ""),
                    "description": r_item.get("description", ""),
                    "weight": r_item.get("weight", 0)
                })

            # -------------------------
            # append only new field
            # -------------------------
            new_item = dict(x)
            new_item["rubrics"] = rubrics

            updated_data.append(new_item)

        with open(PATH_DPO, "w", encoding="utf-8") as f:
            json.dump(updated_data, f, ensure_ascii=False, indent=2)

        print(f"Stage 4 done -> added rubric to {len(updated_data)} samples")    

    if STAGE == 5:
        with open(PATH_DPO, "r", encoding="utf-8") as f:
            data = json.load(f)

        # --------------------------------------------------
        # Find failed samples
        # --------------------------------------------------
        failed_ids = [
            x["id"]
            for x in data
            if not x.get("rubrics") or len(x["rubrics"]) == 0
        ]

        print(f"Found {len(failed_ids)} samples with empty rubrics.")
        print("Failed IDs:")
        print(failed_ids)

        if len(failed_ids) == 0:
            print("No retry needed.")
            exit()

        # --------------------------------------------------
        # Build retry set
        # --------------------------------------------------
        retry_data = [
            x
            for x in data
            if x["id"] in failed_ids
        ]

        engine = TransformersEngine(
            Path_model,
            max_batch_size=BATCH_SIZE
        )

        request_config = RequestConfig(
            max_tokens=1024,
            temperature=0.7,
            top_p=0.9
        )

        # --------------------------------------------------
        # Load template
        # --------------------------------------------------
        with open("PromptHealthBenchCn.txt", "r", encoding="utf-8") as f:
            RUBRIC_TEMPLATE = f.read()

        def build_prompt(x):
            return f"""{RUBRIC_TEMPLATE}

            Question:
            {x['question']}

            Reference Answer:
            {x['answer2']}
            """

        # --------------------------------------------------
        # Requests
        # --------------------------------------------------
        infer_requests = [
            InferRequest(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是一名严格的JSON生成器。"
                            "只输出有效的JSON数组。"
                            "每个对象必须包含："
                            "title, description, weight。"
                            "不要输出任何解释。"
                            "不要输出markdown。"
                            "No <think> tags."
                        ),
                    },
                    {
                        "role": "user",
                        "content": build_prompt(x)
                    }
                ]
            )
            for x in tqdm(
                retry_data,
                desc="Retrying empty rubrics"
            )
        ]

        resp_list = engine.infer(
            infer_requests,
            request_config
        )

        # --------------------------------------------------
        # Update samples
        # --------------------------------------------------
        id2sample = {
            x["id"]: x
            for x in data
        }

        for x, r in zip(retry_data, resp_list):

            text = r.choices[0].message.content

            text = re.sub(
                r"<think>.*?</think>",
                "",
                text,
                flags=re.DOTALL | re.IGNORECASE
            ).strip()

            try:
                raw_rubrics = json.loads(text)
            except Exception:
                raw_rubrics = []

            rubrics = []

            for i, item in enumerate(raw_rubrics):

                if not isinstance(item, dict):
                    continue

                rubrics.append(
                    {
                        "rubric_id": i + 1,
                        "title": item.get("title", ""),
                        "description": item.get("description", ""),
                        "weight": item.get("weight", 0)
                    }
                )

            id2sample[x["id"]]["rubrics"] = rubrics

        # --------------------------------------------------
        # Save
        # --------------------------------------------------
        updated_data = list(id2sample.values())

        with open(PATH_DPO, "w", encoding="utf-8") as f:
            json.dump(
                updated_data,
                f,
                ensure_ascii=False,
                indent=2
            )

        # --------------------------------------------------
        # Statistics
        # --------------------------------------------------
        remaining_failed_ids = [
            x["id"]
            for x in updated_data
            if not x.get("rubrics")
            or len(x["rubrics"]) == 0
        ]

        recovered = len(failed_ids) - len(remaining_failed_ids)

        print("\n" + "=" * 80)
        print("STAGE 5 SUMMARY")
        print("=" * 80)

        print(f"Originally empty : {len(failed_ids)}")
        print(f"Recovered        : {recovered}")
        print(f"Still empty      : {len(remaining_failed_ids)}")

        print("\nOriginal failed IDs:")
        print(failed_ids)

        print("\nRemaining failed IDs:")
        print(remaining_failed_ids)

        print("=" * 80)

    if STAGE == 6:
        with open(PATH_DPO, "r", encoding="utf-8") as f:
            data = json.load(f)

        # --------------------------------------------------
        # 1. Filter valid samples
        # --------------------------------------------------
        valid_data = []
        dropped_data = []

        for x in data:
            rubrics = x.get("rubrics")
            if (
                isinstance(rubrics, list)
                and len(rubrics) > 0
                and all(isinstance(r, dict) for r in rubrics)
            ):
                valid_data.append(x)
            else:
                dropped_data.append(x)

        # --------------------------------------------------
        # 2. Re-ID sequentially
        # --------------------------------------------------
        final_data = []
        for new_id, x in enumerate(valid_data):
            new_x = copy.deepcopy(x)
            new_x["id"] = new_id
            final_data.append(new_x)

        # --------------------------------------------------
        # 3. Overwrite original file
        # --------------------------------------------------
        with open(PATH_DPO, "w", encoding="utf-8") as f:
            json.dump(
                final_data,
                f,
                ensure_ascii=False,
                indent=2
            )

        # --------------------------------------------------
        # 4. Optional debug output (no separate file needed)
        # --------------------------------------------------
        print("=" * 80)
        print("STAGE 6 FINALIZATION")
        print("=" * 80)
        print(f"Original samples : {len(data)}")
        print(f"Valid samples    : {len(final_data)}")
        print(f"Dropped samples  : {len(dropped_data)}")
        print(f"File overwritten : {PATH_DPO}")
        print("=" * 80)