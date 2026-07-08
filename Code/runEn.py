import json
from pathlib import Path
import shutil
from tqdm import tqdm
import re
import os
import copy
import numpy as np
import pandas as pd

os.environ["ASCEND_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"
from swift.infer_engine import TransformersEngine, RequestConfig, InferRequest

data = {
    "Chinese Category": [
        "不安全行为", "主运输", "井工开采", "供水", "供配电", 
        "储装运", "冲击地压防治", "地质", "安全培训", "安全监控", 
        "应急救援", "排水", "掘进与巷道工程", "提升", "支护技术与设备", 
        "智能矿山", "机电", "水害防治", "洗选", "火灾防治", 
        "热害治理", "煤化工", "煤基材料", "爆破", "环保与生态恢复", 
        "瓦斯防治", "矿工", "碳减排与循环利用", "粉尘防治", "职业健康", 
        "调度", "辅助运输", "通讯联络", "通风", "降温制冷", 
        "露天开采", "顶板", "其它"
    ],
    "English Category": [
        "Unsafe Behaviors", "Main Transportation", "Underground Mining", "Water Supply", "Power Supply and Distribution",
        "Storage, Loading, and Transport", "Rockburst Prevention and Control", "Geology", "Safety Training", "Safety Monitoring",
        "Emergency Rescue", "Mine Drainage", "Excavation and Roadway Engineering", "Mine Hoisting", "Support Technology and Equipment",
        "Intelligent Mine", "Mechanical and Electrical Engineering", "Mine Water Control", "Coal Preparation", "Mine Fire Prevention and Control",
        "Thermal Hazard Control", "Coal Chemical Engineering", "Coal-Based Materials", "Blasting", "Environmental and Ecological Restoration",
        "Methane Control in Mining", "Miners", "Carbon Reduction and Circular Utilization", "Mine Dust Control", "Occupational Health",
        "Dispatching", "Auxiliary Transportation", "Communication and Liaison", "Mine Ventilation", "Cooling and Refrigeration",
        "Surface Mining", "Ground Control", "Others"
    ]
}

# Create the DataFrame
df_class = pd.DataFrame(data)

BATCH_SIZE = 24
BATCH_SIZE_7 = 12
Path_model = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-32B"

# STAGE = 1   # Pure id, class, q and reference data
# STAGE = 2   # Qwen32B for generation (DPO) question-only
# STAGE = 3   # Qwen32B for DPO question-class-reference as input
# STAGE = 4   # Qwen32B for Rubric GPRO (given question and answer2)
# STAGE = 5   # regenerate rubrics for some empty (given question and answer2)
# STAGE = 6   # remove samples with empty rubrics and finalize dataset, re-id
# STAGE = 7   # rethink the class if good
STAGE = 8   # reprocess the mismatch class from stage7

PATH_in = Path("DataEn/Input.json")
PATH_DPO = Path("DataEn/CoalBench_EN.json")

for STAGE in [8]:
    if STAGE == 1:
        RUBRICS_PATH = Path("DataEn/Rubrics.json")
        BACKUP_PATH = Path("DataEn/Input.json")

        def clean_and_reindex():
            # backup first
            shutil.copy(RUBRICS_PATH, BACKUP_PATH)

            with open(RUBRICS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)

            new_data = []
            for new_id, item in enumerate(data, start=1):
                item.pop("Rubrics", None)   # remove field if exists
                item["id"] = new_id         # reindex from 1
                new_data.append(item)

            with open(RUBRICS_PATH, "w", encoding="utf-8") as f:
                json.dump(new_data, f, ensure_ascii=False, indent=2)

            print(f"Done: cleaned + reindexed {len(new_data)} items (1..N)")

        if __name__ == "__main__":
            clean_and_reindex()

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
                            "Answer the question directly. "
                            "Do not output any reasoning, thinking process, "
                            "analysis, or <think> tags. "
                            "Only provide the final answer."
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
                Output format:
                Return ONLY the final answer.

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
                            "Answer the question directly. "
                            "Do not output any reasoning, thinking process, "
                            "analysis, or <think> tags. "
                            "Only provide the final answer."
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
        with open("PromptHealthBenchEn.txt", "r", encoding="utf-8") as f:
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
                            "You are a strict JSON generator. "
                            "Output ONLY a valid JSON array of rubric objects. "
                            "Each object must contain: title, description, weight. "
                            "No explanations. No <think> tags."
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
        with open("PromptHealthBenchEn.txt", "r", encoding="utf-8") as f:
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
                            "You are a strict JSON generator. "
                            "Output ONLY a valid JSON array. "
                            "Each object must contain exactly "
                            "title, description, weight. "
                            "No explanations. "
                            "No markdown. "
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
    
    if STAGE == 7:
        # -----------------------------
        with open(PATH_DPO, "r", encoding="utf-8") as f:
            data = json.load(f)

        engine = TransformersEngine(
            Path_model,
            max_batch_size=BATCH_SIZE_7
        )

        request_config = RequestConfig(
            max_tokens=1024,
            temperature=0
        )
        # -----------------------------
        # Prompt (binary decision only)
        # -----------------------------
        def build_prompt(x):
            return f"""
                You are auditing a dataset in the domain of coal mining.

                Class: {x['class']}
                Question: {x['question']}

                Is the class a correct match for the question?

                Return ONLY valid JSON:
                {{"match": true}}
                or
                {{"match": false}}
                """.strip()

        # -----------------------------
        # Build requests
        # -----------------------------
        infer_requests = [
            InferRequest(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict JSON generator. "
                            "Output ONLY a valid JSON array. "
                            "Each object must contain exactly "
                            "title, description, weight. "
                            "No explanations. "
                            "No markdown. "
                            "No <think> tags."
                        ),
                    },
                    {
                        "role": "user",
                        "content": build_prompt(x)
                    }
                ]
            )
            for x in tqdm(data, desc="Building requests")
        ]

        resp_list = engine.infer(infer_requests, request_config)

        # Collect bad cases
        bad_cases = []
        bad_ids = []
        for x, r in tqdm(zip(data, resp_list), total=len(data)):
            text = r.choices[0].message.content

            text = re.sub(
                r"<think>.*?</think>",
                "",
                text,
                flags=re.DOTALL | re.IGNORECASE
            ).strip()

            try:
                result = json.loads(text)
                match = result.get("match", True)
            except Exception:
                match = False

            if not match:
                bad_ids.append(x["id"])
                bad_cases.append(x)

        # -----------------------------
        # Print IDs
        # -----------------------------
        print(f"Bad samples: {len(bad_ids)}")
        print(bad_ids)

        # -----------------------------
        # Save ONLY bad cases to CSV
        # -----------------------------
        df = pd.DataFrame(bad_cases)

        df.to_csv(
            "bad_cases.csv",
            index=False,
            encoding="utf-8-sig"
        )

    if STAGE == 8:
        with open(PATH_DPO, "r", encoding="utf-8") as f:
            data = json.load(f)

        # --------------------------------------------------
        # Remove IDs
        # --------------------------------------------------
        REMOVE_IDS = {
            12, 157, 160, 169, 294, 344, 345, 534, 616,
            635, 725, 731, 735, 749, 754, 755, 756, 757
        }

        data = [
            x for x in data
            if int(x["id"]) not in REMOVE_IDS
        ]

        print(f"After removal: {len(data)} samples")

        # --------------------------------------------------
        # Re-class mapping
        # --------------------------------------------------
        RECLASS = {
            7: "Underground Mining",
            11: "Underground Mining",
            12: "Intelligent Mine",
            15: "Underground Mining",
            37: "Others",
            69: "Others",
            71: "Surface Mining",
            74: "Others",
            75: "Auxiliary Transportation",
            82: "Surface Mining",
            95: "Auxiliary Transportation",
            110: "Coal Preparation",
            131: "Coal Preparation",
            136: "Rockburst Prevention and Control",
            137: "Ground Control",
            138: "Ground Control",
            142: "Coal Preparation",
            145: "Coal Preparation",
            146: "Coal Preparation",
            147: "Coal Preparation",
            153: "Coal Preparation",
            200: "Surface Mining",
            216: "Blasting",
            217: "Blasting",
            274: "Underground Mining",
            277: "Underground Mining",
            281: "Coal Chemical Engineering",
            315: "Surface Mining",
            319: "Surface Mining",
            322: "Environmental and Ecological Restoration",
            323: "Environmental and Ecological Restoration",
            324: "Environmental and Ecological Restoration",
            325: "Environmental and Ecological Restoration",
            327: "Environmental and Ecological Restoration",
            328: "Environmental and Ecological Restoration",
            329: "Environmental and Ecological Restoration",
            330: "Geology",
            331: "Geology",
            332: "Geology",
            333: "Geology",
            334: "Geology",
            335: "Geology",
            336: "Geology",
            337: "Geology",
            338: "Geology",
            339: "Geology",
            340: "Geology",
            341: "Geology",
            342: "Geology",
            343: "Geology",
            344: "Geology",
            345: "Geology",
            346: "Geology",
            347: "Geology",
            348: "Geology",
            349: "Geology",
            350: "Geology",
            351: "Geology",
            352: "Geology",
            353: "Geology",
            354: "Geology",
            355: "Geology",
            356: "Geology",
            357: "Geology",
            358: "Geology",
            359: "Geology",
            360: "Geology",
            409: "Surface Mining",
            530: "Surface Mining",
            531: "Auxiliary Transportation",
            551: "Surface Mining",
            559: "Auxiliary Transportation",
            560: "Auxiliary Transportation",
            713: "Coal Preparation",
            714: "Coal Preparation",
            715: "Coal Preparation",
            716: "Coal Preparation",
            717: "Coal Preparation",
        }

        # --------------------------------------------------
        # Update class
        # --------------------------------------------------
        updated = 0

        for x in data:
            idx = int(x["id"])

            if idx in RECLASS:
                x["class"] = RECLASS[idx]
                updated += 1

        print(f"Updated classes: {updated}")

        # --------------------------------------------------
        # Save
        # --------------------------------------------------
        with open(PATH_DPO, "w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2
            )

        print(f"Saved {len(data)} samples to {PATH_DPO}")