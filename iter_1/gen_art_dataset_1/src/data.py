#!/usr/bin/env python3
"""Load EntailmentBank Task 2 (train + dev) and standardize to exp_sel_data_out schema."""

import json
import re
import sys
from pathlib import Path

from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

WORKSPACE = Path(__file__).parent
DATASETS_DIR = WORKSPACE / "temp" / "datasets"
OUTPUT_PATH = WORKSPACE / "full_data_out.json"


def parse_context(context: str) -> dict[str, str]:
    """Parse 'sentN: text sentM: text ...' into {sentN: text}."""
    parts = re.split(r"(sent\d+):\s*", context)
    result: dict[str, str] = {}
    i = 1
    while i < len(parts) - 1:
        key = parts[i].strip()
        # text goes until the next sentN marker
        text = parts[i + 1].strip()
        result[key] = text
        i += 2
    return result


def parse_proof(proof: str) -> list[dict]:
    """Parse linearized proof into list of step dicts."""
    steps = []
    # Match intermediate steps: premises -> intN: conclusion_text
    for m in re.finditer(r"([\w\s&]+?)\s*->\s*(int\d+):\s*([^;]+)", proof):
        premises_raw, conc_id, conc_text = m.group(1), m.group(2), m.group(3)
        premises = [p.strip() for p in premises_raw.split("&") if p.strip()]
        steps.append({
            "step_id": conc_id.strip(),
            "premise_ids": premises,
            "conclusion_id": conc_id.strip(),
            "conclusion_text": conc_text.strip(),
        })
    # Match final step: premises -> hypothesis
    m = re.search(r"([\w\s&]+?)\s*->\s*hypothesis\s*;?", proof)
    if m:
        premises_raw = m.group(1)
        premises = [p.strip() for p in premises_raw.split("&") if p.strip()]
        steps.append({
            "step_id": "hypothesis",
            "premise_ids": premises,
            "conclusion_id": "hypothesis",
            "conclusion_text": "",
        })
    return steps


def make_input(row: dict) -> str:
    """Build the input string: question + context sentences."""
    ctx = row.get("context", "")
    q = row.get("question", "")
    return f"Question: {q}\nContext: {ctx}"


def make_output(row: dict) -> str:
    """Build the output string: hypothesis + proof."""
    hyp = row.get("hypothesis", "")
    proof = row.get("proof", "")
    return f"Hypothesis: {hyp}\nProof: {proof}"


@logger.catch(reraise=True)
def main() -> None:
    Path("logs").mkdir(exist_ok=True)

    splits = {
        "train": DATASETS_DIR / "task2_train.jsonl",
        "dev": DATASETS_DIR / "task2_dev.jsonl",
    }

    examples = []
    for split_name, path in splits.items():
        logger.info(f"Loading {split_name} from {path}")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        logger.info(f"  {len(rows)} rows in {split_name}")

        for idx, row in enumerate(rows):
            ctx_sentences = parse_context(row.get("context", ""))
            proof_steps = parse_proof(row.get("proof", ""))
            meta = row.get("meta", {})

            example = {
                "input": make_input(row),
                "output": make_output(row),
                "metadata_example_id": row.get("id", ""),
                "metadata_split": split_name,
                "metadata_hypothesis": row.get("hypothesis", ""),
                "metadata_context_json": json.dumps(ctx_sentences),
                "metadata_proof_raw": row.get("proof", ""),
                "metadata_proof_steps_json": json.dumps(proof_steps),
                "metadata_depth_of_proof": row.get("depth_of_proof", 0),
                "metadata_length_of_proof": row.get("length_of_proof", 0),
                "metadata_num_context_sentences": len(ctx_sentences),
                "metadata_distractors_json": json.dumps(meta.get("distractors", [])),
                "metadata_row_index": idx,
                "metadata_task_type": "entailment_tree",
            }
            examples.append(example)

    logger.info(f"Total examples: {len(examples)}")

    output = {
        "datasets": [
            {
                "dataset": "entailment_bank_task2",
                "examples": examples,
            }
        ]
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved {len(examples)} examples to {OUTPUT_PATH}")
    logger.info(f"File size: {OUTPUT_PATH.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
