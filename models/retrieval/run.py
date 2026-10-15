from pathlib import Path
from getpass import getpass
from datetime import datetime
from dotenv import load_dotenv
import json
import os
import shutil
import sys
import time

import pandas as pd
import torch

load_dotenv()

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError(
        "Hãy chọn Runtime > Change runtime type > A100 GPU."
    )

GPU_NAME = torch.cuda.get_device_name(0)
print("GPU:", GPU_NAME)
if "A100" not in GPU_NAME.upper():
    print(
        "CẢNH BÁO: notebook được cấu hình cho A100, "
        f"nhưng GPU hiện tại là {GPU_NAME}."
    )

from alqac_qwen35_9b_entity_aware_private_test_retrieval import (
    ALQACClient,
    CalibratedSourceAwareEvidenceCrawler,
    EvidenceRetrievalConfig,
    ModelRuntime,
    SOURCE_TYPES,
    load_cases,
    run_batch,
)

RUN_NAME = "private_test_qwen35_9b_entity_aware_v2"
RESET_RUN = False

# =====================================================================
# ----------------------- Change path here ----------------------------
#======================================================================
RUN_DIR = (
    Path(os.getenv("RUN_DIR"))
    / RUN_NAME
)

if RESET_RUN and RUN_DIR.exists():
    print("Đang xóa run cũ:", RUN_DIR)
    shutil.rmtree(RUN_DIR)

RUN_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = Path(os.getenv("INPUT_PATH"))
CACHE_PATH = RUN_DIR / "api_cache.json"
RESULTS_JSONL_PATH = RUN_DIR / "batch_results.jsonl"
FAILURES_JSONL_PATH = RUN_DIR / "batch_failures.jsonl"

#==========================================================================

CONFIG = EvidenceRetrievalConfig(
    model_id="Qwen/Qwen3.5-4B",

    # Calibrated staged budget
    base_total_budget=24,
    intermediate_total_budget=32,
    soft_total_budget=40,
    late_total_budget=48,
    hard_total_budget=52,

    # Evidence map
    bootstrap_unique_target=8,
    evidence_refresh_every_new_chunks=4,
    evidence_refresh_every_queries=8,
    max_analysis_chunks=32,

    # Saturation / early stop
    recent_window=8,
    saturation_max_new_chunks=1,
    minimum_candidate_utility=0.14,
    high_priority_gap_threshold=0.72,

    # Tiny-case guardrail
    tiny_case_min_calls=12,
    tiny_case_max_unique_chunks=6,
    tiny_case_no_new_window=6,
    tiny_case_max_utility=0.18,

    # Active source retrieval
    source_probe_start_call=2,
    mandatory_probe_attempts_per_source=1,
    max_source_queries_per_type=4,
    max_source_followup_candidates_per_type=6,

    # Candidate management
    max_candidate_pool_size=800,
    max_seed_queries=28,
    max_chunk_queries=16,
    max_gap_queries=24,
    max_boundary_queries_per_chunk=4,
)

PRIVATE_CASES = load_cases(INPUT_PATH)
assert len(PRIVATE_CASES) > 0

manifest = {
    "run_name": RUN_NAME,
    "updated_at": datetime.now().isoformat(),
    "run_dir": str(RUN_DIR),
    "input_path": str(INPUT_PATH),
    "private_case_count": len(PRIVATE_CASES),
    "model_id": CONFIG.model_id,
    "dtype": "bfloat16",
    "quantization": None,
    "gpu": GPU_NAME,
    "enable_thinking": False,
    "budget_stages": list(CONFIG.budget_stages()),
    "active_sources": list(SOURCE_TYPES),
    "law_retrieval": False,
    "decision_model": False,
}
(RUN_DIR / "run_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

ALQAC_TOKEN = os.getenv("ALQAC_TOKEN")
if not ALQAC_TOKEN:
    raise ValueError("Bạn chưa nhập API key.")

print("RUN_DIR:", RUN_DIR)
print("Số private cases:", len(PRIVATE_CASES))
print("Budget:", CONFIG.budget_stages())


runtime = ModelRuntime(
    model_id=CONFIG.model_id,
    embedding_model_id=CONFIG.embedding_model_id,
    enable_thinking=False,
    deterministic_generation=True,
)
runtime.load()

client = ALQACClient(
    api_url=CONFIG.api_url,
    token=ALQAC_TOKEN,
    cache_path=CACHE_PATH,
    min_interval_seconds=CONFIG.min_interval_seconds,
    timeout_seconds=CONFIG.request_timeout_seconds,
    max_retries=CONFIG.max_retries_per_query,
)

crawler = CalibratedSourceAwareEvidenceCrawler(
    runtime=runtime,
    client=client,
    config=CONFIG,
    run_dir=RUN_DIR,
)

print("Qwen3.5-4B runtime sẵn sàng.")


smoke = runtime.generate_json(
    system=(
        "Bạn chỉ trả về JSON hợp lệ, không giải thích."
    ),
    user=(
        'Trả về đúng object {"status":"ok","model":"qwen3.5-4b"}.'
    ),
    max_new_tokens=80,
)
print(smoke)
assert smoke.get("status") == "ok"

def read_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows

done_rows = read_jsonl(RESULTS_JSONL_PATH)
done_ids = {str(item["case_id"]) for item in done_rows}
pending_ids = [
    item["case_id"]
    for item in PRIVATE_CASES
    if item["case_id"] not in done_ids
]

TOTAL_INPUT_CASES = len(PRIVATE_CASES)

print(
    "Hoàn thành:",
    len(done_ids),
    "/",
    TOTAL_INPUT_CASES,
)
print("Còn lại:", len(pending_ids))
print("5 case tiếp theo:", pending_ids[:5])

run_batch(
    crawler=crawler,
    input_path=INPUT_PATH,
    results_jsonl_path=RESULTS_JSONL_PATH,
    failures_jsonl_path=FAILURES_JSONL_PATH,
    max_cases=None,
)

done_rows = read_jsonl(RESULTS_JSONL_PATH)
failure_rows = read_jsonl(FAILURES_JSONL_PATH)

done_ids = {str(item["case_id"]) for item in done_rows}
missing_ids = [
    item["case_id"]
    for item in PRIVATE_CASES
    if item["case_id"] not in done_ids
]

print("Đã hoàn thành:", len(done_ids), f"/ {TOTAL_INPUT_CASES}")
print("Còn thiếu:", len(missing_ids))
print("Missing IDs:", missing_ids)
print("Số bản ghi failure:", len(failure_rows))

if done_rows:
    summary_rows = []
    for item in done_rows:
        metrics = item.get("metrics", {})
        summary_rows.append({
            "case_id": item.get("case_id"),
            "network_calls": metrics.get("network_calls"),
            "unique_chunks": metrics.get("unique_chunks"),
            "unique_chunks_per_call": metrics.get(
                "unique_chunks_per_call"
            ),
            "stop_reason": metrics.get("stop_reason"),
            "plaintiff": metrics.get(
                "source_statuses", {}
            ).get("plaintiff"),
            "defendant": metrics.get(
                "source_statuses", {}
            ).get("defendant"),
            "witness": metrics.get(
                "source_statuses", {}
            ).get("witness"),
            "prosecutor": metrics.get(
                "source_statuses", {}
            ).get("prosecutor"),
        })
    progress_df = pd.DataFrame(summary_rows)
    print(progress_df)
