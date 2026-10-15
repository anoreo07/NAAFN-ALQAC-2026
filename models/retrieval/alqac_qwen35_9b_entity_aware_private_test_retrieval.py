from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import requests
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForMultimodalLM, AutoProcessor


FRESH_ONLY_PIPELINE = True

LABELS_FORBIDDEN = {
    "A_WIN", "PARTIAL_A_WIN", "PARTIAL_B_WIN", "B_WIN"
}


SOURCE_TYPES = ("plaintiff", "defendant", "witness", "prosecutor")

SOURCE_LABELS = {
    "plaintiff": "lời trình bày của nguyên đơn",
    "defendant": "lời trình bày của bị đơn",
    "witness": "lời khai của người làm chứng",
    "prosecutor": "ý kiến của Viện kiểm sát",
}

SOURCE_MARKERS = {
    "plaintiff": [
        "nguyên đơn trình bày",
        "nguyên đơn cho rằng",
        "nguyên đơn yêu cầu",
        "theo đơn khởi kiện",
        "phía nguyên đơn",
        "lời trình bày của nguyên đơn",
    ],
    "defendant": [
        "bị đơn trình bày",
        "bị đơn cho rằng",
        "bị đơn không đồng ý",
        "phía bị đơn",
        "lời trình bày của bị đơn",
        "ý kiến của bị đơn",
    ],
    "witness": [
        "người làm chứng",
        "lời khai của người làm chứng",
        "nhân chứng trình bày",
        "người chứng kiến",
        "người làm chứng xác nhận",
    ],
    "prosecutor": [
        "viện kiểm sát",
        "kiểm sát viên",
        "đại diện viện kiểm sát",
        "viện kiểm sát đề nghị",
        "phát biểu quan điểm",
    ],
}

SOURCE_STATUS_VALUES = {
    "NOT_FOUND",
    "MENTION_ONLY",
    "PARTIAL",
    "SUBSTANTIVE",
}

SOURCE_ATTRIBUTION_MODES = {
    "DIRECT_STATEMENT",
    "ATTRIBUTED_STATEMENT",
    "MENTION_ONLY",
}

SOURCE_ALIAS_KEYS = {
    "plaintiff": "plaintiff_aliases",
    "defendant": "defendant_aliases",
    "witness": "witness_aliases",
    "prosecutor": "prosecutor_aliases",
}

SOURCE_MODE_RANK = {
    "MENTION_ONLY": 0,
    "ATTRIBUTED_STATEMENT": 1,
    "DIRECT_STATEMENT": 2,
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EvidenceRetrievalConfig:
    api_url: str = "https://alqac-api.ngrok.pro/retrieve"
    model_id: str = "Qwen/Qwen3.5-9B"
    embedding_model_id: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    min_interval_seconds: float = 5.5
    request_timeout_seconds: int = 30
    max_retries_per_query: int = 3

    # Calibrated from public case_fact lengths when split into consecutive,
    # non-overlapping chunks of at most 100 words:
    # median ~25.5, P75=32, P90=40, maximum=52 for regular cases.
    base_total_budget: int = 24
    intermediate_total_budget: int = 32
    soft_total_budget: int = 40
    late_total_budget: int = 48
    hard_total_budget: int = 52

    bootstrap_unique_target: int = 8
    evidence_refresh_every_new_chunks: int = 4
    evidence_refresh_every_queries: int = 8
    max_analysis_chunks: int = 32

    recent_window: int = 8
    saturation_max_new_chunks: int = 1
    high_priority_gap_threshold: float = 0.72
    minimum_candidate_utility: float = 0.14

    # Early stop for unusually short/truncated case_fact distributions.
    tiny_case_min_calls: int = 12
    tiny_case_max_unique_chunks: int = 6
    tiny_case_no_new_window: int = 6
    tiny_case_max_utility: float = 0.18

    max_candidate_pool_size: int = 800
    max_seed_queries: int = 28
    max_chunk_queries: int = 16
    max_gap_queries: int = 24
    max_boundary_queries_per_chunk: int = 4
    max_mutations_per_branch: int = 1

    # Active source retrieval. The crawler deliberately probes these four
    # evidence sources, then lets the LLM generate source-conditioned follow-ups.
    source_probe_start_call: int = 2
    mandatory_probe_attempts_per_source: int = 1
    max_source_queries_per_type: int = 4
    max_source_followup_candidates_per_type: int = 6
    source_substantive_priority: float = 0.20
    source_missing_priority: float = 0.90
    source_partial_priority: float = 0.78
    source_mention_priority: float = 0.84
    min_source_attribution_confidence: float = 0.55
    max_entity_source_candidates_per_type: int = 8

    hard_seen_coverage_reject: float = 0.96
    hard_seen_semantic_reject: float = 0.985
    used_query_semantic_reject: float = 0.94

    # Scheduler balances two retrieval-only tracks:
    # discovery = maximize unique chunks
    # gap = fill factual evidence gaps
    discovery_track_weight: float = 0.55
    gap_track_weight: float = 0.35
    exploration_weight: float = 0.10

    # A short metadata chunk is still preserved, but gets lower learning reward.
    metadata_reward: float = 0.25
    short_chunk_reward: float = 0.45

    def __post_init__(self) -> None:
        budgets = [
            self.base_total_budget,
            self.intermediate_total_budget,
            self.soft_total_budget,
            self.late_total_budget,
            self.hard_total_budget,
        ]
        if budgets != sorted(budgets) or len(set(budgets)) != len(budgets):
            raise ValueError(
                "Budget phải tăng nghiêm ngặt: base < intermediate < soft < late < hard."
            )
        if self.recent_window < 4:
            raise ValueError("recent_window phải >= 4.")
        if self.max_source_queries_per_type < self.mandatory_probe_attempts_per_source:
            raise ValueError(
                "max_source_queries_per_type phải >= mandatory_probe_attempts_per_source."
            )

    def budget_stages(self) -> tuple[int, ...]:
        return (
            self.base_total_budget,
            self.intermediate_total_budget,
            self.soft_total_budget,
            self.late_total_budget,
            self.hard_total_budget,
        )


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

VIETNAMESE_STOPWORDS = {
    "và", "là", "của", "có", "cho", "được", "với", "tại", "trong", "đã",
    "thì", "mà", "nhưng", "do", "nên", "về", "theo", "để", "khi", "này",
    "đó", "các", "một", "những", "người", "ông", "bà", "anh", "chị", "tôi",
    "họ", "ra", "vào", "lại", "không", "phải", "bị", "cũng", "như", "sau",
    "trước", "từ", "đến", "trên", "dưới", "qua", "đang", "sẽ", "nếu",
    "vụ", "việc", "nội", "dung",
}

PREDICTION_MARKERS = [
    "agent cần dự đoán",
    "agent dự đoán",
    "hãy dự đoán",
    "cần dự đoán",
    "dự đoán ai",
    "ai được chấp nhận yêu cầu",
    "ai thắng kiện",
    "nguyên đơn hay bị đơn thắng",
    "bị đơn hay nguyên đơn thắng",
]

QUESTION_PATTERNS = [
    r"\bai\b",
    r"\bở đâu\b",
    r"\bkhi nào\b",
    r"\bnhư thế nào\b",
    r"\btại sao\b",
    r"\bcó phải\b",
]

BAD_ANCHOR_ENDINGS = {
    "và", "của", "khi", "chị", "anh", "tại", "cho", "đi", "một",
    "những", "các", "theo", "với", "trong",
}


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text)).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def content_tokens(text: str) -> list[str]:
    return [
        token
        for token in normalize_text(text).split()
        if token not in VIETNAMESE_STOPWORDS and len(token) > 1
    ]


def clean_case_query(case_query: str) -> str:
    if not isinstance(case_query, str) or not case_query.strip():
        raise ValueError("case_query phải là chuỗi không rỗng.")

    text = unicodedata.normalize("NFC", case_query)
    text = " ".join(text.split())
    lowered = text.lower()

    positions = [
        lowered.find(marker)
        for marker in PREDICTION_MARKERS
        if lowered.find(marker) >= 0
    ]
    if positions:
        text = text[:min(positions)]

    return text.strip(" .?;:")


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?;:])\s+", text)
    return [part.strip(" .?;:") for part in parts if part.strip(" .?;:")]


def truncate_words(text: str, max_words: int) -> str:
    return " ".join(str(text).split()[:max_words])


def has_question_form(query: str) -> bool:
    normalized = normalize_text(query)
    return any(re.search(pattern, normalized) for pattern in QUESTION_PATTERNS)


def sanitize_query(query: str, max_words: int = 13) -> str:
    query = " ".join(str(query).split()).strip(" .?;:,-")
    query = truncate_words(query, max_words)
    return query


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp.replace(path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(value, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clip01(value: Any, default: float = 0.0) -> float:
    return max(0.0, min(1.0, safe_float(value, default)))


def empty_source_coverage() -> dict[str, dict[str, Any]]:
    return {
        source_type: {
            "source_type": source_type,
            "label": SOURCE_LABELS[source_type],
            "status": "NOT_FOUND",
            "evidence_ids": [],
            "substantive_points": [],
            "missing_aspects": [],
            "follow_up_questions": [],
            "query_hooks": [],
            "priority": 0.90,
            "probe_recommended": True,
        }
        for source_type in SOURCE_TYPES
    }


def detect_source_types_in_text(text: str) -> list[str]:
    normalized = normalize_text(text)
    detected = []
    for source_type, markers in SOURCE_MARKERS.items():
        if any(normalize_text(marker) in normalized for marker in markers):
            detected.append(source_type)
    return detected


def source_status_priority(
    status: str,
    config: EvidenceRetrievalConfig,
) -> float:
    if status == "SUBSTANTIVE":
        return config.source_substantive_priority
    if status == "PARTIAL":
        return config.source_partial_priority
    if status == "MENTION_ONLY":
        return config.source_mention_priority
    return config.source_missing_priority



def source_alias_key(source_type: str) -> str:
    return SOURCE_ALIAS_KEYS[source_type]


def source_aliases(
    roles: dict[str, list[str]],
    source_type: str,
) -> list[str]:
    return list(roles.get(source_alias_key(source_type), []))


def valid_source_alias(
    alias: Any,
    chunk_text: str,
) -> Optional[str]:
    value = " ".join(str(alias).split()).strip(" .?;:,-")
    normalized = normalize_text(value)
    if not normalized:
        return None
    if not (1 <= len(value.split()) <= 7):
        return None
    if normalized not in normalize_text(chunk_text):
        return None

    generic = {
        "nguyên đơn",
        "bị đơn",
        "người làm chứng",
        "nhân chứng",
        "viện kiểm sát",
        "kiểm sát viên",
        "đại diện viện kiểm sát",
        "người có quyền lợi nghĩa vụ liên quan",
    }
    if normalized in generic:
        return None
    return value


def merge_attributions(
    attributions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    best: dict[tuple[str, str], dict[str, Any]] = {}

    for item in attributions:
        source_type = str(item.get("source_type", "")).lower()
        mode = str(item.get("mode", "MENTION_ONLY")).upper()
        span = " ".join(str(item.get("evidence_span", "")).split())
        if source_type not in SOURCE_TYPES:
            continue
        if mode not in SOURCE_ATTRIBUTION_MODES:
            continue

        key = (source_type, normalize_text(span))
        previous = best.get(key)
        if (
            previous is None
            or SOURCE_MODE_RANK[mode]
            > SOURCE_MODE_RANK[previous["mode"]]
            or safe_float(item.get("confidence", 0.0))
            > safe_float(previous.get("confidence", 0.0))
        ):
            best[key] = {
                "source_type": source_type,
                "mode": mode,
                "speaker_aliases": list(dict.fromkeys(
                    str(value).strip()
                    for value in item.get("speaker_aliases", [])
                    if str(value).strip()
                )),
                "evidence_span": span,
                "confidence": clip01(item.get("confidence", 0.0)),
            }

    return list(best.values())


def deterministic_source_attributions(
    chunk_text: str,
    roles: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """
    Conservative fallback.

    It labels the content of the returned chunk, never the family/query that
    retrieved it. Named aliases are useful because judgment sections frequently
    begin with 'chị X trình bày' rather than a literal role header.
    """
    normalized = normalize_text(chunk_text)
    output: list[dict[str, Any]] = []

    direct_verbs = (
        "trình bày|khai|cho rằng|yêu cầu|xác nhận|thừa nhận|"
        "không đồng ý|đồng ý|đề nghị|phát biểu"
    )

    explicit_patterns = {
        "plaintiff": [
            r"\bnguyên đơn\b.{0,80}\b(" + direct_verbs + r")\b",
            r"\btheo đơn khởi kiện\b",
            r"\btại đơn khởi kiện\b",
        ],
        "defendant": [
            r"\bbị đơn\b.{0,80}\b(" + direct_verbs + r")\b",
            r"\btại bản tự khai\b.{0,80}\bbị đơn\b",
        ],
        "witness": [
            r"\bngười làm chứng\b.{0,80}\b(" + direct_verbs + r")\b",
            r"\bnhân chứng\b.{0,80}\b(" + direct_verbs + r")\b",
            r"\bngười chứng kiến\b.{0,80}\b(" + direct_verbs + r")\b",
        ],
        "prosecutor": [
            r"\bđại diện viện kiểm sát\b.{0,100}\b("
            + direct_verbs + r")\b",
            r"\bkiểm sát viên\b.{0,100}\b("
            + direct_verbs + r")\b",
            r"\bý kiến (?:của )?viện kiểm sát\b",
        ],
    }

    for source_type, patterns in explicit_patterns.items():
        for pattern in patterns:
            match = re.search(pattern, normalized, flags=re.S)
            if match:
                output.append({
                    "source_type": source_type,
                    "mode": "DIRECT_STATEMENT",
                    "speaker_aliases": [],
                    "evidence_span": match.group(0)[:180],
                    "confidence": 0.88,
                })
                break

    for source_type in SOURCE_TYPES:
        for alias in source_aliases(roles, source_type):
            alias_normalized = normalize_text(alias)
            if not alias_normalized or alias_normalized not in normalized:
                continue

            alias_pattern = re.escape(alias_normalized)
            direct = re.search(
                rf"\b{alias_pattern}\b.{{0,80}}\b({direct_verbs})\b",
                normalized,
                flags=re.S,
            )
            attributed = re.search(
                rf"\b(?:theo lời khai của|theo trình bày của|"
                rf"theo ý kiến của|lời khai của)\s+{alias_pattern}\b",
                normalized,
                flags=re.S,
            )

            if direct:
                mode = "DIRECT_STATEMENT"
                confidence = 0.84
            elif attributed:
                mode = "ATTRIBUTED_STATEMENT"
                confidence = 0.78
            else:
                mode = "MENTION_ONLY"
                confidence = 0.60

            output.append({
                "source_type": source_type,
                "mode": mode,
                "speaker_aliases": [alias],
                "evidence_span": alias,
                "confidence": confidence,
            })

    return merge_attributions(output)


def observed_source_types(
    attributions: list[dict[str, Any]],
) -> list[str]:
    return sorted({
        item["source_type"]
        for item in attributions
        if item.get("mode") in {
            "DIRECT_STATEMENT",
            "ATTRIBUTED_STATEMENT",
        }
    })


def mentioned_source_types(
    attributions: list[dict[str, Any]],
) -> list[str]:
    return sorted({
        item["source_type"]
        for item in attributions
    })


# ---------------------------------------------------------------------------
# Local Qwen3.5-9B BF16 model and embedding model
# ---------------------------------------------------------------------------

class ModelRuntime:
    """
    Text-only inference wrapper for Qwen3.5-9B.

    Qwen3.5-9B is distributed as a causal language model with a vision encoder,
    so the official Transformers entry points are AutoProcessor and
    AutoModelForMultimodalLM. This crawler only supplies text messages.

    The model runs in BF16 on an A100. No 4-bit or 8-bit quantization is used.
    Thinking is disabled because this stage needs short, schema-constrained JSON
    rather than long reasoning traces.
    """

    def __init__(
        self,
        model_id: str,
        embedding_model_id: str,
        *,
        enable_thinking: bool = False,
        deterministic_generation: bool = True,
    ):
        self.model_id = model_id
        self.embedding_model_id = embedding_model_id
        self.enable_thinking = enable_thinking
        self.deterministic_generation = deterministic_generation

        self.processor: Optional[Any] = None
        self.llm: Optional[Any] = None
        self.embedder: Optional[SentenceTransformer] = None
        self.embedding_cache: dict[str, np.ndarray] = {}

    def load(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Hãy chọn Runtime > Change runtime type > A100 GPU."
            )

        gpu_name = torch.cuda.get_device_name(0)
        print("Loading", self.model_id, "on", gpu_name)

        # A100 supports BF16 natively. TF32 improves matrix multiplication
        # throughput without changing the stored BF16 model weights.
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=False,
        )

        self.llm = AutoModelForMultimodalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        self.llm.eval()

        self.embedder = SentenceTransformer(
            self.embedding_model_id,
            device="cuda",
        )

        allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
        print(
            f"Model loaded. CUDA allocated={allocated_gb:.2f} GiB, "
            f"reserved={reserved_gb:.2f} GiB"
        )

    @staticmethod
    def _as_qwen_messages(
        messages: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role", "user"))
            content = str(message.get("content", ""))
            converted.append({
                "role": role,
                "content": [
                    {
                        "type": "text",
                        "text": content,
                    }
                ],
            })
        return converted

    def _prepare_inputs(
        self,
        messages: list[dict[str, str]],
    ) -> dict[str, torch.Tensor]:
        if self.processor is None or self.llm is None:
            raise RuntimeError("ModelRuntime chưa được load().")

        qwen_messages = self._as_qwen_messages(messages)
        common_kwargs = {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
        }

        # The current Qwen3.5 chat template accepts enable_thinking directly.
        # The fallback keeps the notebook robust to minor Transformers API
        # changes while still disabling thinking.
        try:
            inputs = self.processor.apply_chat_template(
                qwen_messages,
                enable_thinking=self.enable_thinking,
                **common_kwargs,
            )
        except TypeError:
            inputs = self.processor.apply_chat_template(
                qwen_messages,
                chat_template_kwargs={
                    "enable_thinking": self.enable_thinking,
                },
                **common_kwargs,
            )

        return inputs.to(self.llm.device)

    def generate_text(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 900,
    ) -> str:
        if self.processor is None or self.llm is None:
            raise RuntimeError("ModelRuntime chưa được load().")

        inputs = self._prepare_inputs(messages)
        input_length = inputs["input_ids"].shape[-1]

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.processor.tokenizer.eos_token_id,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
            "use_cache": True,
        }

        if self.deterministic_generation:
            # Deterministic decoding is more stable for strict JSON schemas.
            generation_kwargs["do_sample"] = False
        else:
            # Qwen3.5 non-thinking parameters recommended by the model card,
            # excluding presence_penalty because Transformers generate() does
            # not expose it consistently across versions.
            generation_kwargs.update({
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
            })

        with torch.inference_mode():
            generated = self.llm.generate(
                **inputs,
                **generation_kwargs,
            )

        new_tokens = generated[0, input_length:]
        return self.processor.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()

    @staticmethod
    def extract_json_object(text: str) -> dict[str, Any]:
        cleaned = text.strip()
        cleaned = re.sub(
            r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE
        )
        cleaned = re.sub(r"\s*```$", "", cleaned)

        decoder = json.JSONDecoder()
        for index, char in enumerate(cleaned):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(cleaned[index:])
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError:
                continue

        raise ValueError(f"Không parse được JSON:\n{cleaned[:1600]}")

    def generate_json(
        self,
        system: str,
        user: str,
        max_new_tokens: int = 1000,
    ) -> dict[str, Any]:
        raw = self.generate_text(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_new_tokens=max_new_tokens,
        )
        parsed = self.extract_json_object(raw)

        # This stage must never output a competition label.
        serialized = json.dumps(parsed, ensure_ascii=False)
        for label in LABELS_FORBIDDEN:
            if label in serialized:
                raise ValueError(
                    "Evidence retriever không được sinh nhãn quyết định."
                )
        return parsed

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.embedder is None:
            raise RuntimeError("ModelRuntime chưa được load().")

        missing = [text for text in texts if text not in self.embedding_cache]
        if missing:
            vectors = self.embedder.encode(
                missing,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=32,
            )
            for text, vector in zip(missing, vectors):
                self.embedding_cache[text] = vector.astype(np.float32)

        return np.stack([self.embedding_cache[text] for text in texts])

    def cosine(self, left: str, right: str) -> float:
        vectors = self.encode([left, right])
        return float(np.dot(vectors[0], vectors[1]))


# ---------------------------------------------------------------------------
# API, cache and rate limiting
# ---------------------------------------------------------------------------

@dataclass
class RetrievalHit:
    chunk_id: str
    text: str
    score: float


@dataclass
class RetrievalCall:
    success: bool
    status_code: Optional[int]
    hit: Optional[RetrievalHit]
    error: Optional[str]
    attempts: int
    from_cache: bool = False


class GlobalRateLimiter:
    def __init__(self, min_interval_seconds: float):
        self.min_interval_seconds = min_interval_seconds
        self.last_request_at: Optional[float] = None

    def wait(self) -> None:
        if self.last_request_at is not None:
            elapsed = time.monotonic() - self.last_request_at
            remaining = self.min_interval_seconds - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self.last_request_at = time.monotonic()


class JsonCache:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data: dict[str, Any] = {}

        if path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                broken = path.with_suffix(".broken.json")
                shutil.copy2(path, broken)
                print("Cache lỗi, đã backup:", broken)

    @staticmethod
    def key(case_id: str, query: str) -> str:
        raw = f"{case_id}|{normalize_text(query)}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, case_id: str, query: str) -> Optional[dict[str, Any]]:
        return self.data.get(self.key(case_id, query))

    def set(self, case_id: str, query: str, value: dict[str, Any]) -> None:
        self.data[self.key(case_id, query)] = value
        atomic_write_json(self.path, self.data)


class ALQACClient:
    def __init__(
        self,
        api_url: str,
        token: str,
        cache_path: Path,
        min_interval_seconds: float,
        timeout_seconds: int,
        max_retries: int,
    ):
        self.api_url = api_url
        self.token = token
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.rate_limiter = GlobalRateLimiter(min_interval_seconds)
        self.cache = JsonCache(cache_path)
        self.session = requests.Session()
        self.network_call_count = 0

    @staticmethod
    def parse_hit(payload: Any) -> RetrievalHit:
        if not isinstance(payload, dict):
            raise ValueError("Response không phải object.")

        results = payload.get("results")
        if not isinstance(results, list) or not results:
            raise ValueError("results rỗng hoặc sai schema.")

        top = results[0]
        chunk_id = top.get("chunk_id")
        text = top.get("text")
        score = top.get("score")

        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError("Thiếu chunk_id.")
        if not isinstance(text, str):
            raise ValueError("Thiếu text.")
        if not isinstance(score, (int, float)):
            raise ValueError("score không phải số.")

        return RetrievalHit(
            chunk_id=chunk_id,
            text=text,
            score=float(score),
        )

    def retrieve(
        self,
        case_id: str,
        query: str,
        max_network_attempts: int,
    ) -> RetrievalCall:
        cached = self.cache.get(case_id, query)
        if cached:
            hit_data = cached.get("hit")
            hit = RetrievalHit(**hit_data) if hit_data else None
            return RetrievalCall(
                success=bool(cached.get("success")),
                status_code=cached.get("status_code"),
                hit=hit,
                error=cached.get("error"),
                attempts=0,
                from_cache=True,
            )

        attempts_allowed = min(
            self.max_retries,
            max(0, max_network_attempts),
        )
        if attempts_allowed <= 0:
            return RetrievalCall(
                False, None, None, "Hết network budget.", 0, False
            )

        headers = {
            "X-API-Key": self.token,
            "Content-Type": "application/json",
        }
        body = {"query": query, "case_id": case_id}

        last_status: Optional[int] = None
        last_error: Optional[str] = None

        for attempt in range(1, attempts_allowed + 1):
            self.rate_limiter.wait()
            self.network_call_count += 1

            try:
                response = self.session.post(
                    self.api_url,
                    headers=headers,
                    json=body,
                    timeout=self.timeout_seconds,
                )
                last_status = response.status_code
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < attempts_allowed:
                    time.sleep(min(5 * 2 ** (attempt - 1), 20))
                    continue
                break

            preview = response.text[:500]

            if response.status_code == 200:
                try:
                    hit = self.parse_hit(response.json())
                except (ValueError, json.JSONDecodeError) as exc:
                    return RetrievalCall(
                        False,
                        200,
                        None,
                        f"Invalid response: {exc}",
                        attempt,
                        False,
                    )

                cached_value = {
                    "success": True,
                    "status_code": 200,
                    "hit": asdict(hit),
                    "error": None,
                }
                self.cache.set(case_id, query, cached_value)
                return RetrievalCall(
                    True, 200, hit, None, attempt, False
                )

            if response.status_code == 403:
                raise RuntimeError("API trả 403: " + preview)

            if response.status_code == 422:
                return RetrievalCall(
                    False,
                    422,
                    None,
                    "Malformed request: " + preview,
                    attempt,
                    False,
                )

            if response.status_code == 429:
                last_error = "Rate limit: " + preview
                if attempt < attempts_allowed:
                    time.sleep(min(6 * attempt, 20))
                    continue
                break

            if 500 <= response.status_code <= 504:
                last_error = (
                    f"Server error {response.status_code}: {preview}"
                )
                if attempt < attempts_allowed:
                    time.sleep(min(5 * 2 ** (attempt - 1), 20))
                    continue
                break

            return RetrievalCall(
                False,
                response.status_code,
                None,
                f"Unexpected HTTP {response.status_code}: {preview}",
                attempt,
                False,
            )

        return RetrievalCall(
            False,
            last_status,
            None,
            last_error or "Unknown API error",
            attempts_allowed,
            False,
        )


# ---------------------------------------------------------------------------
# Query pool and anti-seen scoring
# ---------------------------------------------------------------------------

@dataclass
class QueryCandidate:
    query: str
    family: str
    track: str = "discovery"  # discovery | gap | exploration
    base_priority: float = 0.5
    source_chunk_id: Optional[str] = None
    source_topic_id: Optional[str] = None
    source_gap_id: Optional[str] = None
    parent_query: Optional[str] = None
    rationale: str = ""
    mutation_count: int = 0
    generation: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FamilyStats:
    calls: int = 0
    new_chunks: int = 0
    duplicates: int = 0
    failures: int = 0
    learning_reward_sum: float = 0.0

    def new_chunk_rate(self) -> float:
        return self.new_chunks / self.calls if self.calls else 0.0

    def mean_reward(self) -> float:
        return (
            self.learning_reward_sum / self.calls
            if self.calls else 0.0
        )


class CandidatePool:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.items: list[QueryCandidate] = []
        self.normalized_queries: set[str] = set()

    def add(self, candidate: QueryCandidate) -> bool:
        candidate.query = sanitize_query(candidate.query)
        normalized = normalize_text(candidate.query)

        if len(content_tokens(candidate.query)) < 2:
            return False
        if has_question_form(candidate.query):
            return False
        if not normalized or normalized in self.normalized_queries:
            return False

        self.items.append(candidate)
        self.normalized_queries.add(normalized)

        if len(self.items) > self.max_size:
            self.items.sort(
                key=lambda item: item.base_priority,
                reverse=True,
            )
            removed = self.items[self.max_size:]
            self.items = self.items[:self.max_size]
            for item in removed:
                self.normalized_queries.discard(
                    normalize_text(item.query)
                )
        return True

    def extend(self, candidates: Iterable[QueryCandidate]) -> int:
        return sum(self.add(candidate) for candidate in candidates)

    def remove(self, candidate: QueryCandidate) -> None:
        self.items.remove(candidate)
        self.normalized_queries.discard(normalize_text(candidate.query))

    def __len__(self) -> int:
        return len(self.items)


class SeenIndex:
    def __init__(self, runtime: ModelRuntime):
        self.runtime = runtime
        self.chunk_ids: list[str] = []
        self.texts: list[str] = []
        self.token_sets: list[set[str]] = []
        self.union_tokens: set[str] = set()

    def rebuild(self, archive: dict[str, dict[str, Any]]) -> None:
        self.chunk_ids = list(archive.keys())
        self.texts = [archive[cid]["text"] for cid in self.chunk_ids]
        self.token_sets = [
            set(content_tokens(text)) for text in self.texts
        ]
        self.union_tokens = (
            set().union(*self.token_sets)
            if self.token_sets else set()
        )

    def positive_idf_coverage(self, query: str) -> float:
        query_tokens = set(content_tokens(query))
        if not query_tokens or not self.token_sets:
            return 0.0

        n_docs = len(self.token_sets)
        idf: dict[str, float] = {}
        for token in query_tokens:
            df = sum(token in doc for doc in self.token_sets)
            idf[token] = math.log(
                1.0 + (n_docs - df + 0.5) / (df + 0.5)
            )

        denominator = sum(idf.values()) or 1.0
        return max(
            sum(
                weight
                for token, weight in idf.items()
                if token in document
            ) / denominator
            for document in self.token_sets
        )

    def metrics(self, query: str) -> dict[str, float]:
        if not self.texts:
            return {
                "seen_coverage": 0.0,
                "seen_semantic": 0.0,
                "unseen_term_ratio": 1.0,
                "seen_risk": 0.0,
            }

        query_tokens = content_tokens(query)
        if not query_tokens:
            return {
                "seen_coverage": 1.0,
                "seen_semantic": 1.0,
                "unseen_term_ratio": 0.0,
                "seen_risk": 1.0,
            }

        coverage = self.positive_idf_coverage(query)

        query_vector = self.runtime.encode([query])[0]
        chunk_vectors = self.runtime.encode(self.texts)
        semantic = float(np.max(chunk_vectors @ query_vector))

        unseen = [
            token for token in query_tokens
            if token not in self.union_tokens
        ]
        unseen_ratio = len(unseen) / len(query_tokens)

        seen_risk = 0.55 * coverage + 0.45 * max(0.0, semantic)

        return {
            "seen_coverage": coverage,
            "seen_semantic": semantic,
            "unseen_term_ratio": unseen_ratio,
            "seen_risk": seen_risk,
        }


def max_similarity_to_used(
    runtime: ModelRuntime,
    query: str,
    used_queries: list[str],
) -> float:
    if not used_queries:
        return 0.0

    query_vector = runtime.encode([query])[0]
    used_vectors = runtime.encode(used_queries[-80:])
    return float(np.max(used_vectors @ query_vector))


def family_ucb(
    family: str,
    family_stats: dict[str, FamilyStats],
    total_calls: int,
) -> float:
    stats = family_stats.setdefault(family, FamilyStats())
    if stats.calls == 0:
        return 1.35

    exploitation = (
        0.65 * stats.new_chunk_rate()
        + 0.35 * stats.mean_reward()
    )
    exploration = 0.45 * math.sqrt(
        math.log(total_calls + 2) / stats.calls
    )
    return min(1.5, exploitation + exploration)


def family_new_chunk_prior(
    family: str,
    family_stats: dict[str, FamilyStats],
) -> float:
    stats = family_stats.setdefault(family, FamilyStats())
    return (stats.new_chunks + 1.2) / (stats.calls + 2.4)


def recent_new_chunk_rate(history: list[dict[str, Any]], window: int) -> float:
    recent = history[-window:]
    if not recent:
        return 1.0
    return sum(item.get("reward", 0) > 0 for item in recent) / len(recent)


def candidate_utility(
    candidate: QueryCandidate,
    clean_query: str,
    runtime: ModelRuntime,
    seen_index: SeenIndex,
    used_queries: list[str],
    family_stats: dict[str, FamilyStats],
    history: list[dict[str, Any]],
    evidence_map: dict[str, Any],
    unique_chunks: int,
    bootstrap_unique_target: int,
) -> tuple[float, dict[str, float]]:
    seen = seen_index.metrics(candidate.query)
    used_similarity = max_similarity_to_used(
        runtime,
        candidate.query,
        used_queries,
    )

    relevance = max(
        0.0,
        runtime.cosine(candidate.query, clean_query),
    )
    total_calls = sum(item.calls for item in family_stats.values())
    ucb = family_ucb(
        candidate.family,
        family_stats,
        total_calls,
    )
    new_chunk_prior = family_new_chunk_prior(
        candidate.family,
        family_stats,
    )

    gap_priority = clip01(
        candidate.metadata.get("gap_priority", 0.0)
    )
    coverage_gain = clip01(
        candidate.metadata.get("coverage_gain", 0.45)
    )
    source_priority = clip01(
        candidate.metadata.get("source_priority", 0.0)
    )
    source_probe_bonus = (
        0.10 if candidate.metadata.get("source_probe") else 0.0
    )
    source_followup_bonus = (
        0.07 if candidate.metadata.get("source_followup") else 0.0
    )
    semantic_novelty = 1.0 - used_similarity

    high_gaps = [
        gap for gap in evidence_map.get("unresolved_factual_gaps", [])
        if clip01(gap.get("priority", 0.0)) >= 0.72
    ]

    track_boost = 0.0
    if unique_chunks < bootstrap_unique_target:
        if candidate.track == "discovery":
            track_boost += 0.13
    else:
        recent_rate = recent_new_chunk_rate(history, 8)
        if high_gaps and candidate.track == "gap":
            track_boost += 0.12
        if recent_rate >= 0.40 and candidate.track == "discovery":
            track_boost += 0.07

    # Rotate away from the currently failing track.
    duplicate_streak = 0
    failing_track: Optional[str] = None
    for item in reversed(history):
        if item.get("reward", 0) > 0:
            break
        duplicate_streak += 1
        failing_track = item.get("track")
    if duplicate_streak >= 3 and candidate.track != failing_track:
        track_boost += 0.08

    utility = (
        0.24 * new_chunk_prior
        + 0.20 * gap_priority
        + 0.14 * coverage_gain
        + 0.14 * source_priority
        + 0.12 * relevance
        + 0.09 * seen["unseen_term_ratio"]
        + 0.07 * semantic_novelty
        + 0.05 * min(1.0, ucb)
        + 0.03 * candidate.base_priority
        + source_probe_bonus
        + source_followup_bonus
        + track_boost
        - 0.25 * seen["seen_risk"]
        - 0.11 * used_similarity
    )

    metrics = {
        **seen,
        "used_similarity": used_similarity,
        "semantic_novelty": semantic_novelty,
        "relevance": relevance,
        "family_ucb": ucb,
        "new_chunk_prior": new_chunk_prior,
        "gap_priority": gap_priority,
        "coverage_gain": coverage_gain,
        "source_priority": source_priority,
        "source_probe_bonus": source_probe_bonus,
        "source_followup_bonus": source_followup_bonus,
        "track_boost": track_boost,
        "utility": utility,
    }
    return utility, metrics


def pop_best_candidate(
    pool: CandidatePool,
    clean_query: str,
    runtime: ModelRuntime,
    seen_index: SeenIndex,
    used_queries: list[str],
    family_stats: dict[str, FamilyStats],
    history: list[dict[str, Any]],
    evidence_map: dict[str, Any],
    config: EvidenceRetrievalConfig,
) -> tuple[Optional[QueryCandidate], Optional[dict[str, float]]]:
    best: Optional[QueryCandidate] = None
    best_metrics: Optional[dict[str, float]] = None
    best_score = -999.0
    rejected: list[QueryCandidate] = []

    for candidate in list(pool.items):
        utility, metrics = candidate_utility(
            candidate=candidate,
            clean_query=clean_query,
            runtime=runtime,
            seen_index=seen_index,
            used_queries=used_queries,
            family_stats=family_stats,
            history=history,
            evidence_map=evidence_map,
            unique_chunks=len(seen_index.chunk_ids),
            bootstrap_unique_target=config.bootstrap_unique_target,
        )

        hard_seen_copy = (
            metrics["seen_coverage"]
            >= config.hard_seen_coverage_reject
            and metrics["seen_semantic"]
            >= config.hard_seen_semantic_reject
        )
        hard_used_duplicate = (
            metrics["used_similarity"]
            >= config.used_query_semantic_reject
        )

        if hard_seen_copy or hard_used_duplicate:
            rejected.append(candidate)
            continue

        if utility > best_score:
            best = candidate
            best_metrics = metrics
            best_score = utility

    for candidate in rejected:
        pool.remove(candidate)

    if best is not None:
        pool.remove(best)

    return best, best_metrics


def pop_mandatory_source_candidate(
    pool: CandidatePool,
    source_type: str,
    clean_query: str,
    runtime: ModelRuntime,
    seen_index: SeenIndex,
    used_queries: list[str],
    family_stats: dict[str, FamilyStats],
    history: list[dict[str, Any]],
    evidence_map: dict[str, Any],
    config: EvidenceRetrievalConfig,
) -> tuple[Optional[QueryCandidate], Optional[dict[str, float]]]:
    best: Optional[QueryCandidate] = None
    best_metrics: Optional[dict[str, float]] = None
    best_score = -999.0
    rejected: list[QueryCandidate] = []

    for candidate in list(pool.items):
        if candidate.metadata.get("target_source_type") != source_type:
            continue
        if not candidate.metadata.get("source_probe"):
            continue

        utility, metrics = candidate_utility(
            candidate=candidate,
            clean_query=clean_query,
            runtime=runtime,
            seen_index=seen_index,
            used_queries=used_queries,
            family_stats=family_stats,
            history=history,
            evidence_map=evidence_map,
            unique_chunks=len(seen_index.chunk_ids),
            bootstrap_unique_target=config.bootstrap_unique_target,
        )

        hard_used_duplicate = (
            metrics["used_similarity"]
            >= config.used_query_semantic_reject
        )
        if hard_used_duplicate:
            rejected.append(candidate)
            continue

        if utility > best_score:
            best = candidate
            best_metrics = metrics
            best_score = utility

    for candidate in rejected:
        pool.remove(candidate)

    if best is not None:
        pool.remove(best)

    return best, best_metrics


# ---------------------------------------------------------------------------
# Evidence-only LLM analysis
# ---------------------------------------------------------------------------

AMOUNT_PATTERN = re.compile(
    r"\b\d{1,3}(?:[.,]\d{3})+(?:\s*)?(?:đ|đồng)?\b",
    flags=re.IGNORECASE,
)
DATE_PATTERN = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
IDENTIFIER_PATTERN = re.compile(
    r"\b(?:BKS|GCN|QĐ|HĐ|Số)\s*[:\-]?\s*[A-Z0-9./\-–]+\b",
    flags=re.IGNORECASE,
)


def valid_anchor(anchor: str, source_text: str) -> bool:
    anchor = " ".join(str(anchor).split()).strip(" .?;:,-")
    normalized_anchor = normalize_text(anchor)
    normalized_source = normalize_text(source_text)
    tokens = content_tokens(anchor)

    if not 1 <= len(tokens) <= 8:
        return False
    if normalized_anchor not in normalized_source:
        return False
    if tokens[-1] in BAD_ANCHOR_ENDINGS:
        return False
    return True


def deterministic_anchors(text: str) -> list[str]:
    candidates: list[str] = []
    candidates.extend(AMOUNT_PATTERN.findall(text))
    candidates.extend(DATE_PATTERN.findall(text))
    candidates.extend(IDENTIFIER_PATTERN.findall(text))

    # Short title-cased spans. They are verified against source text later.
    candidates.extend(
        re.findall(
            r"\b(?:[A-ZĐ][\wÀ-ỹđĐ]+(?:\s+|$)){2,4}",
            text,
        )
    )

    output: list[str] = []
    seen = set()
    for candidate in candidates:
        candidate = " ".join(candidate.split()).strip()
        normalized = normalize_text(candidate)
        if (
            normalized
            and normalized not in seen
            and valid_anchor(candidate, text)
        ):
            seen.add(normalized)
            output.append(candidate)
    return output[:14]


def select_diverse_chunks(
    runtime: ModelRuntime,
    archive: dict[str, dict[str, Any]],
    max_chunks: int,
) -> list[dict[str, str]]:
    items = list(archive.values())
    if len(items) <= max_chunks:
        return [
            {"chunk_id": item["chunk_id"], "text": item["text"]}
            for item in items
        ]

    texts = [item["text"] for item in items]
    vectors = runtime.encode(texts)

    selected = [len(items) - 1]
    while len(selected) < max_chunks:
        best_index = None
        best_distance = -1.0

        for index in range(len(items)):
            if index in selected:
                continue
            max_similarity = max(
                float(np.dot(vectors[index], vectors[picked]))
                for picked in selected
            )
            distance = 1.0 - max_similarity
            if distance > best_distance:
                best_distance = distance
                best_index = index

        if best_index is None:
            break
        selected.append(best_index)

    selected.sort()
    return [
        {
            "chunk_id": items[index]["chunk_id"],
            "text": items[index]["text"],
        }
        for index in selected
    ]


class EvidenceAnalyzer:
    def __init__(
        self,
        runtime: ModelRuntime,
        config: EvidenceRetrievalConfig,
    ):
        self.runtime = runtime
        self.config = config


    def extract_case_roles(
        self,
        clean_query: str,
    ) -> dict[str, list[str]]:
        system = """
Bạn nhận diện vai trò tố tụng từ phần mô tả vụ án.
Chỉ trích tên hoặc bí danh xuất hiện nguyên văn trong CASE QUERY.
Không suy đoán người làm chứng, Kiểm sát viên hoặc người khác.
Không áp dụng luật và không dự đoán kết quả.
Chỉ trả về JSON.
"""
        user = f"""
CASE QUERY:
{clean_query}

Schema:
{{
  "plaintiff_aliases": ["tên/bí danh nguyên văn"],
  "defendant_aliases": ["tên/bí danh nguyên văn"]
}}
"""
        try:
            parsed = self.runtime.generate_json(
                system,
                user,
                max_new_tokens=350,
            )
        except Exception as exc:
            print("Cảnh báo extract roles:", type(exc).__name__, exc)
            parsed = {}

        normalized_query = normalize_text(clean_query)

        def valid_aliases(values: Any) -> list[str]:
            output = []
            seen = set()
            if not isinstance(values, list):
                return output
            for value in values:
                alias = " ".join(str(value).split()).strip(" .?;:,-")
                normalized = normalize_text(alias)
                if (
                    normalized
                    and normalized in normalized_query
                    and normalized not in seen
                    and 1 <= len(alias.split()) <= 6
                ):
                    seen.add(normalized)
                    output.append(alias)
            return output[:6]

        return {
            "plaintiff_aliases": valid_aliases(
                parsed.get("plaintiff_aliases", [])
            ),
            "defendant_aliases": valid_aliases(
                parsed.get("defendant_aliases", [])
            ),
            # These are discovered dynamically from retrieved chunks.
            "witness_aliases": [],
            "prosecutor_aliases": [],
        }

    def generate_source_probe_candidates(
        self,
        clean_query: str,
        roles: dict[str, list[str]],
        generation: int = 0,
    ) -> list[QueryCandidate]:
        templates = {
            "plaintiff": [
                "nguyên đơn trình bày yêu cầu khởi kiện",
                "tại đơn khởi kiện nguyên đơn trình bày",
                "tại bản tự khai nguyên đơn trình bày",
                "tại phiên tòa nguyên đơn trình bày",
                "lời trình bày của nguyên đơn",
            ],
            "defendant": [
                "bị đơn trình bày ý kiến",
                "tại bản tự khai bị đơn trình bày",
                "tại phiên tòa bị đơn trình bày",
                "bị đơn không đồng ý yêu cầu khởi kiện",
                "lời trình bày của bị đơn",
            ],
            "witness": [
                "người làm chứng trình bày",
                "người chứng kiến sự việc trình bày",
                "theo lời khai của người chứng kiến",
                "người làm chứng xác nhận sự việc",
                "nhân chứng trình bày sự việc",
            ],
            "prosecutor": [
                "đại diện Viện kiểm sát phát biểu quan điểm",
                "Viện kiểm sát đề nghị Hội đồng xét xử",
                "ý kiến của Kiểm sát viên tại phiên tòa",
            ],
        }

        for alias in roles.get("plaintiff_aliases", [])[:3]:
            templates["plaintiff"].insert(0, f"{alias} trình bày")
            templates["plaintiff"].append(
                f"theo đơn khởi kiện của {alias}"
            )

        for alias in roles.get("defendant_aliases", [])[:3]:
            templates["defendant"].insert(0, f"{alias} trình bày")
            templates["defendant"].append(
                f"ý kiến của {alias} đối với yêu cầu khởi kiện"
            )

        output: list[QueryCandidate] = []
        for source_type in SOURCE_TYPES:
            for index, query in enumerate(templates[source_type]):
                output.append(QueryCandidate(
                    query=sanitize_query(query),
                    family=f"source_probe_{source_type}",
                    track="gap",
                    base_priority=max(0.76, 1.00 - 0.04 * index),
                    rationale=(
                        "Chủ động tìm "
                        + SOURCE_LABELS[source_type]
                    ),
                    generation=generation,
                    metadata={
                        "target_source_type": source_type,
                        "source_probe": True,
                        "source_priority": 0.96,
                        "gap_priority": 0.90,
                        "coverage_gain": 0.86,
                    },
                ))
        return output

    def generate_entity_source_candidates(
        self,
        roles: dict[str, list[str]],
        generation: int,
    ) -> list[QueryCandidate]:
        templates = {
            "plaintiff": [
                "{alias} trình bày",
                "theo lời trình bày của {alias}",
                "{alias} yêu cầu",
                "{alias} cho rằng",
            ],
            "defendant": [
                "{alias} trình bày",
                "theo lời trình bày của {alias}",
                "{alias} không đồng ý",
                "{alias} cho rằng",
            ],
            "witness": [
                "{alias} trình bày",
                "theo lời khai của {alias}",
                "{alias} chứng kiến sự việc",
                "{alias} xác nhận",
                "{alias} nhìn thấy",
                "{alias} nghe thấy",
            ],
            "prosecutor": [
                "{alias} phát biểu quan điểm",
                "ý kiến của {alias}",
            ],
        }

        output: list[QueryCandidate] = []
        for source_type in SOURCE_TYPES:
            candidates_for_type = 0
            for alias in source_aliases(roles, source_type):
                for index, template in enumerate(templates[source_type]):
                    if (
                        candidates_for_type
                        >= self.config.max_entity_source_candidates_per_type
                    ):
                        break
                    query = sanitize_query(
                        template.format(alias=alias)
                    )
                    if not query:
                        continue
                    output.append(QueryCandidate(
                        query=query,
                        family=f"source_entity_{source_type}",
                        track="gap",
                        base_priority=max(0.74, 0.96 - 0.04 * index),
                        rationale=(
                            "Tìm lời trình bày theo tên/bí danh đã "
                            f"xác định của {SOURCE_LABELS[source_type]}"
                        ),
                        generation=generation,
                        metadata={
                            "target_source_type": source_type,
                            "source_probe": True,
                            "entity_source_probe": True,
                            "source_priority": 0.97,
                            "gap_priority": 0.91,
                            "coverage_gain": 0.88,
                            "speaker_alias": alias,
                        },
                    ))
                    candidates_for_type += 1
        return output

    def analyze_source_coverage(
        self,
        clean_query: str,
        archive: dict[str, dict[str, Any]],
        roles: dict[str, list[str]],
        previous_coverage: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        # Critical separation:
        # - retrieval family/target explains how a chunk was found;
        # - source attribution explains whose statement is in the chunk.
        #
        # The LLM never receives first_family, first_found_by, query target,
        # or target_source_hit here, so an off-target retrieval cannot leak
        # into the source classification.
        grouped: dict[str, list[dict[str, Any]]] = {
            source_type: [] for source_type in SOURCE_TYPES
        }
        mention_ids: dict[str, list[str]] = {
            source_type: [] for source_type in SOURCE_TYPES
        }

        for chunk_id, chunk in archive.items():
            attributions = list(chunk.get("source_attributions", []))
            if not attributions:
                attributions = deterministic_source_attributions(
                    chunk.get("text", ""),
                    roles,
                )

            for attribution in attributions:
                source_type = attribution.get("source_type")
                if source_type not in SOURCE_TYPES:
                    continue
                if chunk_id not in mention_ids[source_type]:
                    mention_ids[source_type].append(chunk_id)

                if attribution.get("mode") in {
                    "DIRECT_STATEMENT",
                    "ATTRIBUTED_STATEMENT",
                }:
                    grouped[source_type].append({
                        "chunk_id": chunk_id,
                        "text": chunk.get("text", ""),
                        "attribution_mode": attribution.get("mode"),
                        "speaker_aliases": attribution.get(
                            "speaker_aliases", []
                        ),
                        "evidence_span": attribution.get(
                            "evidence_span", ""
                        ),
                    })

        for source_type in SOURCE_TYPES:
            grouped[source_type] = grouped[source_type][
                :self.config.max_analysis_chunks
            ]

        system = """
Bạn đánh giá độ phủ của bốn nguồn bằng chứng trong nội dung vụ án:
1. lời trình bày của nguyên đơn;
2. lời trình bày của bị đơn;
3. lời khai của người làm chứng/người chứng kiến;
4. ý kiến của Viện kiểm sát.

Dữ liệu đã được nhóm theo NGUỒN QUAN SÁT TRONG NỘI DUNG CHUNK.
Không dùng query hoặc family đã truy xuất chunk để suy ra nguồn.
Một query nhắm người làm chứng có thể trả về đoạn Viện kiểm sát; khi đó
đoạn chỉ thuộc Viện kiểm sát.

Đây chỉ là giai đoạn truy xuất case facts.
Không áp dụng luật, không dự đoán bên thắng và không kết luận yêu cầu
được chấp nhận hay bị bác.
Chỉ trả về JSON.
"""

        user = f"""
CASE QUERY:
{clean_query}

VAI TRÒ VÀ BÍ DANH ĐÃ XÁC ĐỊNH:
{json.dumps(roles, ensure_ascii=False)}

CHUNK ĐÃ ĐƯỢC GÁN NGUỒN THEO NỘI DUNG:
{json.dumps(grouped, ensure_ascii=False)}

SOURCE COVERAGE LẦN TRƯỚC:
{json.dumps(previous_coverage or {{}}, ensure_ascii=False)}

Quy tắc:
- Chỉ evidence_ids trong nhóm tương ứng mới hợp lệ.
- NOT_FOUND: chưa có direct/attributed statement.
- PARTIAL: có nội dung cụ thể nhưng còn thiếu hoặc bị cắt.
- SUBSTANTIVE: lời trình bày đã khá rõ trong các chunk hiện có.
- follow_up_questions là câu hỏi nội bộ.
- query_hooks dài 3-11 từ, có dạng cụm từ trong bản án.
- Ưu tiên query theo tên/bí danh khi đã biết, ví dụ:
  "chị N trình bày", "theo lời khai của chị N",
  không chỉ dùng cụm vai trò chung.
- Không thêm tên, ngày, số tiền, tài sản hoặc giấy tờ mới.

Schema:
{{
  "sources": {{
    "plaintiff": {{
      "status": "NOT_FOUND|PARTIAL|SUBSTANTIVE",
      "evidence_ids": ["chunk_id"],
      "substantive_points": ["..."],
      "missing_aspects": ["..."],
      "follow_up_questions": ["..."],
      "query_hooks": ["..."],
      "priority": 0.0,
      "probe_recommended": true
    }},
    "defendant": {{ "...": "..." }},
    "witness": {{ "...": "..." }},
    "prosecutor": {{ "...": "..." }}
  }}
}}
"""
        try:
            parsed = self.runtime.generate_json(
                system,
                user,
                max_new_tokens=1800,
            )
        except Exception as exc:
            print(
                "Cảnh báo source coverage:",
                type(exc).__name__,
                exc,
            )
            return previous_coverage or empty_source_coverage()

        result = empty_source_coverage()
        raw_sources = parsed.get("sources", {})
        if not isinstance(raw_sources, dict):
            raw_sources = {}

        for source_type in SOURCE_TYPES:
            item = raw_sources.get(source_type, {})
            if not isinstance(item, dict):
                item = {}

            eligible_ids = list(dict.fromkeys(
                row["chunk_id"] for row in grouped[source_type]
            ))
            eligible_set = set(eligible_ids)
            evidence_ids = [
                value
                for value in item.get("evidence_ids", [])
                if value in eligible_set
            ]

            # If a statement was observed, preserve at least one evidence ID
            # even when the JSON model omitted it.
            if eligible_ids and not evidence_ids:
                evidence_ids = eligible_ids[:]

            requested_status = str(
                item.get("status", "NOT_FOUND")
            ).upper()
            if requested_status not in SOURCE_STATUS_VALUES:
                requested_status = "NOT_FOUND"

            if not eligible_ids:
                status = (
                    "MENTION_ONLY"
                    if mention_ids[source_type]
                    else "NOT_FOUND"
                )
            elif requested_status == "SUBSTANTIVE":
                status = "SUBSTANTIVE"
            else:
                status = "PARTIAL"

            hooks = []
            for value in item.get("query_hooks", []):
                hook = sanitize_query(value)
                if (
                    hook
                    and not has_question_form(hook)
                    and len(content_tokens(hook)) >= 2
                ):
                    hooks.append(hook)

            priority = max(
                clip01(item.get("priority", 0.0)),
                source_status_priority(status, self.config),
            )

            result[source_type] = {
                "source_type": source_type,
                "label": SOURCE_LABELS[source_type],
                "status": status,
                "evidence_ids": evidence_ids,
                "mention_ids": mention_ids[source_type],
                "substantive_points": (
                    [
                        str(value).strip()
                        for value in item.get(
                            "substantive_points", []
                        )
                        if str(value).strip()
                    ][:10]
                    if eligible_ids else []
                ),
                "missing_aspects": [
                    str(value).strip()
                    for value in item.get("missing_aspects", [])
                    if str(value).strip()
                ][:8],
                "follow_up_questions": [
                    str(value).strip()
                    for value in item.get(
                        "follow_up_questions", []
                    )
                    if str(value).strip()
                ][:8],
                "query_hooks": hooks[:8],
                "priority": priority,
                "probe_recommended": bool(
                    item.get(
                        "probe_recommended",
                        status != "SUBSTANTIVE",
                    )
                ),
            }

        return result

    def candidates_from_source_coverage(
        self,
        source_coverage: dict[str, Any],
        generation: int,
        source_query_attempts: Optional[dict[str, int]] = None,
    ) -> list[QueryCandidate]:
        source_query_attempts = source_query_attempts or {}
        output: list[QueryCandidate] = []

        for source_type in SOURCE_TYPES:
            item = source_coverage.get(source_type, {})
            status = str(item.get("status", "NOT_FOUND")).upper()
            attempts = int(source_query_attempts.get(source_type, 0))

            if attempts >= self.config.max_source_queries_per_type:
                continue
            if (
                status == "SUBSTANTIVE"
                and not item.get("missing_aspects")
                and not item.get("query_hooks")
            ):
                continue

            priority = clip01(
                item.get(
                    "priority",
                    source_status_priority(status, self.config),
                )
            )

            hooks = list(item.get("query_hooks", []))
            for index, hook in enumerate(
                hooks[:self.config.max_source_followup_candidates_per_type]
            ):
                query = sanitize_query(hook)
                if not query:
                    continue
                output.append(QueryCandidate(
                    query=query,
                    family=f"source_followup_{source_type}",
                    track="gap",
                    base_priority=max(0.70, 0.93 - 0.04 * index),
                    rationale=(
                        "LLM follow-up cho "
                        + SOURCE_LABELS[source_type]
                    ),
                    generation=generation,
                    metadata={
                        "target_source_type": source_type,
                        "source_followup": True,
                        "source_priority": priority,
                        "gap_priority": priority,
                        "coverage_gain": priority,
                    },
                ))

        return output

    def generate_seed_queries(
        self,
        clean_query: str,
    ) -> list[QueryCandidate]:
        candidates: list[QueryCandidate] = []

        for sentence in split_sentences(clean_query):
            if len(content_tokens(sentence)) >= 3:
                candidates.append(QueryCandidate(
                    query=truncate_words(sentence, 18),
                    family="case_query_subset",
                    track="discovery",
                    base_priority=0.95,
                    metadata={"coverage_gain": 0.70},
                ))

        words = clean_query.split()
        for window_size in (5, 8, 11):
            if len(words) <= window_size:
                continue
            step = max(3, window_size // 2)
            for start in range(
                0,
                len(words) - window_size + 1,
                step,
            ):
                window = " ".join(words[start:start + window_size])
                if len(content_tokens(window)) >= 3:
                    candidates.append(QueryCandidate(
                        query=window,
                        family="lexical_window",
                        track="discovery",
                        base_priority=0.76,
                        metadata={"coverage_gain": 0.62},
                    ))

        system = """
Bạn sinh truy vấn BM25 để truy xuất các đoạn sự kiện của một vụ án.
Đây chỉ là giai đoạn lấy bằng chứng vụ án.
Không được dự đoán bên thắng, không áp dụng điều luật, không kết luận pháp lý.
Không dùng checklist cố định vì mỗi vụ án có cấu trúc khác nhau.
Chỉ trả về JSON.
"""

        user = f"""
CASE QUERY:
{clean_query}

Sinh tối đa 14 truy vấn mở đầu để tìm các đoạn sự kiện khác nhau của đúng case.

Quy tắc:
- Dựa trên người, tài sản, hành vi, sự kiện, yêu cầu, phạm vi, số tiền,
  diện tích, giấy tờ hoặc mốc thời gian thực sự xuất hiện trong CASE QUERY.
- Mỗi query dài 3-11 từ.
- Viết như một cụm có thể xuất hiện trong bản án; không viết câu hỏi.
- Không thêm tên riêng, ngày, số tiền, diện tích hoặc tài sản mới.
- Các query phải đa dạng, không chỉ diễn đạt lại cùng một sự kiện.
- Không sinh nhãn thắng/thua và không nhắc điều luật.

Schema:
{{
  "queries": [
    {{
      "query": "...",
      "family": "entity_event|request_scope|object_document|timeline|semantic_diversification",
      "rationale": "vùng bằng chứng dự kiến"
    }}
  ]
}}
"""
        try:
            parsed = self.runtime.generate_json(
                system,
                user,
                max_new_tokens=950,
            )
            for item in parsed.get("queries", []):
                if not isinstance(item, dict):
                    continue
                query = sanitize_query(item.get("query", ""))
                if query:
                    candidates.append(QueryCandidate(
                        query=query,
                        family=str(
                            item.get("family", "semantic_diversification")
                        ),
                        track="discovery",
                        base_priority=0.84,
                        rationale=str(item.get("rationale", "")),
                        metadata={"coverage_gain": 0.72},
                    ))
        except Exception as exc:
            print("Cảnh báo seed LLM:", type(exc).__name__, exc)

        return candidates[:self.config.max_seed_queries * 2]

    def analyze_chunk(
        self,
        clean_query: str,
        chunk_id: str,
        chunk_text: str,
        generation: int,
        roles: dict[str, list[str]],
    ) -> dict[str, Any]:
        system = """
Bạn phân tích một chunk 100 từ được cắt từ nội dung vụ án.
Mục tiêu duy nhất là hỗ trợ truy xuất thêm bằng chứng vụ án.
Không được dự đoán ai thắng, không đánh giá yêu cầu được chấp nhận,
không áp dụng luật và không kết luận pháp lý.
Không giả định chunk ở đầu, giữa hay cuối.
Chỉ trả về JSON.
"""
        user = f"""
CASE QUERY:
{clean_query}

CHUNK ID:
{chunk_id}

CHUNK TEXT:
{chunk_text}

VAI TRÒ/BÍ DANH ĐÃ BIẾT:
{json.dumps(roles, ensure_ascii=False)}

Hãy:
1. Trích tối đa 7 atomic facts chỉ có trong CHUNK.
2. Trích tối đa 10 anchors nguyên văn, mỗi anchor 1-8 từ nội dung.
3. Sinh tối đa 12 query nhằm tìm các đoạn khác của cùng case:
   - làm rõ sự kiện, phạm vi yêu cầu, quan điểm hoặc chứng cứ được nhắc;
   - tìm phần còn thiếu quanh một sự kiện hoặc đối tượng;
   - tìm tài liệu, số lượng, diện tích, tài sản hoặc mốc thời gian liên quan;
   - tìm một góc sự kiện khác với CHUNK.
4. Vì các chunk là các đoạn 100 từ liên tiếp, nếu CHUNK bắt đầu hoặc kết thúc
   giữa câu/ý, sinh tối đa 4 boundary_queries để tìm đoạn ngay trước hoặc ngay sau.
   Boundary query phải là cụm 3-10 từ có khả năng xuất hiện ở đoạn lân cận,
   được neo vào người/sự kiện có thật trong CHUNK, nhưng không chép nguyên
   cả câu và không thêm định danh mới.
5. Query phải là cụm từ có thể xuất hiện trong bản án, không phải câu hỏi.
6. Xác định NGUỒN THỰC SỰ CÓ TRONG NỘI DUNG CHUNK, độc lập với query
   đã tìm ra chunk:
   - DIRECT_STATEMENT: chunk trực tiếp ghi người/nguồn đó trình bày.
   - ATTRIBUTED_STATEMENT: chunk dẫn "theo lời khai/trình bày của ...".
   - MENTION_ONLY: chỉ nhắc đến tên hoặc vai trò.
   Nguyên đơn và bị đơn có thể chỉ được ghi bằng tên/bí danh.
   Người làm chứng có thể được ghi là một người cụ thể "chị X", "ông Y",
   "người chứng kiến", "người biết sự việc", hoặc người nói mình đã
   nhìn thấy/nghe thấy/chứng kiến sự kiện.
   Không gán witness cho nguyên đơn/bị đơn chỉ vì họ kể lại sự việc.
7. Trích discovered_aliases là tên/bí danh xuất hiện nguyên văn trong CHUNK.
   Không suy đoán tên không có trong CHUNK.

Không thêm tên riêng, ngày, số tiền, diện tích, biển số hoặc giấy tờ mới.
Không lặp cả câu từ CHUNK.
Không sinh nội dung thắng/thua hoặc điều luật.

Schema:
{{
  "atomic_facts": ["..."],
  "anchors": ["..."],
  "queries": [
    {{
      "query": "...",
      "family": "local_event|request_scope|position_statement|object_document|timeline|semantic_diversification",
      "track": "discovery|exploration",
      "rationale": "..."
    }}
  ],
  "boundary_queries": [
    {{
      "direction": "previous|next",
      "query": "...",
      "rationale": "dấu hiệu chunk bị cắt ở đầu/cuối"
    }}
  ],
  "source_attributions": [
    {{
      "source_type": "plaintiff|defendant|witness|prosecutor",
      "mode": "DIRECT_STATEMENT|ATTRIBUTED_STATEMENT|MENTION_ONLY",
      "speaker_aliases": ["tên/bí danh nguyên văn"],
      "evidence_span": "cụm nguyên văn ngắn trong CHUNK",
      "confidence": 0.0
    }}
  ],
  "discovered_aliases": {{
    "plaintiff": ["..."],
    "defendant": ["..."],
    "witness": ["..."],
    "prosecutor": ["..."]
  }}
}}
"""
        try:
            parsed = self.runtime.generate_json(
                system,
                user,
                max_new_tokens=1200,
            )
        except Exception as exc:
            print("Cảnh báo analyze_chunk:", type(exc).__name__, exc)
            parsed = {}

        facts = [
            str(value).strip()
            for value in parsed.get("atomic_facts", [])
            if str(value).strip()
        ][:7]

        anchors = deterministic_anchors(chunk_text)
        seen_anchor = {normalize_text(value) for value in anchors}
        for value in parsed.get("anchors", []):
            anchor = str(value).strip()
            normalized = normalize_text(anchor)
            if (
                normalized
                and normalized not in seen_anchor
                and valid_anchor(anchor, chunk_text)
            ):
                seen_anchor.add(normalized)
                anchors.append(anchor)

        source_attribution_items = deterministic_source_attributions(
            chunk_text,
            roles,
        )

        for item in parsed.get("source_attributions", []):
            if not isinstance(item, dict):
                continue
            source_type = str(
                item.get("source_type", "")
            ).lower()
            mode = str(
                item.get("mode", "MENTION_ONLY")
            ).upper()
            confidence = clip01(item.get("confidence", 0.0))
            evidence_span = " ".join(
                str(item.get("evidence_span", "")).split()
            )

            if source_type not in SOURCE_TYPES:
                continue
            if mode not in SOURCE_ATTRIBUTION_MODES:
                continue
            if (
                confidence
                < self.config.min_source_attribution_confidence
            ):
                continue
            if (
                evidence_span
                and normalize_text(evidence_span)
                not in normalize_text(chunk_text)
            ):
                continue

            aliases = []
            for alias in item.get("speaker_aliases", []):
                valid = valid_source_alias(alias, chunk_text)
                if valid:
                    aliases.append(valid)

            source_attribution_items.append({
                "source_type": source_type,
                "mode": mode,
                "speaker_aliases": aliases,
                "evidence_span": evidence_span,
                "confidence": confidence,
            })

        source_attribution_items = merge_attributions(
            source_attribution_items
        )

        discovered_aliases: dict[str, list[str]] = {
            source_type: [] for source_type in SOURCE_TYPES
        }
        raw_aliases = parsed.get("discovered_aliases", {})
        if not isinstance(raw_aliases, dict):
            raw_aliases = {}

        for source_type in SOURCE_TYPES:
            for alias in raw_aliases.get(source_type, []):
                valid = valid_source_alias(alias, chunk_text)
                if valid:
                    discovered_aliases[source_type].append(valid)

        for attribution in source_attribution_items:
            source_type = attribution["source_type"]
            for alias in attribution.get("speaker_aliases", []):
                if alias not in discovered_aliases[source_type]:
                    discovered_aliases[source_type].append(alias)

        # Prevent a known plaintiff/defendant alias from being simultaneously
        # learned as a witness solely because the person describes an event.
        party_aliases = {
            normalize_text(alias)
            for key in ("plaintiff_aliases", "defendant_aliases")
            for alias in roles.get(key, [])
        }
        discovered_aliases["witness"] = [
            alias
            for alias in discovered_aliases["witness"]
            if normalize_text(alias) not in party_aliases
        ]

        candidates: list[QueryCandidate] = []
        for item in parsed.get("queries", []):
            if not isinstance(item, dict):
                continue
            query = sanitize_query(item.get("query", ""))
            if not query:
                continue
            track = str(item.get("track", "discovery"))
            if track not in {"discovery", "exploration"}:
                track = "discovery"
            candidates.append(QueryCandidate(
                query=query,
                family=str(item.get("family", "local_event")),
                track=track,
                base_priority=0.75,
                source_chunk_id=chunk_id,
                rationale=str(item.get("rationale", "")),
                generation=generation,
                metadata={"coverage_gain": 0.62},
            ))

        for item in parsed.get("boundary_queries", [])[
            :self.config.max_boundary_queries_per_chunk
        ]:
            if not isinstance(item, dict):
                continue
            query = sanitize_query(item.get("query", ""), max_words=10)
            direction = str(item.get("direction", "next")).lower()
            if direction not in {"previous", "next"}:
                direction = "next"
            if (
                query
                and not has_question_form(query)
                and len(content_tokens(query)) >= 2
            ):
                candidates.append(QueryCandidate(
                    query=query,
                    family=(
                        "boundary_backward"
                        if direction == "previous"
                        else "boundary_forward"
                    ),
                    track="exploration",
                    base_priority=0.78,
                    source_chunk_id=chunk_id,
                    rationale=str(item.get("rationale", "")),
                    generation=generation,
                    metadata={
                        "coverage_gain": 0.82,
                        "boundary_direction": direction,
                        "boundary_query": True,
                    },
                ))

        # Deterministic anchor combinations are limited and typed only by text.
        for anchor in anchors[:8]:
            candidates.append(QueryCandidate(
                query=anchor,
                family="anchor_exact",
                track="exploration",
                base_priority=0.52,
                source_chunk_id=chunk_id,
                generation=generation,
                metadata={"coverage_gain": 0.42},
            ))

        for left_index in range(min(5, len(anchors))):
            for right_index in range(
                left_index + 1,
                min(7, len(anchors)),
            ):
                left = anchors[left_index]
                right = anchors[right_index]
                candidates.append(QueryCandidate(
                    query=truncate_words(f"{left} {right}", 10),
                    family="anchor_pair",
                    track="exploration",
                    base_priority=0.59,
                    source_chunk_id=chunk_id,
                    generation=generation,
                    metadata={"coverage_gain": 0.52},
                ))

        candidates.sort(
            key=lambda candidate: (
                bool(candidate.metadata.get("boundary_query")),
                candidate.base_priority,
            ),
            reverse=True,
        )
        return {
            "atomic_facts": facts,
            "anchors": anchors[:14],
            "candidates": candidates[:self.config.max_chunk_queries],
            "source_attributions": source_attribution_items,
            "discovered_aliases": discovered_aliases,
        }

    def analyze_evidence_map(
        self,
        clean_query: str,
        archive: dict[str, dict[str, Any]],
        previous_map: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        selected = select_diverse_chunks(
            self.runtime,
            archive,
            self.config.max_analysis_chunks,
        )

        system = """
Bạn xây bản đồ bằng chứng sự kiện cho một vụ án.
Đây chưa phải giai đoạn tìm luật và chưa phải giai đoạn ra quyết định.
Không được dự đoán bên thắng, không áp dụng điều luật, không gán nhãn
chấp nhận/bác yêu cầu và không suy đoán phán quyết.
Không dùng checklist cố định; hãy tạo chủ đề động từ nội dung thực tế.
Chỉ trả về JSON.
"""

        user = f"""
CASE QUERY:
{clean_query}

CÁC CHUNK ĐÃ TRUY XUẤT:
{json.dumps(selected, ensure_ascii=False)}

BẢN ĐỒ LẦN TRƯỚC:
{json.dumps(previous_map or {{}}, ensure_ascii=False)}

Hãy tổng hợp trạng thái truy xuất bằng chứng:

1. atomic_facts:
   Chỉ ghi sự kiện có chứng cứ trực tiếp trong các chunk và kèm chunk_id.

2. evidence_topics:
   Tạo các cụm chủ đề động, không theo checklist cố định.
   Một topic có thể là một sự kiện, yêu cầu, tài sản, giao dịch, diễn biến,
   quan điểm, chứng từ, phạm vi, con số hoặc mâu thuẫn thực sự xuất hiện.
   Đánh giá coverage LOW/MEDIUM/HIGH dựa trên số góc thông tin đã có,
   không dựa trên khả năng thắng kiện.

3. unresolved_factual_gaps:
   Chỉ nêu phần sự kiện còn thiếu có khả năng giúp hiểu đầy đủ case
   hoặc giúp tìm thêm chunk khác biệt.
   Không nêu điều luật và không nêu kết quả cần thắng/thua.
   Mỗi gap có 3-6 query_hooks dạng cụm từ có thể xuất hiện trong bản án.
   Không viết câu hỏi và không thêm tên, ngày, số tiền, diện tích mới.

4. coverage_summary:
   Mô tả ngắn những vùng bằng chứng đã có và vùng còn thiếu.

Schema:
{{
  "atomic_facts": [
    {{
      "fact": "...",
      "evidence_ids": ["chunk_id"],
      "confidence": 0.0
    }}
  ],
  "evidence_topics": [
    {{
      "topic_id": "t1",
      "title": "...",
      "description": "...",
      "anchors": ["cụm có trong evidence"],
      "evidence_ids": ["chunk_id"],
      "coverage": "LOW|MEDIUM|HIGH",
      "missing_aspects": ["..."],
      "priority": 0.0
    }}
  ],
  "unresolved_factual_gaps": [
    {{
      "gap_id": "g1",
      "topic_id": "t1 hoặc null",
      "description": "...",
      "basis": "...",
      "query_hooks": ["..."],
      "priority": 0.0
    }}
  ],
  "coverage_summary": "...",
  "broad_search_needed": true
}}
"""
        try:
            parsed = self.runtime.generate_json(
                system,
                user,
                max_new_tokens=1800,
            )
        except Exception as exc:
            print("Cảnh báo evidence map:", type(exc).__name__, exc)
            return previous_map or {
                "atomic_facts": [],
                "evidence_topics": [],
                "unresolved_factual_gaps": [],
                "coverage_summary": "",
                "broad_search_needed": True,
            }

        valid_ids = set(archive)
        facts = []
        for item in parsed.get("atomic_facts", []):
            if not isinstance(item, dict):
                continue
            evidence_ids = [
                value for value in item.get("evidence_ids", [])
                if value in valid_ids
            ]
            fact = str(item.get("fact", "")).strip()
            if fact and evidence_ids:
                facts.append({
                    "fact": fact,
                    "evidence_ids": evidence_ids,
                    "confidence": clip01(item.get("confidence", 0.7)),
                })

        topics = []
        for index, item in enumerate(
            parsed.get("evidence_topics", []),
            start=1,
        ):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            if not title:
                continue
            evidence_ids = [
                value for value in item.get("evidence_ids", [])
                if value in valid_ids
            ]
            coverage = str(item.get("coverage", "LOW")).upper()
            if coverage not in {"LOW", "MEDIUM", "HIGH"}:
                coverage = "LOW"
            topics.append({
                "topic_id": str(item.get("topic_id", f"t{index}")),
                "title": title,
                "description": str(item.get("description", "")).strip(),
                "anchors": [
                    str(value).strip()
                    for value in item.get("anchors", [])
                    if str(value).strip()
                ][:10],
                "evidence_ids": evidence_ids,
                "coverage": coverage,
                "missing_aspects": [
                    str(value).strip()
                    for value in item.get("missing_aspects", [])
                    if str(value).strip()
                ][:8],
                "priority": clip01(item.get("priority", 0.5)),
            })

        gaps = []
        for index, item in enumerate(
            parsed.get("unresolved_factual_gaps", []),
            start=1,
        ):
            if not isinstance(item, dict):
                continue
            description = str(item.get("description", "")).strip()
            if not description:
                continue
            hooks = []
            for value in item.get("query_hooks", []):
                hook = sanitize_query(value)
                if (
                    hook
                    and not has_question_form(hook)
                    and len(content_tokens(hook)) >= 2
                ):
                    hooks.append(hook)
            gaps.append({
                "gap_id": str(item.get("gap_id", f"g{index}")),
                "topic_id": item.get("topic_id"),
                "description": description,
                "basis": str(item.get("basis", "")).strip(),
                "query_hooks": hooks[:6],
                "priority": clip01(item.get("priority", 0.5)),
            })

        return {
            "atomic_facts": facts,
            "evidence_topics": topics,
            "unresolved_factual_gaps": gaps,
            "coverage_summary": str(
                parsed.get("coverage_summary", "")
            ).strip(),
            "broad_search_needed": bool(
                parsed.get("broad_search_needed", True)
            ),
        }

    def candidates_from_evidence_map(
        self,
        evidence_map: dict[str, Any],
        generation: int,
    ) -> list[QueryCandidate]:
        output: list[QueryCandidate] = []

        topic_priority = {
            str(topic.get("topic_id")): clip01(
                topic.get("priority", 0.5)
            )
            for topic in evidence_map.get("evidence_topics", [])
        }

        for gap in evidence_map.get("unresolved_factual_gaps", []):
            gap_id = str(gap.get("gap_id", ""))
            topic_id = (
                str(gap.get("topic_id"))
                if gap.get("topic_id") is not None
                else None
            )
            priority = clip01(gap.get("priority", 0.5))
            priority = max(
                priority,
                topic_priority.get(topic_id or "", 0.0),
            )

            for hook in gap.get("query_hooks", []):
                query = sanitize_query(hook)
                if query:
                    output.append(QueryCandidate(
                        query=query,
                        family="evidence_gap",
                        track="gap",
                        base_priority=0.90,
                        source_topic_id=topic_id,
                        source_gap_id=gap_id,
                        rationale=str(gap.get("description", "")),
                        generation=generation,
                        metadata={
                            "gap_priority": priority,
                            "coverage_gain": priority,
                        },
                    ))

        return output[:self.config.max_gap_queries]

    def mutate_duplicate_query(
        self,
        original: QueryCandidate,
        returned_chunk: str,
        clean_query: str,
        generation: int,
    ) -> list[QueryCandidate]:
        if original.mutation_count >= self.config.max_mutations_per_branch:
            return []

        system = """
Bạn sửa một query BM25 đã trả lại chunk cũ.
Mục tiêu là truy xuất bằng chứng sự kiện khác của cùng case.
Không được dự đoán bên thắng, không áp dụng luật.
Chỉ trả về JSON.
"""
        user = f"""
CASE QUERY:
{clean_query}

QUERY CŨ:
{original.query}

CHUNK CŨ:
{returned_chunk}

Sinh tối đa 4 query thay thế.

Quy tắc:
- Dạng cụm từ có thể xuất hiện trong bản án, không viết câu hỏi.
- Giữ liên quan với vùng bằng chứng mà query cũ nhắm tới.
- Mỗi query cần ít nhất 2 từ nội dung không xuất hiện trong CHUNK CŨ.
- Giảm các từ khiến CHUNK CŨ có thể tiếp tục đứng top-1.
- Không thêm tên, ngày, số tiền, diện tích hoặc giấy tờ mới.
- Không nhắc luật hoặc kết quả vụ án.

Schema:
{{
  "queries": [
    {{
      "query": "...",
      "rationale": "..."
    }}
  ]
}}
"""
        try:
            parsed = self.runtime.generate_json(
                system,
                user,
                max_new_tokens=650,
            )
        except Exception as exc:
            print("Cảnh báo mutation:", type(exc).__name__, exc)
            return []

        chunk_tokens = set(content_tokens(returned_chunk))
        output = []
        for item in parsed.get("queries", []):
            if not isinstance(item, dict):
                continue
            query = sanitize_query(item.get("query", ""))
            if not query:
                continue
            unseen = [
                token for token in content_tokens(query)
                if token not in chunk_tokens
            ]
            if len(unseen) < 2:
                continue
            output.append(QueryCandidate(
                query=query,
                family=original.family,
                track=(
                    original.track
                    if original.track in {"discovery", "gap", "exploration"}
                    else "exploration"
                ),
                base_priority=max(0.48, original.base_priority - 0.08),
                source_chunk_id=original.source_chunk_id,
                source_topic_id=original.source_topic_id,
                source_gap_id=original.source_gap_id,
                parent_query=original.query,
                rationale=str(item.get("rationale", "")),
                mutation_count=original.mutation_count + 1,
                generation=generation,
                metadata={
                    **original.metadata,
                    "is_mutation": True,
                },
            ))
        return output


# ---------------------------------------------------------------------------
# Import/archive/checkpoint helpers
# ---------------------------------------------------------------------------

def load_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))



def family_stats_to_dict(
    stats: dict[str, FamilyStats],
) -> dict[str, dict[str, Any]]:
    return {
        family: asdict(value)
        for family, value in stats.items()
    }


def family_stats_from_dict(
    value: dict[str, dict[str, Any]],
) -> dict[str, FamilyStats]:
    return {
        family: FamilyStats(**stats)
        for family, stats in value.items()
    }


def checkpoint_name(case_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", case_id)
    return f"{safe}.json"


# ---------------------------------------------------------------------------
# Fresh-only evidence retrieval crawler
# ---------------------------------------------------------------------------

class CalibratedSourceAwareEvidenceCrawler:
    def __init__(
        self,
        runtime: ModelRuntime,
        client: ALQACClient,
        config: EvidenceRetrievalConfig,
        run_dir: Path,
    ):
        self.runtime = runtime
        self.client = client
        self.config = config
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.analyzer = EvidenceAnalyzer(runtime, config)

    def checkpoint_path(self, case_id: str) -> Path:
        return self.checkpoint_dir / checkpoint_name(case_id)

    def initialize_state(
        self,
        case: dict[str, Any],
    ) -> dict[str, Any]:
        case_id = str(case.get("case_id", "")).strip()
        case_query = str(case.get("case_query", "")).strip()
        if not case_id or not case_query:
            raise ValueError("Case phải có case_id và case_query.")

        pool = CandidatePool(self.config.max_candidate_pool_size)
        clean_query = clean_case_query(case_query)
        case_roles = self.analyzer.extract_case_roles(clean_query)

        state = {
            "case_id": case_id,
            "case_query": case_query,
            "clean_query": clean_query,
            "case_roles": case_roles,

            # Fresh run: all retrieval state starts empty.
            "archive": {},
            "used_queries": [],
            "query_logs": [],
            "history": [],
            "family_stats": {},
            "pool": pool,

            "evidence_map": {
                "atomic_facts": [],
                "evidence_topics": [],
                "unresolved_factual_gaps": [],
                "coverage_summary": "",
                "broad_search_needed": True,
            },
            "source_coverage": empty_source_coverage(),
            "source_query_attempts": {
                source_type: 0 for source_type in SOURCE_TYPES
            },
            "source_probe_attempts": {
                source_type: 0 for source_type in SOURCE_TYPES
            },
            "source_followup_attempts": {
                source_type: 0 for source_type in SOURCE_TYPES
            },
            "source_target_stats": {
                source_type: {
                    "attempts": 0,
                    "target_hits": 0,
                    "off_target_new_chunks": 0,
                    "duplicates": 0,
                    "failures": 0,
                }
                for source_type in SOURCE_TYPES
            },
            "source_probes_added": False,
            "source_bootstrap_refreshed": False,

            "generation": 0,
            "new_network_calls": 0,
            "target_budget": self.config.base_total_budget,
            "budget_history": [{
                "from": 0,
                "to": self.config.base_total_budget,
                "reason": "initial_base_budget",
                "network_calls": 0,
                "unique_chunks": 0,
            }],
            "last_best_candidate_utility": None,
            "new_chunks_since_refresh": 0,
            "queries_since_refresh": 0,
            "stop_reason": None,
        }

        state["pool"].extend(
            self.analyzer.generate_seed_queries(state["clean_query"])
        )
        return state

    def serialize_state(self, state: dict[str, Any]) -> dict[str, Any]:
        value = dict(state)
        value["pool"] = [asdict(item) for item in state["pool"].items]
        value["family_stats"] = family_stats_to_dict(
            state["family_stats"]
        )
        return value

    def deserialize_state(self, value: dict[str, Any]) -> dict[str, Any]:
        pool = CandidatePool(self.config.max_candidate_pool_size)
        for item in value.get("pool", []):
            pool.add(QueryCandidate(**item))
        value["pool"] = pool
        value["family_stats"] = family_stats_from_dict(
            value.get("family_stats", {})
        )
        return value

    def save_checkpoint(self, state: dict[str, Any]) -> None:
        atomic_write_json(
            self.checkpoint_path(state["case_id"]),
            self.serialize_state(state),
        )

    def load_checkpoint(
        self,
        case_id: str,
    ) -> Optional[dict[str, Any]]:
        path = self.checkpoint_path(case_id)
        if not path.exists():
            return None
        return self.deserialize_state(load_json_file(path))

    @staticmethod
    def merge_fact_list(
        existing: list[dict[str, Any]],
        incoming: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged = {
            normalize_text(item.get("fact", "")): item
            for item in existing
            if item.get("fact")
        }
        for item in incoming:
            key = normalize_text(item.get("fact", ""))
            if key:
                merged[key] = item
        return list(merged.values())

    def merge_discovered_aliases(
        self,
        state: dict[str, Any],
        discovered: dict[str, list[str]],
    ) -> bool:
        changed = False
        roles = state.setdefault("case_roles", {})
        for source_type in SOURCE_TYPES:
            key = source_alias_key(source_type)
            current = roles.setdefault(key, [])
            current_norm = {
                normalize_text(alias) for alias in current
            }
            for alias in discovered.get(source_type, []):
                normalized = normalize_text(alias)
                if normalized and normalized not in current_norm:
                    current.append(alias)
                    current_norm.add(normalized)
                    changed = True
        return changed

    def expand_chunk(
        self,
        state: dict[str, Any],
        chunk_id: str,
    ) -> None:
        item = state["archive"][chunk_id]
        analysis = self.analyzer.analyze_chunk(
            state["clean_query"],
            chunk_id,
            item["text"],
            state["generation"] + 1,
            state["case_roles"],
        )

        attributions = analysis["source_attributions"]
        observed_sources = observed_source_types(attributions)
        mentioned_sources = mentioned_source_types(attributions)

        if len(observed_sources) == 1:
            observed_source = observed_sources[0]
            for candidate in analysis["candidates"]:
                if candidate.metadata.get("boundary_query"):
                    candidate.metadata.update({
                        "target_source_type": observed_source,
                        "source_followup": True,
                        "source_priority": 0.82,
                        "gap_priority": max(
                            0.82,
                            clip01(
                                candidate.metadata.get(
                                    "gap_priority",
                                    0.0,
                                )
                            ),
                        ),
                    })

        state["pool"].extend(analysis["candidates"])

        aliases_changed = self.merge_discovered_aliases(
            state,
            analysis["discovered_aliases"],
        )
        if aliases_changed:
            state["pool"].extend(
                self.analyzer.generate_entity_source_candidates(
                    state["case_roles"],
                    generation=state["generation"] + 1,
                )
            )

        local_facts = [
            {
                "fact": fact,
                "evidence_ids": [chunk_id],
                "confidence": 0.75,
            }
            for fact in analysis["atomic_facts"]
        ]
        state["evidence_map"]["atomic_facts"] = self.merge_fact_list(
            state["evidence_map"].get("atomic_facts", []),
            local_facts,
        )

        item["anchors"] = analysis["anchors"]
        item["local_facts"] = analysis["atomic_facts"]
        item["source_attributions"] = attributions
        item["observed_source_types"] = observed_sources
        item["mentioned_source_types"] = mentioned_sources

        # Legacy field kept for readers of old result format. It now means
        # observed content source, never retrieval target.
        item["source_types"] = observed_sources

    def refresh_evidence_map(self, state: dict[str, Any]) -> None:
        if not state["archive"]:
            return

        evidence_map = self.analyzer.analyze_evidence_map(
            state["clean_query"],
            state["archive"],
            state.get("evidence_map"),
        )
        evidence_map["atomic_facts"] = self.merge_fact_list(
            state["evidence_map"].get("atomic_facts", []),
            evidence_map.get("atomic_facts", []),
        )
        state["evidence_map"] = evidence_map
        state["pool"].extend(
            self.analyzer.candidates_from_evidence_map(
                evidence_map,
                state["generation"] + 1,
            )
        )

        source_coverage = self.analyzer.analyze_source_coverage(
            state["clean_query"],
            state["archive"],
            state["case_roles"],
            state.get("source_coverage"),
        )
        state["source_coverage"] = source_coverage
        state["pool"].extend(
            self.analyzer.candidates_from_source_coverage(
                source_coverage,
                state["generation"] + 1,
                state.get("source_query_attempts"),
            )
        )
        state["pool"].extend(
            self.analyzer.generate_entity_source_candidates(
                state["case_roles"],
                generation=state["generation"] + 1,
            )
        )

        state["new_chunks_since_refresh"] = 0
        state["queries_since_refresh"] = 0


    def source_is_covered(
        self,
        state: dict[str, Any],
        source_type: str,
    ) -> bool:
        item = state.get("source_coverage", {}).get(source_type, {})
        return str(item.get("status", "NOT_FOUND")).upper() in {
            "PARTIAL",
            "SUBSTANTIVE",
        }

    def next_mandatory_source(
        self,
        state: dict[str, Any],
        total_calls: int,
    ) -> Optional[str]:
        if total_calls < self.config.source_probe_start_call:
            return None

        for source_type in SOURCE_TYPES:
            if self.source_is_covered(state, source_type):
                continue
            attempts = int(
                state.get("source_probe_attempts", {}).get(
                    source_type,
                    0,
                )
            )
            if (
                attempts
                < self.config.mandatory_probe_attempts_per_source
            ):
                return source_type
        return None

    def mandatory_source_work_remaining(
        self,
        state: dict[str, Any],
    ) -> bool:
        for source_type in SOURCE_TYPES:
            if self.source_is_covered(state, source_type):
                continue
            attempts = int(
                state.get("source_probe_attempts", {}).get(
                    source_type,
                    0,
                )
            )
            if (
                attempts
                < self.config.mandatory_probe_attempts_per_source
            ):
                return True
        return False

    def prune_exhausted_source_candidates(
        self,
        state: dict[str, Any],
    ) -> None:
        for candidate in list(state["pool"].items):
            source_type = candidate.metadata.get("target_source_type")
            if source_type not in SOURCE_TYPES:
                continue
            attempts = int(
                state.get("source_query_attempts", {}).get(
                    source_type,
                    0,
                )
            )
            if attempts >= self.config.max_source_queries_per_type:
                state["pool"].remove(candidate)

    def learning_reward(self, text: str) -> float:
        word_count = len(text.split())
        if text.startswith(("A_description:", "B_description:")):
            return self.config.metadata_reward
        if word_count < 30:
            return self.config.short_chunk_reward
        return 1.0

    def total_network_calls(self, state: dict[str, Any]) -> int:
        return int(state.get("new_network_calls", 0))

    def high_factual_gap_exists(
        self,
        state: dict[str, Any],
    ) -> bool:
        return any(
            clip01(gap.get("priority", 0.0))
            >= self.config.high_priority_gap_threshold
            for gap in state.get("evidence_map", {}).get(
                "unresolved_factual_gaps",
                [],
            )
        )

    def active_source_gap_exists(
        self,
        state: dict[str, Any],
    ) -> bool:
        for source_type, item in state.get(
            "source_coverage",
            {},
        ).items():
            attempts = int(
                state.get("source_query_attempts", {}).get(
                    source_type,
                    0,
                )
            )
            if attempts >= self.config.max_source_queries_per_type:
                continue

            status = str(
                item.get("status", "NOT_FOUND")
            ).upper()
            has_missing_detail = bool(
                item.get("missing_aspects")
                or item.get("query_hooks")
            )
            if status != "SUBSTANTIVE" or has_missing_detail:
                return True
        return False

    def recent_yield(
        self,
        state: dict[str, Any],
        window: Optional[int] = None,
    ) -> dict[str, float]:
        window = window or self.config.recent_window
        recent = state.get("history", [])[-window:]
        if not recent:
            return {
                "observations": 0,
                "new_chunks": 0,
                "new_chunk_rate": 0.0,
                "duplicates": 0,
                "failures": 0,
            }

        new_chunks = sum(
            item.get("reward", 0) > 0
            for item in recent
        )
        duplicates = sum(
            bool(item.get("duplicate", False))
            for item in recent
        )
        failures = sum(
            not bool(item.get("success", False))
            for item in recent
        )
        return {
            "observations": len(recent),
            "new_chunks": int(new_chunks),
            "new_chunk_rate": new_chunks / len(recent),
            "duplicates": int(duplicates),
            "failures": int(failures),
        }

    def next_budget_stage(
        self,
        current_target: int,
    ) -> Optional[int]:
        for target in self.config.budget_stages():
            if target > current_target:
                return target
        return None

    def budget_extension_decision(
        self,
        state: dict[str, Any],
        best_utility: Optional[float],
    ) -> tuple[bool, str, Optional[int], dict[str, Any]]:
        total_calls = self.total_network_calls(state)
        current_target = int(state["target_budget"])
        next_target = self.next_budget_stage(current_target)

        signals: dict[str, Any] = {
            **self.recent_yield(state),
            "current_target": current_target,
            "next_target": next_target,
            "total_calls": total_calls,
            "best_utility": (
                None
                if best_utility is None
                else float(best_utility)
            ),
            "mandatory_source_remaining": (
                self.mandatory_source_work_remaining(state)
            ),
            "high_factual_gap": self.high_factual_gap_exists(state),
            "active_source_gap": self.active_source_gap_exists(state),
            "broad_search_needed": bool(
                state.get("evidence_map", {}).get(
                    "broad_search_needed",
                    True,
                )
            ),
        }

        if next_target is None:
            return False, "already_at_hard_budget", None, signals

        recent_new = int(signals["new_chunks"])
        utility = (
            float(best_utility)
            if best_utility is not None
            else -1.0
        )
        gap_or_source = bool(
            signals["high_factual_gap"]
            or signals["active_source_gap"]
        )
        mandatory = bool(signals["mandatory_source_remaining"])
        broad_needed = bool(signals["broad_search_needed"])

        # Stage 24 -> 32: continue when retrieval still yields chunks,
        # or source probes are not yet completed.
        if next_target <= self.config.intermediate_total_budget:
            allowed = recent_new >= 2 or mandatory
            reason = (
                "base_to_intermediate:"
                f" recent_new={recent_new}, mandatory={mandatory}"
            )
            return allowed, reason, next_target, signals

        # Stage 32 -> 40: this is the calibrated P90 region.
        # Require both usable yield and a reason to continue searching.
        if next_target <= self.config.soft_total_budget:
            allowed = (
                recent_new >= 2
                and utility >= self.config.minimum_candidate_utility
                and (
                    gap_or_source
                    or broad_needed
                    or recent_new >= 3
                )
            )
            reason = (
                "intermediate_to_soft:"
                f" recent_new={recent_new}, utility={utility:.3f},"
                f" gap_or_source={gap_or_source}, broad={broad_needed}"
            )
            return allowed, reason, next_target, signals

        # Stage 40 -> 48: only continue with strong recent discovery,
        # or strong yield plus a still-unresolved evidence source/gap.
        if next_target <= self.config.late_total_budget:
            allowed = (
                utility >= 0.18
                and (
                    recent_new >= 3
                    or (recent_new >= 2 and gap_or_source)
                )
            )
            reason = (
                "soft_to_late:"
                f" recent_new={recent_new}, utility={utility:.3f},"
                f" gap_or_source={gap_or_source}"
            )
            return allowed, reason, next_target, signals

        # Stage 48 -> 52: final reserve. It is intentionally strict.
        allowed = (
            utility >= 0.22
            and recent_new >= 2
            and (
                gap_or_source
                or recent_new >= 3
            )
        )
        reason = (
            "late_to_hard:"
            f" recent_new={recent_new}, utility={utility:.3f},"
            f" gap_or_source={gap_or_source}"
        )
        return allowed, reason, next_target, signals

    def maybe_extend_budget(
        self,
        state: dict[str, Any],
        best_utility: Optional[float],
    ) -> bool:
        total_calls = self.total_network_calls(state)
        if total_calls < int(state["target_budget"]):
            return True

        allowed, reason, next_target, signals = (
            self.budget_extension_decision(
                state,
                best_utility,
            )
        )

        if not allowed or next_target is None:
            state["budget_history"].append({
                "from": int(state["target_budget"]),
                "to": int(state["target_budget"]),
                "reason": "extension_denied: " + reason,
                "network_calls": total_calls,
                "unique_chunks": len(state.get("archive", {})),
                "signals": signals,
            })
            return False

        old_target = int(state["target_budget"])
        state["target_budget"] = int(next_target)
        state["budget_history"].append({
            "from": old_target,
            "to": int(next_target),
            "reason": reason,
            "network_calls": total_calls,
            "unique_chunks": len(state.get("archive", {})),
            "signals": signals,
        })
        print(
            f"Mở rộng budget {old_target} → {next_target} | {reason}"
        )
        return True

    def tiny_case_saturated(
        self,
        state: dict[str, Any],
        best_utility: Optional[float],
    ) -> bool:
        total_calls = self.total_network_calls(state)
        if total_calls < self.config.tiny_case_min_calls:
            return False
        if len(state.get("archive", {})) > self.config.tiny_case_max_unique_chunks:
            return False
        if self.mandatory_source_work_remaining(state):
            return False

        recent = state.get("history", [])[
            -self.config.tiny_case_no_new_window:
        ]
        if len(recent) < self.config.tiny_case_no_new_window:
            return False
        if any(item.get("reward", 0) > 0 for item in recent):
            return False

        utility = (
            float(best_utility)
            if best_utility is not None
            else -1.0
        )
        return utility < self.config.tiny_case_max_utility

    def retrieval_saturated(
        self,
        state: dict[str, Any],
        best_utility: Optional[float],
    ) -> bool:
        if self.mandatory_source_work_remaining(state):
            return False

        recent = state.get("history", [])[
            -self.config.recent_window:
        ]
        if len(recent) < self.config.recent_window:
            return False

        recent_new = sum(
            item.get("reward", 0) > 0
            for item in recent
        )
        if recent_new > self.config.saturation_max_new_chunks:
            return False

        high_gap_exists = self.high_factual_gap_exists(state)
        source_gap_exists = self.active_source_gap_exists(state)
        utility = (
            float(best_utility)
            if best_utility is not None
            else -1.0
        )

        # Keep searching when a real evidence gap remains and a viable
        # candidate still exists.
        if (
            (high_gap_exists or source_gap_exists)
            and utility >= self.config.minimum_candidate_utility
        ):
            return False

        return True

    def run_case(
        self,
        case: dict[str, Any],
        fresh_start: bool = True,
    ) -> dict[str, Any]:
        case_id = str(case.get("case_id", "")).strip()
        if not case_id:
            raise ValueError("Thiếu case_id.")

        # The notebook calls fresh_start=True. This removes only a checkpoint
        # inside the current fresh RUN_DIR; no prior result is imported.
        if fresh_start:
            self.checkpoint_path(case_id).unlink(missing_ok=True)
            state = self.initialize_state(case)
            print("Khởi tạo hoàn toàn từ đầu: 0 chunk, 0 query, 0 API call.")
        else:
            state = self.load_checkpoint(case_id)
            if state is None:
                state = self.initialize_state(case)
                print("Không có checkpoint trong run hiện tại; bắt đầu từ 0.")
            else:
                print(
                    "Resume checkpoint của chính run hiện tại:",
                    len(state["archive"]),
                    "chunks,",
                    len(state["used_queries"]),
                    "queries.",
                )

        state.setdefault(
            "budget_history",
            [{
                "from": 0,
                "to": int(
                    state.get(
                        "target_budget",
                        self.config.base_total_budget,
                    )
                ),
                "reason": "resume_or_initial_budget",
                "network_calls": int(
                    state.get("new_network_calls", 0)
                ),
                "unique_chunks": len(state.get("archive", {})),
            }],
        )
        state.setdefault("last_best_candidate_utility", None)
        state.setdefault(
            "source_query_attempts",
            {source_type: 0 for source_type in SOURCE_TYPES},
        )
        state.setdefault(
            "source_probe_attempts",
            {source_type: 0 for source_type in SOURCE_TYPES},
        )
        state.setdefault(
            "source_followup_attempts",
            {source_type: 0 for source_type in SOURCE_TYPES},
        )
        state.setdefault("source_probes_added", False)
        state.setdefault("source_bootstrap_refreshed", False)
        state.setdefault("source_target_stats", {
            source_type: {
                "attempts": 0,
                "target_hits": 0,
                "off_target_new_chunks": 0,
                "duplicates": 0,
                "failures": 0,
            }
            for source_type in SOURCE_TYPES
        })
        for key in SOURCE_ALIAS_KEYS.values():
            state.setdefault("case_roles", {}).setdefault(key, [])

        seen_index = SeenIndex(self.runtime)
        seen_index.rebuild(state["archive"])

        persisted_network_calls = int(
            state.get("new_network_calls", 0)
        )
        client_start = self.client.network_call_count

        while True:
            state["new_network_calls"] = (
                persisted_network_calls
                + self.client.network_call_count
                - client_start
            )
            total_calls = self.total_network_calls(state)

            if total_calls >= self.config.hard_total_budget:
                state["stop_reason"] = "hard_budget"
                break

            if (
                total_calls >= self.config.source_probe_start_call
                and not state.get("source_probes_added", False)
            ):
                state["pool"].extend(
                    self.analyzer.generate_source_probe_candidates(
                        state["clean_query"],
                        state["case_roles"],
                        generation=state["generation"] + 1,
                    )
                )
                state["source_probes_added"] = True

            # Evaluate what the two broad calls already found before mandatory
            # source probes, so a source with substantive evidence is not
            # probed redundantly.
            if (
                total_calls >= self.config.source_probe_start_call
                and state["archive"]
                and not state.get(
                    "source_bootstrap_refreshed",
                    False,
                )
            ):
                self.refresh_evidence_map(state)
                state["source_bootstrap_refreshed"] = True

            if (
                state["new_chunks_since_refresh"]
                >= self.config.evidence_refresh_every_new_chunks
                or state["queries_since_refresh"]
                >= self.config.evidence_refresh_every_queries
            ):
                self.refresh_evidence_map(state)

            self.prune_exhausted_source_candidates(state)

            candidate = None
            candidate_metrics = None
            mandatory_source = self.next_mandatory_source(
                state,
                total_calls,
            )
            if mandatory_source is not None:
                candidate, candidate_metrics = (
                    pop_mandatory_source_candidate(
                        pool=state["pool"],
                        source_type=mandatory_source,
                        clean_query=state["clean_query"],
                        runtime=self.runtime,
                        seen_index=seen_index,
                        used_queries=state["used_queries"],
                        family_stats=state["family_stats"],
                        history=state["history"],
                        evidence_map=state["evidence_map"],
                        config=self.config,
                    )
                )

            if candidate is None:
                candidate, candidate_metrics = pop_best_candidate(
                    pool=state["pool"],
                    clean_query=state["clean_query"],
                    runtime=self.runtime,
                    seen_index=seen_index,
                    used_queries=state["used_queries"],
                    family_stats=state["family_stats"],
                    history=state["history"],
                    evidence_map=state["evidence_map"],
                    config=self.config,
                )

            best_utility = (
                candidate_metrics["utility"]
                if candidate_metrics else None
            )

            if candidate is None:
                self.refresh_evidence_map(state)
                candidate, candidate_metrics = pop_best_candidate(
                    pool=state["pool"],
                    clean_query=state["clean_query"],
                    runtime=self.runtime,
                    seen_index=seen_index,
                    used_queries=state["used_queries"],
                    family_stats=state["family_stats"],
                    history=state["history"],
                    evidence_map=state["evidence_map"],
                    config=self.config,
                )
                if candidate is None:
                    state["stop_reason"] = "candidate_pool_exhausted"
                    break
                best_utility = candidate_metrics["utility"]

            state["last_best_candidate_utility"] = best_utility

            if self.tiny_case_saturated(state, best_utility):
                state["stop_reason"] = "tiny_case_saturated"
                break

            if self.retrieval_saturated(state, best_utility):
                state["stop_reason"] = "evidence_retrieval_saturated"
                break

            if total_calls >= int(state["target_budget"]):
                extended = self.maybe_extend_budget(
                    state,
                    best_utility,
                )
                if not extended:
                    state["stop_reason"] = "dynamic_budget_not_extended"
                    break

            if (
                candidate_metrics is not None
                and candidate_metrics["utility"]
                < self.config.minimum_candidate_utility
                and len(state["history"]) >= self.config.recent_window
            ):
                state["stop_reason"] = "low_candidate_utility"
                break

            remaining = self.config.hard_total_budget - total_calls
            print(
                f"\nTotal call {total_calls + 1} | "
                f"track={candidate.track} | "
                f"family={candidate.family} | "
                f"utility={candidate_metrics['utility']:.3f}\n"
                f"query: {candidate.query}"
            )

            call = self.client.retrieve(
                state["case_id"],
                candidate.query,
                max_network_attempts=remaining,
            )
            state["new_network_calls"] = (
                persisted_network_calls
                + self.client.network_call_count
                - client_start
            )
            state["used_queries"].append(candidate.query)
            state["queries_since_refresh"] += 1

            target_source_type = candidate.metadata.get(
                "target_source_type"
            )
            if target_source_type in SOURCE_TYPES:
                state["source_query_attempts"][
                    target_source_type
                ] += 1
                state["source_target_stats"][
                    target_source_type
                ]["attempts"] += 1
                if candidate.metadata.get("source_probe"):
                    state["source_probe_attempts"][
                        target_source_type
                    ] += 1
                if candidate.metadata.get("source_followup"):
                    state["source_followup_attempts"][
                        target_source_type
                    ] += 1

            family = state["family_stats"].setdefault(
                candidate.family,
                FamilyStats(),
            )
            family.calls += 1

            reward = 0.0
            duplicate = False
            candidates_added = 0
            returned_observed_sources: list[str] = []
            target_source_hit = False

            if not call.success or call.hit is None:
                family.failures += 1
                if target_source_type in SOURCE_TYPES:
                    state["source_target_stats"][
                        target_source_type
                    ]["failures"] += 1
            elif call.hit.chunk_id in state["archive"]:
                duplicate = True
                family.duplicates += 1
                existing_chunk = state["archive"][call.hit.chunk_id]
                returned_observed_sources = list(
                    existing_chunk.get(
                        "observed_source_types",
                        existing_chunk.get("source_types", []),
                    )
                )
                target_source_hit = (
                    target_source_type in returned_observed_sources
                )
                if target_source_type in SOURCE_TYPES:
                    stats = state["source_target_stats"][
                        target_source_type
                    ]
                    stats["duplicates"] += 1
                    if target_source_hit:
                        stats["target_hits"] += 1
                mutations = self.analyzer.mutate_duplicate_query(
                    candidate,
                    call.hit.text,
                    state["clean_query"],
                    state["generation"] + 1,
                )
                candidates_added += state["pool"].extend(mutations)
            else:
                reward = self.learning_reward(call.hit.text)
                family.new_chunks += 1
                family.learning_reward_sum += reward

                state["archive"][call.hit.chunk_id] = {
                    "chunk_id": call.hit.chunk_id,
                    "text": call.hit.text,
                    "api_score": call.hit.score,
                    "first_found_by": candidate.query,
                    "first_family": candidate.family,
                    "first_target_source_type": target_source_type,
                    "generation": state["generation"],
                    "found_by": [{
                        "query": candidate.query,
                        "family": candidate.family,
                        "track": candidate.track,
                        "target_source_type": target_source_type,
                        "api_score": call.hit.score,
                    }],
                }

                self.expand_chunk(state, call.hit.chunk_id)
                new_chunk = state["archive"][call.hit.chunk_id]
                returned_observed_sources = list(
                    new_chunk.get("observed_source_types", [])
                )
                target_source_hit = (
                    target_source_type in returned_observed_sources
                )
                new_chunk["target_source_hit"] = target_source_hit
                new_chunk["retrieval_target_source_type"] = (
                    target_source_type
                )

                if target_source_type in SOURCE_TYPES:
                    stats = state["source_target_stats"][
                        target_source_type
                    ]
                    if target_source_hit:
                        stats["target_hits"] += 1
                    else:
                        stats["off_target_new_chunks"] += 1

                state["new_chunks_since_refresh"] += 1
                state["generation"] += 1
                seen_index.rebuild(state["archive"])

                if target_source_type in SOURCE_TYPES:
                    self.refresh_evidence_map(state)

            log = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "query": candidate.query,
                "family": candidate.family,
                "track": candidate.track,
                "source_chunk_id": candidate.source_chunk_id,
                "source_topic_id": candidate.source_topic_id,
                "source_gap_id": candidate.source_gap_id,
                "target_source_type": target_source_type,
                "observed_source_types": returned_observed_sources,
                "target_source_hit": target_source_hit,
                "source_probe": bool(
                    candidate.metadata.get("source_probe")
                ),
                "source_followup": bool(
                    candidate.metadata.get("source_followup")
                ),
                "parent_query": candidate.parent_query,
                "rationale": candidate.rationale,
                "mutation_count": candidate.mutation_count,
                "candidate_metrics": candidate_metrics,
                "success": call.success,
                "status_code": call.status_code,
                "error": call.error,
                "attempts": call.attempts,
                "network_attempts": 0 if call.from_cache else call.attempts,
                "from_cache": call.from_cache,
                "returned_chunk_id": (
                    call.hit.chunk_id if call.hit else None
                ),
                "api_score": call.hit.score if call.hit else None,
                "is_duplicate": duplicate,
                "reward": reward,
                "unique_chunks_after": len(state["archive"]),
                "pool_size_after": len(state["pool"]),
                "total_network_calls_after": self.total_network_calls(state),
                "new_candidates_added": candidates_added,
            }
            state["query_logs"].append(log)
            state["history"].append({
                "reward": reward,
                "track": candidate.track,
                "family": candidate.family,
                "duplicate": duplicate,
                "success": call.success,
            })

            print(
                "→",
                "NEW" if reward > 0 else (
                    "DUPLICATE" if duplicate else "FAIL"
                ),
                "| unique=", len(state["archive"]),
                "| pool=", len(state["pool"]),
            )

            self.save_checkpoint(state)

            gc.collect()
            torch.cuda.empty_cache()

        # Final evidence map after stopping.
        self.refresh_evidence_map(state)
        self.save_checkpoint(state)

        total_calls = self.total_network_calls(state)
        new_chunks = sum(
            log.get("reward", 0) > 0
            for log in state["query_logs"]
            if not log.get("from_cache", False)
        )
        duplicates = sum(
            bool(log.get("is_duplicate", False))
            for log in state["query_logs"]
        )
        failures = sum(
            not bool(log.get("success", False))
            for log in state["query_logs"]
        )

        all_chunks = list(state["archive"].values())
        result = {
            "case_id": state["case_id"],
            "case_query": state["case_query"],
            "clean_query": state["clean_query"],

            "all_unique_chunks": all_chunks,
            "case_evidence": list(state["archive"].keys()),

            "evidence_map": state["evidence_map"],
            "atomic_facts": state["evidence_map"].get(
                "atomic_facts", []
            ),
            "evidence_topics": state["evidence_map"].get(
                "evidence_topics", []
            ),
            "unresolved_factual_gaps": state["evidence_map"].get(
                "unresolved_factual_gaps", []
            ),
            "case_roles": state.get("case_roles", {}),
            "source_coverage": state.get(
                "source_coverage",
                empty_source_coverage(),
            ),
            "source_query_attempts": state.get(
                "source_query_attempts",
                {},
            ),
            "source_probe_attempts": state.get(
                "source_probe_attempts",
                {},
            ),
            "source_followup_attempts": state.get(
                "source_followup_attempts",
                {},
            ),
            "source_target_stats": state.get(
                "source_target_stats",
                {},
            ),
            "budget_history": state.get("budget_history", []),
            "chunking_assumption": {
                "chunk_size_words": 100,
                "consecutive": True,
                "overlap": 0,
            },

            # Handoff contains only facts/evidence. No law and no decision.
            "handoff_for_later_stages": {
                "case_id": state["case_id"],
                "clean_query": state["clean_query"],
                "case_evidence": list(state["archive"].keys()),
                "atomic_facts": state["evidence_map"].get(
                    "atomic_facts", []
                ),
                "evidence_topics": state["evidence_map"].get(
                    "evidence_topics", []
                ),
                "unresolved_factual_gaps": state["evidence_map"].get(
                    "unresolved_factual_gaps", []
                ),
                "case_roles": state.get("case_roles", {}),
                "source_coverage": state.get(
                    "source_coverage",
                    empty_source_coverage(),
                ),
                "chunking_assumption": {
                    "chunk_size_words": 100,
                    "consecutive": True,
                    "overlap": 0,
                },
            },

            "query_logs": state["query_logs"],
            "family_stats": family_stats_to_dict(
                state["family_stats"]
            ),
            "metrics": {
                "network_calls": total_calls,
                "unique_chunks": len(state["archive"]),
                "new_chunks": new_chunks,
                "unique_chunks_per_call": (
                    len(state["archive"]) / total_calls
                    if total_calls else 0.0
                ),
                "duplicates": duplicates,
                "failures": failures,
                "logical_queries": len(state["query_logs"]),
                "remaining_candidates": len(state["pool"]),
                "target_budget": state["target_budget"],
                "budget_stages": list(
                    self.config.budget_stages()
                ),
                "hard_budget": self.config.hard_total_budget,
                "budget_extensions": sum(
                    item.get("to", 0) > item.get("from", 0)
                    for item in state.get("budget_history", [])
                ),
                "boundary_queries_called": sum(
                    str(log.get("family", "")).startswith(
                        "boundary_"
                    )
                    for log in state["query_logs"]
                ),
                "evidence_topic_count": len(
                    state["evidence_map"].get(
                        "evidence_topics", []
                    )
                ),
                "high_priority_gap_count": sum(
                    clip01(gap.get("priority", 0.0))
                    >= self.config.high_priority_gap_threshold
                    for gap in state["evidence_map"].get(
                        "unresolved_factual_gaps", []
                    )
                ),
                "source_statuses": {
                    source_type: item.get("status", "NOT_FOUND")
                    for source_type, item in state.get(
                        "source_coverage",
                        {},
                    ).items()
                },
                "source_query_attempts": state.get(
                    "source_query_attempts",
                    {},
                ),
                "source_probe_attempts": state.get(
                    "source_probe_attempts",
                    {},
                ),
                "source_followup_attempts": state.get(
                    "source_followup_attempts",
                    {},
                ),
                "source_target_stats": state.get(
                    "source_target_stats",
                    {},
                ),
                "source_target_hit_rate": {
                    source_type: (
                        stats.get("target_hits", 0)
                        / stats.get("attempts", 1)
                        if stats.get("attempts", 0)
                        else 0.0
                    )
                    for source_type, stats in state.get(
                        "source_target_stats",
                        {},
                    ).items()
                },
                "stop_reason": state["stop_reason"],
            },

            # Explicit guardrail for downstream code.
            "prediction": None,
            "law_evidence": [],
            "stage": "ENTITY_AWARE_SOURCE_ATTRIBUTION_CASE_EVIDENCE_RETRIEVAL_ONLY",
        }

        result_path = (
            self.run_dir
            / f"{state['case_id']}_entity_aware_source_evidence_result.json"
        )
        atomic_write_json(result_path, result)

        handoff_path = (
            self.run_dir
            / f"{state['case_id']}_entity_aware_law_handoff.json"
        )
        atomic_write_json(
            handoff_path,
            result["handoff_for_later_stages"],
        )

        print("\nĐã lưu:")
        print(result_path)
        print(handoff_path)
        return result


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def load_cases(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".jsonl":
        values = []
        with path.open("r", encoding="utf-8-sig") as file:
            for line in file:
                if line.strip():
                    values.append(json.loads(line))
    else:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(data, list):
            values = data
        elif isinstance(data, dict) and isinstance(data.get("cases"), list):
            values = data["cases"]
        elif isinstance(data, dict) and "case_id" in data:
            values = [data]
        else:
            raise ValueError("Không nhận diện được format input.")

    output = []
    seen = set()
    for item in values:
        case_id = str(item.get("case_id", "")).strip()
        case_query = str(item.get("case_query", "")).strip()
        if case_id and case_query and case_id not in seen:
            seen.add(case_id)
            output.append({
                **item,
                "case_id": case_id,
                "case_query": case_query,
            })
    return output


def completed_case_ids(path: Path) -> set[str]:
    output = set()
    if not path.exists():
        return output

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("case_id"):
                output.add(str(item["case_id"]))
    return output


def run_batch(
    crawler: CalibratedSourceAwareEvidenceCrawler,
    input_path: Path,
    results_jsonl_path: Path,
    failures_jsonl_path: Path,
    max_cases: Optional[int] = 3,
) -> None:
    cases = load_cases(input_path)
    done = completed_case_ids(results_jsonl_path)
    pending = [case for case in cases if case["case_id"] not in done]

    if max_cases is not None:
        pending = pending[:max_cases]

    print("Input:", len(cases))
    print("Đã hoàn thành:", len(done))
    print("Chạy lần này:", len(pending))

    for index, case in enumerate(pending, start=1):
        print(f"\n===== [{index}/{len(pending)}] {case['case_id']} =====")
        try:
            result = crawler.run_case(
                case,
                fresh_start=False,
            )
            append_jsonl(results_jsonl_path, {
                "case_id": result["case_id"],
                "case_evidence": result["case_evidence"],
                "metrics": result["metrics"],
                "stage": result["stage"],
            })
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            failure = {
                "case_id": case["case_id"],
                "error_type": type(exc).__name__,
                "error": str(exc),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            append_jsonl(failures_jsonl_path, failure)
            print("ERROR:", failure)

        gc.collect()
        torch.cuda.empty_cache()
