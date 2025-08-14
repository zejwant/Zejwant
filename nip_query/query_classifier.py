"""
nip_query.query_classifier

Enterprise-scale NL-to-SQL classifier.

Responsibilities
----------------
- Detect query type(s): aggregation, filtering, join, time_series, window, groupby, ranking, text_search.
- Score complexity: normalized [0..1] + label {'low','medium','high'}.
- Sensitivity detection: PII, financial, compliance-sensitive.
- Combine rule-based heuristics with optional ML (spaCy / transformers).
- Produce a structured dict suitable for routing & planning.
- Structured logging and robust error handling.
- Pluggable ML backends via dependency injection.

Public API
----------
- classify_query(user_input: str, context: Optional[Mapping], options: Optional[Mapping]) -> Dict[str, Any]
  (alias: classify(...))

Design
------
- Avoids heavy imports at module import time; models are loaded lazily on first use.
- Unit-test friendly: pass custom `nlp_backend` / `pii_detector` / `intent_model` in options.
- Does not depend on DB schema; purely NL analysis.

Example
-------
    result = classify_query("Show average revenue by region for Q1 vs Q2")
    print(result["types"], result["complexity"], result["safety"])
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Mapping, Optional, Protocol, Tuple, runtime_checkable

import re
import math

# Lightweight structured logger from package __init__.py
try:
    from nip_query import get_logger  # type: ignore
except Exception:  # pragma: no cover - test fallback
    import logging

    def get_logger():  # type: ignore
        logger = logging.getLogger("nip_query.query_classifier_fallback")
        if not logger.handlers:
            handler = logging.StreamHandler()
            logger.addHandler(handler)
            logger.setLevel("INFO")
        return logger


# --------------------------
# Protocols (pluggable ML)
# --------------------------

@runtime_checkable
class NLPBackend(Protocol):
    """Minimal NLP backend interface (spaCy-like)."""
    def tokenize(self, text: str) -> List[str]: ...
    def ents(self, text: str) -> List[Tuple[str, str]]: ...  # (text, label)
    def pos(self, text: str) -> List[Tuple[str, str]]: ...   # (token, pos)


@runtime_checkable
class IntentModel(Protocol):
    """Optional intent classifier (transformers-like)."""
    def predict_labels(self, text: str) -> List[str]: ...


@runtime_checkable
class PIIDetector(Protocol):
    """Optional PII detector."""
    def detect(self, text: str) -> Dict[str, Any]: ...


# --------------------------
# Data structures
# --------------------------

AGG_KEYWORDS = {
    "sum", "total", "avg", "average", "count", "min", "max", "median",
    "std", "variance", "percentile", "quantile", "distribution",
}
GROUPBY_KEYWORDS = {"by", "per", "group", "bucket", "bin"}
FILTER_KEYWORDS = {
    "where", "with", "having", "only", "top", "greater than", "less than",
    "between", "equals", "equal to", "after", "before", "last", "this", "previous",
}
JOIN_KEYWORDS = {"join", "vs", "versus", "compare", "union", "merge", "combine", "lookup"}
TS_KEYWORDS = {"time", "date", "daily", "weekly", "monthly", "quarterly", "yearly", "trend", "moving average", "rolling"}
WINDOW_KEYWORDS = {"over", "partition", "window", "lag", "lead", "rank", "dense_rank"}
RANKING_KEYWORDS = {"top", "bottom", "rank", "most", "least"}
TEXT_SEARCH_KEYWORDS = {"contains", "like", "match", "regex", "search"}

FINANCIAL_KEYWORDS = {
    "revenue", "profit", "margin", "arpu", "mrr", "arr", "invoice", "balance sheet",
    "p&l", "ledger", "capex", "opex", "payable", "receivable"
}
COMPLIANCE_KEYWORDS = {"gdpr", "hipaa", "sox", "pci", "iso", "soc2", "audit", "compliance"}

PII_PATTERNS = [
    (r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b", "SSN"),
    (r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", "CREDIT_CARD"),
    (r"\b[A-Z]{5}\d{4}[A-Z]\b", "PAN"),  # India PAN format
    (r"\b\d{12}\b", "AADHAAR"),
    (r"\b\d{10}\b", "PHONE10"),
    (r"\b(?:\w|\.|-)+@(?:\w|-)+\.\w+\b", "EMAIL"),
]

# --------------------------
# Light NLP backend (fallback)
# --------------------------

class _RegexNLP:
    """Fallback NLP backend using simple regex/tokenization (no heavy deps)."""
    _word_re = re.compile(r"[A-Za-z_]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?")

    def tokenize(self, text: str) -> List[str]:
        return [m.group(0).lower() for m in self._word_re.finditer(text)]

    def ents(self, text: str) -> List[Tuple[str, str]]:
        # Minimal named-entity heuristic: detect DATE-like tokens
        ents = []
        # detect month, quarter, year patterns
        if re.search(r"\b(q[1-4]|quarter|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|20\d{2})\b", text, re.I):
            ents.append(("TIME_EXPR", "DATE"))
        return ents

    def pos(self, text: str) -> List[Tuple[str, str]]:
        return [(t, "X") for t in self.tokenize(text)]


# Lazy global singletons (loaded on demand)
_SPACY_BACKEND: Optional[NLPBackend] = None
_XFORMER_INTENT: Optional[IntentModel] = None


def _load_spacy_backend(lang: str = "en_core_web_sm") -> Optional[NLPBackend]:
    global _SPACY_BACKEND
    if _SPACY_BACKEND is not None:
        return _SPACY_BACKEND
    try:  # pragma: no cover (heavy dep)
        import spacy  # type: ignore
        nlp = spacy.load(lang)

        class _SpacyBackend(NLPBackend):
            def tokenize(self, text: str) -> List[str]:
                doc = nlp(text)
                return [t.text.lower() for t in doc]

            def ents(self, text: str) -> List[Tuple[str, str]]:
                doc = nlp(text)
                return [(ent.text, ent.label_) for ent in doc.ents]

            def pos(self, text: str) -> List[Tuple[str, str]]:
                doc = nlp(text)
                return [(t.text, t.pos_) for t in doc]

        _SPACY_BACKEND = _SpacyBackend()
        return _SPACY_BACKEND
    except Exception:
        return None


def _load_transformer_intent(model_name: str = "typeform/distilbert-base-uncased-mnli") -> Optional[IntentModel]:
    global _XFORMER_INTENT
    if _XFORMER_INTENT is not None:
        return _XFORMER_INTENT
    try:  # pragma: no cover (heavy dep)
        from transformers import pipeline  # type: ignore
        clf = pipeline("zero-shot-classification", model=model_name)

        class _ZSLIntent(IntentModel):
            _labels = [
                "aggregation", "filtering", "join", "time_series", "window", "groupby",
                "ranking", "text_search"
            ]

            def predict_labels(self, text: str) -> List[str]:
                out = clf(text, candidate_labels=self._labels, multi_label=True)
                pairs = list(zip(out["labels"], out["scores"]))
                # keep labels above a confidence threshold
                return [lab for lab, sc in pairs if sc >= 0.35][:5]

        _XFORMER_INTENT = _ZSLIntent()
        return _XFORMER_INTENT
    except Exception:
        return None


# --------------------------
# Classification internals
# --------------------------

@dataclass(frozen=True)
class Classification:
    user_input: str
    types: List[str]
    complexity: float   # normalized [0..1]
    complexity_label: str  # 'low'|'medium'|'high'
    safety: Dict[str, Any]
    domain: Optional[str]
    entities_count: int
    estimated_joins: int
    token_estimate: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normalize_complexity(score: float) -> float:
    return min(1.0, max(0.0, float(score)))


def _complexity_label(x: float) -> str:
    if x < 0.35:
        return "low"
    if x < 0.7:
        return "medium"
    return "high"


def _token_estimate(text: str) -> int:
    # Rough heuristic: ~4 chars per token, clamp to sensible min
    return max(32, math.ceil(len(text) / 4))


def _contains_any(text: str, keywords: set[str]) -> bool:
    tl = text.lower()
    return any(k in tl for k in keywords)


def _count_keywords(text: str, keywords: set[str]) -> int:
    tl = text.lower()
    return sum(tl.count(k) for k in keywords)


def _detect_types(tokens: List[str], text: str, *, intent_model: Optional[IntentModel]) -> List[str]:
    types: List[str] = []

    # Heuristic detection
    tl = text.lower()
    if _contains_any(tl, AGG_KEYWORDS):
        types.append("aggregation")
    if _contains_any(tl, GROUPBY_KEYWORDS):
        types.append("groupby")
    if _contains_any(tl, FILTER_KEYWORDS):
        types.append("filtering")
    if _contains_any(tl, JOIN_KEYWORDS):
        types.append("join")
    if _contains_any(tl, TS_KEYWORDS):
        types.append("time_series")
    if _contains_any(tl, WINDOW_KEYWORDS):
        types.append("window")
    if _contains_any(tl, RANKING_KEYWORDS):
        types.append("ranking")
    if _contains_any(tl, TEXT_SEARCH_KEYWORDS):
        types.append("text_search")

    # ML intent (optional): union + uniqueness preserving order
    if intent_model:
        try:
            ml_labels = intent_model.predict_labels(text)
            for lab in ml_labels:
                if lab not in types:
                    types.append(lab)
        except Exception:
            pass

    return types or ["filtering"]  # sensible default


def _estimate_complexity(text: str, types: List[str]) -> float:
    """
    Complexity heuristic:
    - base on length
    - + joins & groupby & window
    - + number of entities-like words
    - + time-series adds modest complexity
    """
    tl = text.lower()
    length_complexity = min(1.0, len(text) / 220.0)
    joins = _count_keywords(tl, JOIN_KEYWORDS)
    groups = _count_keywords(tl, GROUPBY_KEYWORDS)
    windows = _count_keywords(tl, WINDOW_KEYWORDS)
    ts = 1 if "time_series" in types else 0
    entities = len(set(re.findall(r"\b(product|sku|region|country|channel|customer|order|invoice|store|city|month|quarter|year)\b", tl)))

    score = 0.12 + 0.35 * length_complexity + 0.18 * min(joins, 3) + 0.12 * min(groups, 2) + 0.12 * min(windows, 2) + 0.06 * ts + 0.05 * min(entities, 6)
    return _normalize_complexity(score)


def _detect_domain(text: str) -> Optional[str]:
    tl = text.lower()
    if _contains_any(tl, {"invoice", "billing", "ar", "ap", "ledger"}):
        return "billing"
    if _contains_any(tl, {"revenue", "profit", "margin", "p&l", "mrr", "arr", "forecast"}):
        return "finance"
    if _contains_any(tl, {"customer", "cohort", "churn", "retention"}):
        return "customer_success"
    if _contains_any(tl, {"inventory", "warehouse", "shipment", "supply"}):
        return "supply_chain"
    if _contains_any(tl, {"order", "sale", "quote", "lead", "crm"}):
        return "sales"
    if _contains_any(tl, {"latency", "error rate", "throughput", "p99"}):
        return "platform"
    return None


def _detect_sensitivity(text: str, *, pii_detector: Optional[PIIDetector]) -> Dict[str, Any]:
    tl = text.lower()
    # Financial/compliance keywords
    financial = _contains_any(tl, FINANCIAL_KEYWORDS)
    compliance = _contains_any(tl, COMPLIANCE_KEYWORDS)

    pii_hits: List[Dict[str, str]] = []
    # Regex-based quick detection
    for pat, label in PII_PATTERNS:
        for m in re.finditer(pat, text):
            pii_hits.append({"match": m.group(0), "label": label})

    # Optional external PII detector
    if pii_detector:
        try:
            extra = pii_detector.detect(text) or {}
            if isinstance(extra.get("hits"), list):
                pii_hits.extend(extra["hits"])
        except Exception:
            pass

    return {
        "pii": len(pii_hits) > 0,
        "financial_sensitive": bool(financial),
        "compliance_sensitive": bool(compliance),
        "hits": pii_hits[:50],  # cap for safety
    }


# --------------------------
# Public API
# --------------------------

def classify_query(
    user_input: str,
    *,
    context: Optional[Mapping[str, Any]] = None,
    options: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Classify an NL query for downstream routing/planning.

    Parameters
    ----------
    user_input : str
        The natural-language question.
    context : Mapping[str, Any], optional
        Arbitrary metadata (tenant_id, locale, user role). Not required.
    options : Mapping[str, Any], optional
        - "nlp_backend": NLPBackend      -> custom backend (spaCy-like), else tries spaCy, else regex fallback
        - "intent_model": IntentModel    -> optional transformers zero-shot model
        - "pii_detector": PIIDetector    -> optional custom detector
        - "spacy_model": str             -> spaCy model name to auto-load (default: en_core_web_sm)
        - "transformer_model": str       -> HF model for zero-shot (default set above)
        - "enable_ml": bool              -> False to force pure heuristics

    Returns
    -------
    Dict[str, Any]
        {
          "types": List[str],
          "complexity": float,               # normalized [0..1]
          "complexity_label": "low|medium|high",
          "safety": {
              "pii": bool,
              "financial_sensitive": bool,
              "compliance_sensitive": bool,
              "hits": List[{"match": str, "label": str}]
          },
          "domain": Optional[str],
          "entities_count": int,
          "estimated_joins": int,
          "token_estimate": int
        }
    """
    log = get_logger()
    ctx = context or {}
    opts = options or {}

    # Choose NLP backend
    nlp: Optional[NLPBackend] = None
    intent: Optional[IntentModel] = None
    pii_det: Optional[PIIDetector] = opts.get("pii_detector") if isinstance(opts.get("pii_detector"), PIIDetector.__constraints__ if hasattr(PIIDetector, "__constraints__") else object) else opts.get("pii_detector")  # type: ignore

    enable_ml = bool(opts.get("enable_ml", True))
    try:
        nlp = opts.get("nlp_backend")
        if nlp is None and enable_ml:
            nlp = _load_spacy_backend(opts.get("spacy_model", "en_core_web_sm"))
        if nlp is None:
            nlp = _RegexNLP()
    except Exception:
        nlp = _RegexNLP()

    if enable_ml:
        try:
            intent = opts.get("intent_model")
            if intent is None:
                intent = _load_transformer_intent(opts.get("transformer_model", "typeform/distilbert-base-uncased-mnli"))
        except Exception:
            intent = None

    query = user_input.strip()
    if not query:
        log.error("classification_failed empty_input", extra={"event": "classify_error", "user_input": user_input})
        return {
            "types": [],
            "complexity": 0.0,
            "complexity_label": "low",
            "safety": {"pii": False, "financial_sensitive": False, "compliance_sensitive": False, "hits": []},
            "domain": None,
            "entities_count": 0,
            "estimated_joins": 0,
            "token_estimate": 32,
        }

    try:
        tokens = nlp.tokenize(query) if nlp else _RegexNLP().tokenize(query)
        # ML entities if available (not used heavily here; placeholder for future usage)
        ents = []
        try:
            ents = nlp.ents(query) if nlp else []
        except Exception:
            ents = []

        types = _detect_types(tokens, query, intent_model=intent)
        complexity = _estimate_complexity(query, types)
        label = _complexity_label(complexity)
        safety = _detect_sensitivity(query, pii_detector=pii_det)
        domain = _detect_domain(query)
        joins = _count_keywords(query.lower(), JOIN_KEYWORDS)
        entities_count = len(set([t for t in tokens if t.isalpha()]))  # coarse proxy

        result = Classification(
            user_input=query,
            types=types,
            complexity=complexity,
            complexity_label=label,
            safety=safety,
            domain=domain,
            entities_count=entities_count,
            estimated_joins=joins,
            token_estimate=_token_estimate(query),
        ).to_dict()

        # Log structured summary (no PII values)
        try:
            log.info(
                "query_classified",
                extra={
                    "event": "classified",
                    "types": result["types"],
                    "complexity": result["complexity"],
                    "complexity_label": result["complexity_label"],
                    "domain": result["domain"],
                    "estimated_joins": result["estimated_joins"],
                    "entities_count": result["entities_count"],
                    "pii": result["safety"]["pii"],
                    "financial_sensitive": result["safety"]["financial_sensitive"],
                    "compliance_sensitive": result["safety"]["compliance_sensitive"],
                },
            )
        except Exception:
            pass

        return result

    except Exception as e:
        # Fail-safe classification
        try:
            log.error(f"classification_failed error={e}", extra={"event": "classify_error"})
        except Exception:
            pass
        return {
            "types": ["filtering"],
            "complexity": 0.4,
            "complexity_label": "medium",
            "safety": {"pii": False, "financial_sensitive": False, "compliance_sensitive": False, "hits": []},
            "domain": None,
            "entities_count": 0,
            "estimated_joins": 0,
            "token_estimate": _token_estimate(user_input or ""),
        }


# Short alias used by router/run_query
def classify(*, user_input: str, context: Optional[Mapping[str, Any]] = None, options: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Alias for classify_query to match different call sites."""
    return classify_query(user_input=user_input, context=context, options=options)


__all__ = ["classify_query", "classify"]
  
