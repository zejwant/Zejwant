"""
nip_query.llm_based

Enterprise LLM-backed SQL generation helper.

Features
--------
- Accepts plain-English question + DB schema (dict: table -> set(columns))
- Calls a pluggable LLM client to generate SQL
- Validates generated SQL via pluggable `sql_validator`
- Detects hallucinations (referenced columns/tables not in schema)
- Retries with clarifying prompts; falls back to degraded template generation
- Full structured logging of prompts, LLM responses, and validation results
- Unit-testable: inject mock llm_client and sql_validator

Public API
----------
- generate_sql_with_llm(user_input, schema, ...)
- simulate_llm_generation(...)  # convenience for tests
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

# Try to reuse package logger if available
try:
    from nip_query import get_logger  # type: ignore
except Exception:
    def get_logger() -> logging.Logger:  # fallback
        logger = logging.getLogger("nip_query.llm_based_fallback")
        if not logger.handlers:
            handler = logging.StreamHandler()
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger


# ------------------------
# Types & dataclasses
# ------------------------
@dataclass(frozen=True)
class LLMResponse:
    text: str
    raw: Dict[str, Any]  # opaque raw LLM response (provider-specific)


@dataclass
class LLMAttemptRecord:
    attempt: int
    prompt: str
    response: Optional[LLMResponse]
    parse_ok: bool
    hallucination: bool
    missing_tables: List[str]
    missing_columns: List[Tuple[str, str]]  # (table, column)
    validation: Dict[str, Any]


@dataclass
class LLMPlanResult:
    sql: Optional[str]
    params: List[Any]
    strategy: str  # 'llm', 'clarify', 'fallback_template', or 'failed'
    attempts: List[LLMAttemptRecord]
    final_validation: Dict[str, Any]


# ------------------------
# Helpers: SQL parsing & hallucination detection
# ------------------------

_SELECT_FROM_JOIN_RE = re.compile(
    r"(?:from|join)\s+([`\"]?([A-Za-z_][A-Za-z0-9_]*)[`\"]?)(?:\s+as\s+[`\"]?[A-Za-z_][A-Za-z0-9_]*[`\"]?)?",
    re.I,
)
_COLUMN_RE = re.compile(
    r"select\s+(.*?)\s+from\s", re.I | re.S
)
_COLUMN_SPLIT_RE = re.compile(r"\s*,\s*|\s+as\s+", re.I)


def _extract_tables(sql: str) -> List[str]:
    """Return a list of candidate table names referenced in FROM/JOIN clauses."""
    found = []
    for m in _SELECT_FROM_JOIN_RE.finditer(sql):
        # group 2 is bare identifier (no quotes)
        name = m.group(2) or m.group(1)
        if name:
            found.append(name.lower())
    return list(dict.fromkeys(found))


def _extract_columns(sql: str) -> List[str]:
    """
    Attempt to extract explicit column names from the SELECT clause.
    This is conservative and approximate; useful for hallucination checks.
    """
    m = _COLUMN_RE.search(sql)
    if not m:
        return []
    col_block = m.group(1)
    # Remove function calls e.g., sum(col) -> sum(col)
    # Split on commas (simple)
    cols = [c.strip() for c in col_block.split(",")]
    # Clean aliases and table qualifiers
    cleaned = []
    for c in cols:
        # drop expressions after space (alias) e.g., "col as alias"
        c = re.split(r"\s+as\s+", c, flags=re.I)[0]
        # drop function wrappers: sum(col) -> col
        inner = re.sub(r"[A-Za-z_]+\((.*?)\)", r"\1", c)
        # handle table.column
        if "." in inner:
            inner = inner.split(".")[-1]
        # drop quotes/backticks
        inner = inner.strip("`\" ")
        if inner:
            cleaned.append(inner.lower())
    return cleaned


def _detect_hallucinations(sql: str, schema: Dict[str, Iterable[str]]) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Compare referenced tables/columns against provided schema.

    Returns:
        (missing_tables, missing_columns)
    """
    schema_norm = {tbl.lower(): {col.lower() for col in cols} for tbl, cols in schema.items()}
    tables = _extract_tables(sql)
    columns = _extract_columns(sql)
    missing_tables = [t for t in tables if t not in schema_norm]
    # columns: try to associate with tables; naive approach:
    missing_columns: List[Tuple[str, str]] = []
    # If only one table present, check columns against it
    if tables:
        if len(tables) == 1:
            tbl = tables[0]
            for c in columns:
                if tbl in schema_norm and c not in schema_norm[tbl]:
                    missing_columns.append((tbl, c))
        else:
            # For multiple tables, check column existence in any table
            for c in columns:
                found = any(c in cols for cols in schema_norm.values())
                if not found:
                    missing_columns.append(("<unknown>", c))
    else:
        # No table discovered — we can't reliably detect columns
        for c in columns:
            found_any = any(c in cols for cols in schema_norm.values())
            if not found_any:
                missing_columns.append(("<unknown>", c))
    return missing_tables, missing_columns


# ------------------------
# Default LLM client stub (must be replaced in prod)
# ------------------------
def default_llm_client(prompt: str, *, max_tokens: int = 512, temperature: float = 0.0, **kwargs) -> LLMResponse:
    """
    Default stub for LLM calls. Intentionally raises to force injection.
    Production: inject a function that calls your LLM provider and returns LLMResponse.
    """
    raise NotImplementedError(
        "No default LLM client provided. Inject one via `llm_client` parameter. "
        "Example signature: llm_client(prompt, max_tokens=512, temperature=0.0) -> LLMResponse"
    )


# ------------------------
# Default SQL validator (lightweight)
# ------------------------
def default_sql_validator(sql: str) -> Dict[str, Any]:
    """
    Very lightweight SQL validator.
    - Ensures only a single statement (no semicolons)
    - Basic SELECT-only guard (deny INSERT/UPDATE/DELETE by default)
    - Return dict: {valid: bool, errors: List[str], warnings: List[str]}
    Replace with your production sql_validator which should check syntax & safety.
    """
    errors = []
    warnings = []
    if ";" in sql.strip().rstrip(";"):
        errors.append("multiple_statements_or_semicolon_detected")
    # Basic whitelist
    if not re.match(r"^\s*select\s", sql, re.I | re.S):
        errors.append("only_select_statements_are_allowed")
    # crude suspicious pattern guard
    if re.search(r"\b(drop|truncate|alter|delete|insert|update)\b", sql, re.I):
        errors.append("dangerous_sql_keywords_detected")
    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


# ------------------------
# Fallback template generator (degraded)
# ------------------------
def degraded_template_generator(user_input: str, schema: Dict[str, Iterable[str]]) -> Tuple[str, List[Any]]:
    """
    Build a very conservative SQL using heuristics:
      - Pick the most-likely single table from schema by keyword matching
      - SELECT a small set of columns or '*' if uncertain
      - Apply a LIMIT 100
    This avoids LLM and is deterministic.
    """
    text = user_input.lower()
    # score tables by keyword overlap
    table_scores: Dict[str, int] = {}
    for tbl, cols in schema.items():
        score = 0
        score += sum(1 for kw in re.findall(r"\w+", text) if kw in tbl.lower())
        score += sum(1 for kw in re.findall(r"\w+", text) if any(kw in (c.lower()) for c in cols))
        table_scores[tbl] = score
    # choose best
    best_table = max(table_scores.items(), key=lambda kv: (kv[1], kv[0]))[0] if table_scores else None
    if not best_table:
        return ("SELECT * FROM information_schema.tables LIMIT 1", [])
    # choose up to 4 columns
    chosen_cols = list(schema[best_table])[:4] or ["*"]
    sql = f"SELECT {', '.join(chosen_cols)} FROM {best_table} LIMIT 100"
    return sql, []


# ------------------------
# Core LLM-based generation function
# ------------------------
def generate_sql_with_llm(
    user_input: str,
    schema: Dict[str, Iterable[str]],
    *,
    llm_client: Callable[..., LLMResponse] = default_llm_client,
    sql_validator: Callable[[str], Dict[str, Any]] = default_sql_validator,
    max_attempts: int = 3,
    temperature: float = 0.0,
    max_tokens: int = 512,
    logger: Optional[logging.Logger] = None,
    prompt_template: Optional[str] = None,
    clarify_template: Optional[str] = None,
) -> LLMPlanResult:
    """
    Generate SQL via an LLM with hallucination checks and retries.

    Parameters
    ----------
    user_input : str
        Natural-language question.
    schema : Dict[str, Iterable[str]]
        DB schema mapping table -> iterable of column names
    llm_client : Callable
        Must accept (prompt, max_tokens=int, temperature=float, **kwargs) and return an LLMResponse
    sql_validator : Callable
        Accepts SQL string and returns dict with keys {"valid": bool, "errors": [...], "warnings": [...]}
    max_attempts : int
        Total LLM attempts before fallback
    temperature, max_tokens : LLM call options
    logger : logging.Logger, optional
        Structured logger; default is nip_query package logger

    Returns
    -------
    LLMPlanResult
    """
    log = logger or get_logger()
    attempts: List[LLMAttemptRecord] = []

    # Prepare templates
    if prompt_template is None:
        prompt_template = (
            "You are an expert SQL generator. Given the database schema and the user's question, "
            "generate a single read-only SELECT SQL statement (no data manipulation). "
            "Schema: {schema}\n\nUser question: {question}\n\nRespond with only the SQL."
        )
    if clarify_template is None:
        clarify_template = (
            "The prior SQL referenced columns or tables not present in the schema. "
            "Please re-generate a corrected SELECT statement using only the schema below.\n\nSchema: {schema}\n\n"
            "Original question: {question}\n\nPrior SQL: {prior_sql}\n\n"
            "Missing tables: {missing_tables}\nMissing columns: {missing_columns}\n\nRespond with only the corrected SQL."
        )

    # Build canonical schema text for prompt (truncate if large)
    schema_text = _format_schema_for_prompt(schema, max_chars=4000)

    # Attempt loop
    last_sql: Optional[str] = None
    strategy = "failed"
    for attempt in range(1, max_attempts + 1):
        if attempt == 1:
            prompt = prompt_template.format(schema=schema_text, question=user_input)
        else:
            # Clarify prompt with prior SQL and missing items
            prev = attempts[-1]
            prompt = clarify_template.format(
                schema=schema_text,
                question=user_input,
                prior_sql=(prev.response.text if prev.response else last_sql or ""),
                missing_tables=", ".join(prev.missing_tables or []),
                missing_columns=", ".join(f"{t}.{c}" for t, c in (prev.missing_columns or [])),
            )

        log.info("llm_prompt", extra={"event": "llm_prompt", "attempt": attempt, "prompt_preview": prompt[:400]})
        response = None
        try:
            raw = llm_client(prompt, max_tokens=max_tokens, temperature=temperature)
            response = raw
            log.info("llm_response", extra={"event": "llm_response", "attempt": attempt, "response_preview": (raw.text[:800] if raw and raw.text else "")})
        except Exception as e:
            # Record failure and continue to next attempt (or fallback)
            log.error("llm_call_failed", extra={"event": "llm_error", "attempt": attempt, "error": str(e)})
            attempts.append(LLMAttemptRecord(
                attempt=attempt,
                prompt=prompt,
                response=None,
                parse_ok=False,
                hallucination=False,
                missing_tables=[],
                missing_columns=[],
                validation={"valid": False, "errors": [f"llm_call_failed: {e}"], "warnings": []},
            ))
            continue

        # Try to parse SQL text
        generated_sql = (response.text or "").strip() if response else ""
        # If user or LLM included triple-backticks or markdown, strip them
        generated_sql = _strip_code_fences(generated_sql)

        # Validation
        validation = {}
        try:
            validation = sql_validator(generated_sql)
        except Exception as e:
            validation = {"valid": False, "errors": [f"validator_exception: {e}"], "warnings": []}

        # Hallucination detection
        missing_tables, missing_columns = _detect_hallucinations(generated_sql, schema)

        halluc = bool(missing_tables or missing_columns)
        parse_ok = bool(generated_sql)

        attempts.append(LLMAttemptRecord(
            attempt=attempt,
            prompt=prompt,
            response=response,
            parse_ok=parse_ok,
            hallucination=halluc,
            missing_tables=missing_tables,
            missing_columns=missing_columns,
            validation=validation,
        ))

        # If validation fails badly, try again
        if not validation.get("valid", False) or halluc:
            log.info(
                "llm_generation_issue",
                extra={
                    "event": "llm_issue",
                    "attempt": attempt,
                    "validation": validation,
                    "hallucination": halluc,
                    "missing_tables": missing_tables,
                    "missing_columns": missing_columns,
                },
            )
            last_sql = generated_sql
            # If last attempt, we'll fall back below
            continue

        # Success
        strategy = "llm"
        final_sql = generated_sql
        # TODO: parameter extraction could be a future step. For now return empty params.
        plan = LLMPlanResult(sql=final_sql, params=[], strategy=strategy, attempts=attempts, final_validation=validation)
        log.info("llm_plan_success", extra={"event": "llm_success", "strategy": strategy, "attempts": len(attempts)})
        return plan

    # If we reach here, LLM failed to produce a valid non-hallucinated SQL.
    # Fall back to degraded template generator
    try:
        fallback_sql, fallback_params = degraded_template_generator(user_input, schema)
        fallback_validation = sql_validator(fallback_sql)
        strategy = "fallback_template" if fallback_validation.get("valid", False) else "failed"
        final_validation = fallback_validation
        plan = LLMPlanResult(sql=fallback_sql if fallback_validation.get("valid", False) else None,
                             params=fallback_params,
                             strategy=strategy,
                             attempts=attempts,
                             final_validation=final_validation)
        log.warning("llm_fallback_used", extra={"event": "llm_fallback", "strategy": strategy, "attempts": len(attempts)})
        return plan
    except Exception as e:
        # As a last resort, return failure structure
        final_validation = {"valid": False, "errors": [f"fallback_failed: {e}"], "warnings": []}
        plan = LLMPlanResult(sql=None, params=[], strategy="failed", attempts=attempts, final_validation=final_validation)
        log.error("llm_all_attempts_failed", extra={"event": "llm_failed_all", "error": str(e)})
        return plan


# ------------------------
# Utility functions
# ------------------------
def _format_schema_for_prompt(schema: Dict[str, Iterable[str]], *, max_chars: int = 2000) -> str:
    """
    Render schema to text for inclusion in an LLM prompt. Keep concise.
    """
    parts: List[str] = []
    for tbl, cols in schema.items():
        cols_list = ", ".join(list(cols)[:20])
        parts.append(f"{tbl}({cols_list})")
    out = "\n".join(parts)
    if len(out) > max_chars:
        return out[: max_chars - 20] + "\n... (truncated)"
    return out


def _strip_code_fences(text: str) -> str:
    # Remove ```sql ... ``` or ``` ... ``` blocks and leading/trailing backticks
    fence_re = re.compile(r"```(?:sql)?\s*(.*?)\s*```", re.S | re.I)
    m = fence_re.search(text)
    if m:
        return m.group(1).strip()
    return text.strip("` \n")


# ------------------------
# Simulation helper (useful for tests)
# ------------------------
def simulate_llm_generation(
    question: str,
    schema: Dict[str, Iterable[str]],
    *,
    fake_llm_responses: Optional[List[str]] = None,
    **kwargs,
) -> LLMPlanResult:
    """
    Convenience wrapper that constructs a fake llm_client from `fake_llm_responses`
    (a list of strings to be returned sequentially). Useful for unit tests.
    """
    responses = list(fake_llm_responses or [])

    def fake_client(prompt: str, **kw) -> LLMResponse:
        # Pop the next response, or raise to simulate errors
        if not responses:
            raise RuntimeError("No more fake responses")
        txt = responses.pop(0)
        return LLMResponse(text=txt, raw={"simulated": True})

    return generate_sql_with_llm(
        question,
        schema,
        llm_client=fake_client,
        sql_validator=kwargs.get("sql_validator", default_sql_validator),
        max_attempts=kwargs.get("max_attempts", 3),
        temperature=kwargs.get("temperature", 0.0),
        max_tokens=kwargs.get("max_tokens", 512),
        logger=kwargs.get("logger", None),
    )


__all__ = [
    "generate_sql_with_llm",
    "simulate_llm_generation",
    "LLMResponse",
    "LLMPlanResult",
  ]
  
