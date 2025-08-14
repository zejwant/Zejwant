"""
nip_query package initializer.

Provides:
- Structured logging initialization for the query engine.
- A high-level `run_query(user_input: str, ...)` orchestration function:
    classify -> route -> execute -> format
- Clean, modular imports (lazy to avoid circular dependencies).
- Full type hints and concise, useful docstrings.
- Ready for unit testing with dependency injection hooks.

Usage:
    from nip_query import run_query

    result = run_query("Show me last month's revenue by product")
    print(result["text"])  # Plain-English summary
    df = result.get("data")  # Optional raw/structured data (e.g., list[dict] / DataFrame)
"""

from __future__ import annotations

import logging
import logging.config
import os
import time
import uuid
from typing import Any, Dict, Optional, Mapping, MutableMapping

__all__ = [
    "run_query",
    "configure_logging",
    "get_logger",
]

# ----------------------------
# Logging setup (structured)
# ----------------------------

_DEFAULT_LOG_LEVEL = os.getenv("NIP_QUERY_LOG_LEVEL", "INFO").upper()
_LOGGER_NAME = "nip_query"


def _kvfmt(record: logging.LogRecord) -> str:
    """
    Lightweight key=value logfmt-style formatter without external deps.

    Includes stable fields that are helpful for distributed tracing and debugging.
    """
    # Base fields
    base = {
        "ts": f"{getattr(record, 'ts', time.time()):.6f}",
        "level": record.levelname,
        "logger": record.name,
        "event": getattr(record, "event", record.getMessage().split(" ")[0][:32] if record.getMessage() else ""),
        "msg": record.getMessage(),
        "query_id": getattr(record, "query_id", ""),
        "module": record.module,
        "func": record.funcName,
        "line": record.lineno,
    }
    # Render key=value with spaces, quoting only if needed
    def render(v: Any) -> str:
        s = str(v)
        if any(c.isspace() for c in s) or "=" in s:
            return f'"{s}"'
        return s

    return " ".join(f"{k}={render(v)}" for k, v in base.items() if v != "")


class _KVFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return _kvfmt(record)


def configure_logging(level: str = _DEFAULT_LOG_LEVEL) -> None:
    """
    Configure package logging with a structured (logfmt-style) formatter.

    Idempotent: safe to call multiple times (handlers replaced on the package logger only).
    """
    logger = logging.getLogger(_LOGGER_NAME)
    logger.handlers.clear()
    logger.setLevel(level)

    handler = logging.StreamHandler()
    handler.setFormatter(_KVFormatter())
    logger.addHandler(handler)
    logger.propagate = False  # Keep logs from duplicating to root unless explicitly desired


def get_logger() -> logging.Logger:
    """
    Get the package logger. Ensures logging is configured once on first import.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    if not logger.handlers:
        configure_logging()
    return logger


# Ensure logging is configured on import
configure_logging()


# ----------------------------
# Utilities
# ----------------------------

def _safe_import(module_path: str):
    """
    Import a module path safely with a clearer error message for missing deps/files.
    """
    try:
        __import__(module_path)
        return globals()[module_path.split(".")[0]] if "." not in module_path else __import__(module_path, fromlist=["*"])
    except ImportError as exc:
        raise ImportError(
            f"[nip_query] Failed to import '{module_path}'. Ensure the module exists and "
            f"that your PYTHONPATH is set correctly. Original error: {exc}"
        ) from exc


def _gen_query_id() -> str:
    """Generate a stable correlation ID per query invocation."""
    return uuid.uuid4().hex


# ----------------------------
# Public API
# ----------------------------

def run_query(
    user_input: str,
    *,
    context: Optional[Mapping[str, Any]] = None,
    options: Optional[Mapping[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Orchestrate the query lifecycle:
      1) Classify the query (complexity/intent) using `query_classifier`.
      2) Route using `query_router` (select DB/table/engine or NL->SQL path).
      3) Execute via `query_processor` (safe SQL execution / retrieval).
      4) Format through `result_formatter` (plain-English insight + optional visuals).

    Parameters
    ----------
    user_input : str
        The natural-language query from the user.
    context : Mapping[str, Any], optional
        Read-only contextual metadata (e.g., user_id, tenant_id, locale).
    options : Mapping[str, Any], optional
        Execution options (e.g., timeout_ms, max_rows, strict_mode, dry_run).
    logger : logging.Logger, optional
        Inject a custom logger (useful for tests). Defaults to the package logger.

    Returns
    -------
    Dict[str, Any]
        A structured result payload:
            {
              "query_id": str,               # correlation id for tracing
              "classification": Any,         # classifier output (opaque type)
              "route": Any,                  # routing decision/plan (opaque type)
              "raw": Any,                    # raw execution result (e.g., rows, DataFrame)
              "text": str,                   # plain-English formatted summary/insight
              "meta": Dict[str, Any]         # timings, limits, etc.
            }

    Notes
    -----
    - Imports are lazy inside this function to avoid any circular dependency issues
      within the `nip_query` package.
    - The function is unit-test friendly: pass fakes/mocks by monkeypatching the
      target modules or injecting a custom logger. You can also pass `options`
      like `{"dry_run": True}` for planner-only tests.
    - This function does not raise on "expected" user errors; it returns a safe
      message in `text` and logs the details at DEBUG/INFO/ERROR appropriately.
    """
    log = logger or get_logger()
    query_id = _gen_query_id()
    t0 = time.time()
    ctx: Mapping[str, Any] = context or {}
    opts: Mapping[str, Any] = options or {}

    # Bind correlation fields into log records via 'extra'
    def _log(level: int, event: str, msg: str, **fields: Any) -> None:
        log.log(level, msg, extra={"event": event, "query_id": query_id, "ts": time.time(), **fields})

    _log(logging.INFO, "start", "run_query invoked", user_input=user_input)

    # 1) Classify
    try:
        classifier = _safe_import("nip_query.query_classifier")
        classify_fn = getattr(classifier, "classify_query", None) or getattr(classifier, "classify", None)
        if classify_fn is None:
            raise AttributeError("query_classifier must expose 'classify_query' or 'classify'")
        classification = classify_fn(user_input=user_input, context=ctx, options=opts)  # type: ignore[call-arg]
        _log(logging.DEBUG, "classified", "classification complete", classification=str(classification))
    except Exception as e:
        _log(logging.ERROR, "classify_error", f"classification failed: {e}")
        return {
            "query_id": query_id,
            "classification": None,
            "route": None,
            "raw": None,
            "text": "Sorry, I couldn't understand the question. Please try rephrasing.",
            "meta": {"error": "classification_failed", "elapsed_ms": int((time.time() - t0) * 1000)},
        }

    # 2) Route
    try:
        router = _safe_import("nip_query.query_router")
        route_fn = getattr(router, "route", None) or getattr(router, "route_query", None)
        if route_fn is None:
            raise AttributeError("query_router must expose 'route' or 'route_query'")
        route_plan = route_fn(classification=classification, user_input=user_input, context=ctx, options=opts)  # type: ignore[call-arg]
        _log(logging.DEBUG, "routed", "routing decision complete", route=str(route_plan))
    except Exception as e:
        _log(logging.ERROR, "route_error", f"routing failed: {e}")
        return {
            "query_id": query_id,
            "classification": classification,
            "route": None,
            "raw": None,
            "text": "I understood the question, but couldn't plan how to execute it.",
            "meta": {"error": "routing_failed", "elapsed_ms": int((time.time() - t0) * 1000)},
        }

    # Optional dry run (planner-only)
    if opts.get("dry_run"):
        _log(logging.INFO, "dry_run", "dry run requested; returning plan only")
        # Try to prettify via formatter if available
        try:
            formatter = _safe_import("nip_query.result_formatter")
            fmt_fn = getattr(formatter, "format_plan", None)
            text = fmt_fn(route_plan) if callable(fmt_fn) else f"Plan: {route_plan}"
        except Exception:
            text = f"Plan: {route_plan}"
        return {
            "query_id": query_id,
            "classification": classification,
            "route": route_plan,
            "raw": None,
            "text": text,
            "meta": {"elapsed_ms": int((time.time() - t0) * 1000), "dry_run": True},
        }

    # 3) Execute
    try:
        processor = _safe_import("nip_query.query_processor")
        exec_fn = getattr(processor, "execute", None) or getattr(processor, "execute_plan", None)
        if exec_fn is None:
            raise AttributeError("query_processor must expose 'execute' or 'execute_plan'")
        raw_result = exec_fn(route_plan, context=ctx, options=opts)  # type: ignore[call-arg]
        _log(logging.DEBUG, "executed", "execution complete")
    except Exception as e:
        _log(logging.ERROR, "execute_error", f"execution failed: {e}")
        return {
            "query_id": query_id,
            "classification": classification,
            "route": route_plan,
            "raw": None,
            "text": "The query plan failed during execution. Please try again or refine the question.",
            "meta": {"error": "execution_failed", "elapsed_ms": int((time.time() - t0) * 1000)},
        }

    # 4) Format
    try:
        formatter = _safe_import("nip_query.result_formatter")
        fmt_fn = getattr(formatter, "format_result", None) or getattr(formatter, "format", None)
        if fmt_fn is None:
            raise AttributeError("result_formatter must expose 'format_result' or 'format'")
        text = fmt_fn(raw_result, classification=classification, route=route_plan, context=ctx, options=opts)  # type: ignore[call-arg]
        _log(logging.INFO, "formatted", "formatting complete")
    except Exception as e:
        _log(logging.ERROR, "format_error", f"formatting failed: {e}")
        text = "The query executed, but I couldn't format the result."

    elapsed_ms = int((time.time() - t0) * 1000)
    return {
        "query_id": query_id,
        "classification": classification,
        "route": route_plan,
        "raw": raw_result,
        "text": text,
        "meta": {"elapsed_ms": elapsed_ms},
  }
  
