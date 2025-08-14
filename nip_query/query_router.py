"""
nip_query.query_router

Enterprise-grade router that picks between rule-based and LLM-based SQL generation.

Responsibilities
---------------
- Accepts classification metadata produced by `query_classifier`.
- Applies decision logic (complexity threshold, domain rules, cost/latency controls).
- Produces a structured, testable routing plan (no execution here).
- Logs routing decisions in structured form.
- Exposes:
    - route(...): pure routing (classification -> RoutePlan)
    - route_query(...): simulation helper for quick/manual tests and mocks

Design Notes
------------
- Zero heavy imports at module import time.
- Logging is provided by nip_query.get_logger(), which is lightweight.
- All thresholds/budgets are encapsulated in RouterConfig to enable runtime updates and unit tests.
- Dependency injection points:
    - `cost_estimator`, `latency_estimator` callables
    - `domain_policy` callable
- Safe defaults: prefer rule-based when in doubt, low budget, or strict latency SLOs.

Example
-------
    plan = route_query("Show revenue growth QoQ for top 10 SKUs in EMEA")
    print(plan.strategy)  # "llm_based" (for complex, cross-domain queries)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple, Union, Set

# Lightweight structured logger from package __init__.py
try:
    from nip_query import get_logger  # type: ignore
except Exception:  # pragma: no cover - in tests you can monkeypatch logging
    import logging

    def get_logger() -> logging.Logger:  # type: ignore
        logger = logging.getLogger("nip_query.query_router_fallback")
        if not logger.handlers:
            handler = logging.StreamHandler()
            logger.addHandler(handler)
            logger.setLevel("INFO")
        return logger


# ---------- Data structures ----------

@dataclass(frozen=True)
class RoutePlan:
    """
    A structured routing decision.

    Attributes
    ----------
    strategy : str
        Either "rule_based" or "llm_based".
    reason : str
        Human-readable justification for the decision.
    cost_tier : str
        "low" | "medium" | "high" — a coarse indicator for downstream cost guards.
    estimates : Dict[str, Any]
        Heuristics for tokens, cost, latency, etc. (opaque to callers).
    constraints : Dict[str, Any]
        Relevant constraints (budgets, SLOs, safety flags) used in decision.
    metadata : Dict[str, Any]
        Additional metadata (e.g., domain, tables, complexity).
    """
    strategy: str
    reason: str
    cost_tier: str
    estimates: Dict[str, Any]
    constraints: Dict[str, Any]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RouterConfig:
    """
    Router configuration with sane enterprise defaults.

    complexity_threshold : float
        [0..1]. <= threshold → rule_based by default, unless overridden by policies.
    allowed_llm_domains : Optional[Set[str]]
        If provided, restrict LLM usage to these domains only.
    prefer_rule_based_domains : Set[str]
        Domains where rule-based is preferred (e.g., finance-critical paths).
    hard_llm_budget_per_query : float
        Absolute currency limit for a single LLM call (prevents runaway cost).
    soft_daily_budget_remaining : float
        Rolling budget remaining today; can bias decision to rule_based if low.
    latency_slo_ms : int
        If the latency SLO is very tight, bias to rule_based.
    allow_llm : bool
        Global kill-switch for LLM usage.
    """
    complexity_threshold: float = 0.45
    allowed_llm_domains: Optional[Set[str]] = None
    prefer_rule_based_domains: Set[str] = frozenset({"finance", "billing", "compliance"})
    hard_llm_budget_per_query: float = 0.10  # e.g., 10 cents limit per query
    soft_daily_budget_remaining: float = 50.0
    latency_slo_ms: int = 2000
    allow_llm: bool = True


# ---------- Default estimators & policies (override in tests if needed) ----------

def default_cost_estimator(tokens: int, model: str = "gpt-4o-mini") -> float:
    """
    Very rough token→cost estimator (USD) for routing purposes only.
    Replace with your exact provider pricing in production.
    """
    # Assume $0.15 / 1M input tokens ≈ 1.5e-7 per token; output ignored for routing.
    # This is intentionally conservative and should be adjusted.
    return 1.5e-7 * tokens


def default_latency_estimator(strategy: str, tokens: int) -> int:
    """
    Return estimated end-to-end latency in milliseconds for the selected strategy.
    """
    if strategy == "rule_based":
        # Deterministic templates + validation
        return 80 + min(tokens // 100, 50)
    # LLM-based: prompt + generation + validation
    return 600 + min(tokens // 20, 2500)


def default_domain_policy(
    domain: Optional[str],
    config: RouterConfig,
    *,
    classification: Mapping[str, Any],
) -> Optional[str]:
    """
    Domain policy hook.

    Returns:
        - "rule_based" or "llm_based" to force a strategy
        - None to indicate no opinion
    """
    if domain and domain in config.prefer_rule_based_domains:
        return "rule_based"
    if config.allowed_llm_domains is not None:
        # If domain not explicitly allowed for LLM, prefer rule_based
        if domain not in config.allowed_llm_domains:
            return "rule_based"
    return None


# ---------- Core routing logic ----------

def _estimate_tokens(user_input: str, classification: Mapping[str, Any]) -> int:
    """
    Crude token estimation for routing (chars ~ 4 tokens). Prefer classifier output if present.
    """
    if "token_estimate" in classification:
        try:
            return int(classification["token_estimate"])
        except Exception:
            pass
    return max(32, len(user_input) // 4)


def _complexity_score(classification: Mapping[str, Any]) -> float:
    """
    Fetch a normalized complexity score [0..1] from classifier; fallback to heuristic.
    """
    cx = classification.get("complexity")
    if isinstance(cx, (float, int)):
        # If classifier returns e.g., 0..1 or 0..100, normalize
        return float(cx) if 0 <= float(cx) <= 1 else min(1.0, float(cx) / 100.0)
    # Heuristic: more joins/entities → higher complexity
    joins = int(classification.get("estimated_joins", 0) or 0)
    entities = int(classification.get("entities_count", 0) or 0)
    return min(1.0, 0.2 + 0.15 * joins + 0.05 * entities)


def route(
    *,
    classification: Mapping[str, Any],
    user_input: str,
    context: Optional[Mapping[str, Any]] = None,
    options: Optional[Mapping[str, Any]] = None,
    config: Optional[RouterConfig] = None,
    cost_estimator: Callable[[int, str], float] = default_cost_estimator,
    latency_estimator: Callable[[str, int], int] = default_latency_estimator,
    domain_policy: Callable[[Optional[str], RouterConfig], Optional[str]] = lambda d, c, **k: default_domain_policy(d, c, **k),
) -> RoutePlan:
    """
    Compute a routing decision (no execution).

    Parameters
    ----------
    classification : Mapping[str, Any]
        Output from `query_classifier`. Expected keys (optional but helpful):
            - "complexity": float in [0..1] or [0..100]
            - "domain": str (e.g., "finance", "sales")
            - "safety": {"pii": bool, "compliance_sensitive": bool, ...}
            - "entities_count": int
            - "estimated_joins": int
            - "token_estimate": int
    user_input : str
        Original user prompt (for heuristic fallbacks).
    context : Mapping[str, Any], optional
        Request context (tenant, user, SLO, etc.). Recognized keys (optional):
            - "latency_slo_ms": int
            - "force_strategy": "rule_based"|"llm_based"
    options : Mapping[str, Any], optional
        Execution options; recognized keys:
            - "model": str (for token → cost estimation)
            - "allow_llm": bool (overrides config.allow_llm)
            - "cost_hard_limit": float (per-query USD)
    config : RouterConfig, optional
        Router configuration. Uses defaults when omitted.
    cost_estimator, latency_estimator, domain_policy : callables
        Dependency-injection points for tests/custom logic.

    Returns
    -------
    RoutePlan
        Structured plan indicating which strategy to use and why.
    """
    log = get_logger()
    cfg = config or RouterConfig()
    ctx = dict(context or {})
    opts = dict(options or {})

    # 1) Hard overrides
    force = ctx.get("force_strategy") or opts.get("force_strategy")
    if force in {"rule_based", "llm_based"}:
        strategy = str(force)
        reason = "forced_by_context_option"
        tokens = _estimate_tokens(user_input, classification)
        model = str(opts.get("model", "gpt-4o-mini"))
        est_cost = cost_estimator(tokens, model)
        latency = latency_estimator(strategy, tokens)
        plan = RoutePlan(
            strategy=strategy,
            reason=reason,
            cost_tier=_cost_tier(est_cost),
            estimates={"tokens": tokens, "cost_usd": est_cost, "latency_ms": latency, "model": model},
            constraints=_constraints_snapshot(cfg, ctx, opts),
            metadata=_metadata_snapshot(classification, user_input),
        )
        _log_route(log, plan, event="route_forced")
        return plan

    # 2) Gather signals
    model = str(opts.get("model", "gpt-4o-mini"))
    tokens = _estimate_tokens(user_input, classification)
    complexity = _complexity_score(classification)
    domain = classification.get("domain")
    safety = classification.get("safety") or {}
    slo_ms = int(ctx.get("latency_slo_ms", cfg.latency_slo_ms))

    # 3) Policy: safety & domain
    if isinstance(safety, Mapping) and (safety.get("compliance_sensitive") or safety.get("pii")):
        # In regulated cases, prefer deterministic paths
        safety_bias = "rule_based"
    else:
        safety_bias = None

    domain_bias = domain_policy(domain, cfg, classification=classification)

    # 4) Cost & latency estimates
    est_cost_llm = cost_estimator(tokens, model)
    est_latency_rule = latency_estimator("rule_based", tokens)
    est_latency_llm = latency_estimator("llm_based", tokens)

    allow_llm = bool(opts.get("allow_llm", cfg.allow_llm))
    hard_limit = float(opts.get("cost_hard_limit", cfg.hard_llm_budget_per_query))
    daily_remaining = float(opts.get("daily_budget_remaining", cfg.soft_daily_budget_remaining))

    # 5) Decision logic (ordered by strictness)
    #    a) Global LLM kill-switch
    if not allow_llm:
        return _finalize(
            log, cfg, ctx, opts, classification, user_input,
            strategy="rule_based",
            reason="llm_disabled_globally",
            model=model, tokens=tokens, est_latency_rule=est_latency_rule, est_latency_llm=est_latency_llm,
            est_cost_llm=est_cost_llm,
        )

    #    b) Safety & domain policies
    if safety_bias == "rule_based":
        return _finalize(
            log, cfg, ctx, opts, classification, user_input,
            strategy="rule_based",
            reason="safety_policy_preference",
            model=model, tokens=tokens, est_latency_rule=est_latency_rule, est_latency_llm=est_latency_llm,
            est_cost_llm=est_cost_llm,
        )
    if domain_bias in {"rule_based", "llm_based"}:
        return _finalize(
            log, cfg, ctx, opts, classification, user_input,
            strategy=domain_bias,
            reason="domain_policy_preference",
            model=model, tokens=tokens, est_latency_rule=est_latency_rule, est_latency_llm=est_latency_llm,
            est_cost_llm=est_cost_llm,
        )

    #    c) Latency SLO guard
    if est_latency_llm > slo_ms and est_latency_rule <= slo_ms:
        return _finalize(
            log, cfg, ctx, opts, classification, user_input,
            strategy="rule_based",
            reason="latency_slo_guard",
            model=model, tokens=tokens, est_latency_rule=est_latency_rule, est_latency_llm=est_latency_llm,
            est_cost_llm=est_cost_llm,
        )

    #    d) Cost guards
    if est_cost_llm > hard_limit or daily_remaining < est_cost_llm:
        return _finalize(
            log, cfg, ctx, opts, classification, user_input,
            strategy="rule_based",
            reason="cost_guard_triggered",
            model=model, tokens=tokens, est_latency_rule=est_latency_rule, est_latency_llm=est_latency_llm,
            est_cost_llm=est_cost_llm,
        )

    #    e) Complexity threshold
    if complexity <= cfg.complexity_threshold:
        # Simple queries are cheaper/faster as rule-based
        return _finalize(
            log, cfg, ctx, opts, classification, user_input,
            strategy="rule_based",
            reason="below_complexity_threshold",
            model=model, tokens=tokens, est_latency_rule=est_latency_rule, est_latency_llm=est_latency_llm,
            est_cost_llm=est_cost_llm,
        )

    #    f) Default: LLM for complex queries
    return _finalize(
        log, cfg, ctx, opts, classification, user_input,
        strategy="llm_based",
        reason="above_complexity_threshold",
        model=model, tokens=tokens, est_latency_rule=est_latency_rule, est_latency_llm=est_latency_llm,
        est_cost_llm=est_cost_llm,
    )


# ---------- Public helper: simulation ----------

def route_query(
    user_input: str,
    *,
    mock_classification: Optional[Mapping[str, Any]] = None,
    context: Optional[Mapping[str, Any]] = None,
    options: Optional[Mapping[str, Any]] = None,
    config: Optional[RouterConfig] = None,
    **overrides: Any,
) -> RoutePlan:
    """
    Simulate routing with optional mock classification.

    If `mock_classification` is None, a lightweight heuristic classifier is used:
    - complexity based on length & keyword joins
    - domain guessed from keywords

    This does NOT execute rule_based or llm_based. It only returns a RoutePlan.
    You can pass overrides like:
        - cost_estimator=..., latency_estimator=..., domain_policy=...

    Examples
    --------
    >>> plan = route_query("total sales last week by region")
    >>> plan.strategy  # "rule_based" (likely)
    """
    classification = mock_classification or _mock_classify(user_input)
    return route(
        classification=classification,
        user_input=user_input,
        context=context,
        options=options,
        config=config,
        cost_estimator=overrides.get("cost_estimator", default_cost_estimator),
        latency_estimator=overrides.get("latency_estimator", default_latency_estimator),
        domain_policy=overrides.get("domain_policy", lambda d, c, **k: default_domain_policy(d, c, **k)),
    )


# ---------- Internals ----------

def _finalize(
    log,
    cfg: RouterConfig,
    ctx: Mapping[str, Any],
    opts: Mapping[str, Any],
    classification: Mapping[str, Any],
    user_input: str,
    *,
    strategy: str,
    reason: str,
    model: str,
    tokens: int,
    est_latency_rule: int,
    est_latency_llm: int,
    est_cost_llm: float,
) -> RoutePlan:
    plan = RoutePlan(
        strategy=strategy,
        reason=reason,
        cost_tier=_cost_tier(est_cost_llm if strategy == "llm_based" else 0.0),
        estimates={
            "tokens": tokens,
            "cost_usd_llm": est_cost_llm,
            "latency_ms_rule": est_latency_rule,
            "latency_ms_llm": est_latency_llm,
            "model": model,
        },
        constraints=_constraints_snapshot(cfg, ctx, opts),
        metadata=_metadata_snapshot(classification, user_input),
    )
    _log_route(log, plan, event="route_decision")
    return plan


def _constraints_snapshot(cfg: RouterConfig, ctx: Mapping[str, Any], opts: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "complexity_threshold": cfg.complexity_threshold,
        "allowed_llm_domains": sorted(cfg.allowed_llm_domains) if cfg.allowed_llm_domains else None,
        "prefer_rule_based_domains": sorted(cfg.prefer_rule_based_domains),
        "hard_llm_budget_per_query": float(opts.get("cost_hard_limit", cfg.hard_llm_budget_per_query)),
        "soft_daily_budget_remaining": float(opts.get("daily_budget_remaining", cfg.soft_daily_budget_remaining)),
        "latency_slo_ms": int(ctx.get("latency_slo_ms", cfg.latency_slo_ms)),
        "allow_llm": bool(opts.get("allow_llm", cfg.allow_llm)),
    }


def _metadata_snapshot(classification: Mapping[str, Any], user_input: str) -> Dict[str, Any]:
    return {
        "domain": classification.get("domain"),
        "complexity": float(_complexity_score(classification)),
        "safety": classification.get("safety"),
        "entities_count": classification.get("entities_count"),
        "estimated_joins": classification.get("estimated_joins"),
        "user_input_preview": (user_input[:120] + "…") if len(user_input) > 120 else user_input,
    }


def _cost_tier(cost_usd: float) -> str:
    if cost_usd <= 0.001:
        return "low"
    if cost_usd <= 0.01:
        return "medium"
    return "high"


def _log_route(logger, plan: RoutePlan, *, event: str) -> None:
    try:
        logger.info(
            f"router_decision strategy={plan.strategy} reason={plan.reason}",
            extra={
                "event": event,
                "query_id": None,  # nip_query.run_query will inject a real query_id
                "ts": None,
                "routing_plan": plan.to_dict(),
            },
        )
    except Exception:
        # Never break the caller because of logging
        pass


def _mock_classify(user_input: str) -> Dict[str, Any]:
    """
    Tiny heuristic classifier for simulations and unit tests.
    """
    text = user_input.lower()
    domain_map = {
        "revenue": "finance",
        "invoice": "billing",
        "tax": "compliance",
        "churn": "customer_success",
        "customer": "sales",
        "order": "sales",
        "inventory": "supply_chain",
        "latency": "platform",
    }
    domain = None
    for key, dom in domain_map.items():
        if key in text:
            domain = dom
            break

    # Estimate joins/entities/complexity
    join_keywords = sum(k in text for k in (" join ", " vs ", " compare ", " correlation ", " by "))
    entities = sum(k in text for k in (" product", " sku", " region", " country", " channel", " city", " month", " quarter"))
    length_complexity = min(1.0, len(user_input) / 200.0)

    complexity = min(1.0, 0.15 + 0.2 * join_keywords + 0.05 * entities + 0.4 * length_complexity)

    safety = {
        "pii": any(k in text for k in ("ssn", "aadhaar", "credit card", "dob")),
        "compliance_sensitive": "tax" in text or "gdpr" in text,
    }

    return {
        "domain": domain,
        "complexity": complexity,
        "estimated_joins": join_keywords,
        "entities_count": entities,
        "safety": safety,
        "token_estimate": max(32, len(user_input) // 4),
    }


__all__ = [
    "RoutePlan",
    "RouterConfig",
    "route",
    "route_query",
    "default_cost_estimator",
    "default_latency_estimator",
    "default_domain_policy",
]
