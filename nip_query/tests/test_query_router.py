# tests/test_query_router.py
import pytest
from nip_query.query_router import route, RouterConfig, route_query

def test_low_complexity_prefers_rule_based():
    cls = {"complexity": 0.2, "domain": "sales"}
    plan = route(classification=cls, user_input="total sales last week")
    assert plan.strategy == "rule_based"
    assert plan.reason == "below_complexity_threshold"

def test_high_complexity_prefers_llm():
    cls = {"complexity": 0.9, "domain": "sales"}
    plan = route(classification=cls, user_input="compare revenue growth by region and channel QoQ with anomaly detection")
    assert plan.strategy == "llm_based"

def test_cost_guard_blocks_llm():
    cls = {"complexity": 0.9, "domain": "sales", "token_estimate": 20000}
    plan = route(classification=cls, user_input="complex query", options={"cost_hard_limit": 1e-6})
    assert plan.strategy == "rule_based"
    assert plan.reason == "cost_guard_triggered"

def test_domain_policy_prefers_rule_based():
    cls = {"complexity": 0.6, "domain": "finance"}
    plan = route(classification=cls, user_input="revenue recognition by quarter")
    assert plan.strategy == "rule_based"
    assert plan.reason == "domain_policy_preference"

def test_forced_strategy_override():
    cls = {"complexity": 0.1, "domain": "sales"}
    plan = route(classification=cls, user_input="simple", context={"force_strategy": "llm_based"})
    assert plan.strategy == "llm_based"
    assert plan.reason == "forced_by_context_option"
  
