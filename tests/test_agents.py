"""Specialist agent behaviour with deterministic mock LLM outputs."""

from __future__ import annotations

from agents.budget import BudgetAgent
from agents.orchestrator import OrchestratorAgent
from agents.policy import PolicyAgent
from agents.risk import RiskAssessor
from agents.supplier import SupplierAgent
from app.schemas import BudgetResult


def _state(amount: float = 30_000, item: str = "laptops", quantity: int = 20) -> dict:
    return {
        "thread_id": "test-thread",
        "request_id": "test-thread",
        "requester": "demo@example.com",
        "user_request": f"Buy {quantity} {item}, budget {amount} USD",
        "parsed_request": {
            "item": item,
            "quantity": quantity,
            "total_amount_usd": amount,
            "currency": "USD",
            "category": "laptops" if item == "laptops" else "general",
            "specifications": [],
            "confidence": 0.9,
        },
        "orchestrator": {"confidence": 0.9},
        "plan": [],
        "task_index": 0,
        "iterations": 0,
        "errors": [],
    }


def test_policy_agent_flags_cfo_threshold(runtime):
    agent = PolicyAgent(runtime.llm, runtime.settings, runtime.retriever)
    result = agent.run(_state(amount=30_000))["policy"]

    assert result["requires_approval"] is True
    assert result["compliant"] is True
    assert result["confidence"] >= 0.75
    assert result["citations"], "policy findings must be auditable"
    assert all(citation["doc_type"] == "policy" for citation in result["citations"])


def test_policy_agent_allows_small_purchase(runtime):
    agent = PolicyAgent(runtime.llm, runtime.settings, runtime.retriever)
    result = agent.run(_state(amount=5_000))["policy"]

    assert result["requires_approval"] is False
    assert result["compliant"] is True


def test_supplier_agent_filters_to_requested_category(runtime):
    agent = SupplierAgent(runtime.llm, runtime.settings, runtime.retriever)
    result = agent.run(_state())["suppliers"]

    names = [candidate["name"] for candidate in result["candidates"]]
    assert "Orbit Systems" in names
    assert "NovaTech Devices" in names
    assert all(name not in {"Clarity Displays", "GreenSeat"} for name in names)
    assert result["recommended_supplier"] in names
    assert result["confidence"] > 0
    assert result["citations"]
    assert all(citation["doc_type"] == "supplier" for citation in result["citations"])


def test_supplier_agent_computes_total_price(runtime):
    agent = SupplierAgent(runtime.llm, runtime.settings, runtime.retriever)
    result = agent.run(_state(quantity=20))["suppliers"]

    by_name = {candidate["name"]: candidate for candidate in result["candidates"]}
    # NovaTech lists USD 1,150 with a documented 5% discount at 20+ units.
    assert by_name["NovaTech Devices"]["total_price_usd"] == 21_850.0
    assert by_name["BudgetByte Refurbishing"]["total_price_usd"] == 14_400.0


def test_budget_agent_uses_labelled_demo_budget(runtime):
    agent = BudgetAgent(runtime.llm, runtime.settings, runtime.budget_provider)
    result = agent.run(_state(amount=30_000))["budget"]

    assert result["source"] == "local_demo"
    assert result["available_usd"] == 78_000.0
    assert result["within_budget"] is True


def test_budget_agent_reports_over_budget(runtime):
    agent = BudgetAgent(runtime.llm, runtime.settings, runtime.budget_provider)
    result = agent.run(_state(amount=100_000))["budget"]

    assert result["within_budget"] is False
    assert "exceeds" in result["notes"].lower()


def test_budget_agent_surfaces_connector_errors(runtime):
    class BrokenProvider:
        def fetch_budget(self):
            from connectors.google_sheets import BudgetConnectorError

            raise BudgetConnectorError("sheets credentials rejected")

    agent = BudgetAgent(runtime.llm, runtime.settings, BrokenProvider())
    update = agent.run(_state(amount=30_000))
    result = BudgetResult.model_validate(update["budget"])

    assert result.within_budget is None
    assert result.source == "unavailable"
    assert update["errors"][0]["agent"] == "budget"


def test_risk_assessor_requires_approval_above_threshold(runtime):
    state = _state(amount=30_000)
    state.update(
        {
            "policy": {"compliant": True, "requires_approval": True, "confidence": 0.9},
            "suppliers": {"recommended_supplier": "Orbit Systems", "confidence": 0.88},
            "budget": {"within_budget": True, "confidence": 0.85},
        }
    )
    assessment = RiskAssessor(runtime.settings).assess(state)

    assert assessment.approval_required is True
    assert assessment.amount_usd == 30_000
    assert any("exceeds" in reason for reason in assessment.reasons)


def test_risk_assessor_requires_approval_on_missing_data(runtime):
    state = _state(amount=0)
    state.update(
        {
            "policy": {"compliant": False, "requires_approval": True, "confidence": 0.2},
            "suppliers": {},
            "budget": {"within_budget": None, "confidence": 0.0},
            "errors": [{"agent": "policy", "message": "failed"}],
        }
    )
    assessment = RiskAssessor(runtime.settings).assess(state)

    assert assessment.approval_required is True
    assert any("missing" in reason.lower() for reason in assessment.reasons)
    assert any("budget" in reason.lower() for reason in assessment.reasons)


def test_risk_assessor_allows_clean_low_value_request(runtime):
    state = _state(amount=5_000)
    state.update(
        {
            "policy": {"compliant": True, "requires_approval": False, "confidence": 0.9},
            "suppliers": {"recommended_supplier": "Clarity Displays", "confidence": 0.88},
            "budget": {"within_budget": True, "confidence": 0.85},
            "errors": [],
        }
    )
    assessment = RiskAssessor(runtime.settings).assess(state)

    assert assessment.approval_required is False
    assert assessment.reasons == []


def test_orchestrator_produces_all_specialist_tasks(runtime):
    agent = OrchestratorAgent(runtime.llm, runtime.settings, runtime.retriever)
    update = agent.run(
        {
            "user_request": "Buy 20 laptops, budget 30k USD",
            "requester": "demo@example.com",
            "errors": [],
        }
    )

    assert update["parsed_request"]["quantity"] == 20
    assert update["parsed_request"]["total_amount_usd"] == 30_000
    assert [task["agent"] for task in update["plan"]] == ["policy", "suppliers", "budget"]
    assert update["status"] == "processing"


def test_orchestrator_falls_back_safely_when_llm_fails(runtime, monkeypatch):
    agent = OrchestratorAgent(runtime.llm, runtime.settings, runtime.retriever)

    def fail(*args, **kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(agent.llm, "generate_structured", fail)
    update = agent.run(
        {
            "user_request": "Buy 4 monitors, budget 2k USD",
            "requester": "demo@example.com",
            "errors": [],
        }
    )

    assert update["orchestrator"]["source"] == "fallback_parser"
    assert update["errors"][0]["agent"] == "orchestrator"
    assert update["parsed_request"]["quantity"] == 4
    assert update["parsed_request"]["total_amount_usd"] == 2000
