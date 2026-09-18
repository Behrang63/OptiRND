"""
Tests for core/contracts.py - Pydantic v2 boundary enforcement.

Verifies: valid payloads pass, inverted triplets / non-positive financials /
out-of-range probabilities / invalid currency literals are rejected with
field-precise errors.
"""
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from core.contracts import (  # noqa: E402
    EvaluationResult,
    PortfolioProject,
    ProposalInput,
    QualitativeAssessment,
)


VALID = {
    "title": "سامانه هوشمند عیب‌یابی خط نورد",
    "cost_p10": 3000.0, "cost_p50": 4000.0, "cost_p90": 5500.0,
    "benefit_p10": 6000.0, "benefit_p50": 8000.0, "benefit_p90": 11000.0,
}


def test_valid_proposal_input_passes():
    m = ProposalInput.model_validate(VALID)
    assert m.years == 3 and m.currency_type == "USD"
    assert m.enable_energy_risk is True


def test_inverted_cost_triplet_rejected_with_field_error():
    bad = dict(VALID, cost_p10=6000.0)  # p10 > p50
    with pytest.raises(ValidationError) as e:
        ProposalInput.model_validate(bad)
    assert "cost" in str(e.value).lower()


def test_inverted_nested_triplet_rejected():
    bad = dict(VALID, mtbf_triplet=(2000.0, 900.0, 1200.0))
    with pytest.raises(ValidationError):
        ProposalInput.model_validate(bad)


def test_non_positive_financial_rejected():
    bad = dict(VALID, benefit_p50=0.0)
    with pytest.raises(ValidationError):
        ProposalInput.model_validate(bad)


def test_probability_out_of_range_rejected():
    bad = dict(VALID, p_success=1.5)
    with pytest.raises(ValidationError):
        ProposalInput.model_validate(bad)


def test_invalid_currency_literal_rejected():
    bad = dict(VALID, currency_type="GBP")
    with pytest.raises(ValidationError):
        ProposalInput.model_validate(bad)


def test_years_out_of_bounds_rejected():
    bad = dict(VALID, years=0)
    with pytest.raises(ValidationError):
        ProposalInput.model_validate(bad)


def test_cbam_enabled_requires_positive_export_volume():
    bad = dict(VALID, enable_cbam_tax=True)  # default export triplet is zeros
    with pytest.raises(ValidationError):
        ProposalInput.model_validate(bad)
    # And with volume provided it passes:
    ok = ProposalInput.model_validate(
        dict(VALID, enable_cbam_tax=True, export_tons_triplet=(100000.0, 200000.0, 300000.0))
    )
    assert ok.enable_cbam_tax is True


def test_qualitative_assessment_clamps_extra_fields():
    q = QualitativeAssessment.model_validate(
        {"technical_success_probability": 0.9, "risk_summary": "ok", "unknown_key": 1}
    )
    assert q.risk_summary == "ok"


def test_evaluation_result_contract_roundtrip():
    data = {
        "title": "پروژه", "cost_p10": 1.0, "cost_p50": 2.0, "cost_p90": 3.0,
        "benefit_p10": 4.0, "benefit_p50": 5.0, "benefit_p90": 6.0,
        "p_success": 0.9, "total_downtime_hours": 1.0, "energy_loss_toman": 0.0,
        "downtime_loss_toman": 0.0, "lead_time_delay_days": 0.0,
        "iran_tax_credit_toman": 0.0, "carbon_savings_toman": 0.0,
        "adjusted_net_benefit": 0.0, "mean_roi": 10.0, "var_95": 1.0,
        "probability_of_loss": 5.0, "risk_status": "HIGH_CONFIDENCE",
        "quantitative_metrics": {},
        "qualitative_assessment": {"risk_summary": "fine"},
    }
    m = EvaluationResult.model_validate(data)
    assert m.risk_status == "HIGH_CONFIDENCE"
    # invalid enum value rejected
    bad = dict(data, risk_status="SOMETHING_ELSE")
    with pytest.raises(ValidationError):
        EvaluationResult.model_validate(bad)


def test_portfolio_project_rejects_negative_downtime():
    with pytest.raises(ValidationError):
        PortfolioProject.model_validate({
            "title": "x", "cost_p10": 1, "cost_p50": 2, "cost_p90": 3,
            "benefit_p10": 1, "benefit_p50": 2, "benefit_p90": 3,
            "dt_p10": -1, "dt_p50": 0, "dt_p90": 0,
        })


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
