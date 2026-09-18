from typing import Dict, Any, Optional, List
import numpy as np

from core.contracts import ProposalInput, EvaluationResult, QualitativeAssessment

# وارد کردن تمامی موتورهای محاسباتی هسته سامانه
from core.monte_carlo import MonteCarloEngine
from core.energy_simulation import EnergyRiskEngine
from core.reliability_engine import ReliabilityEngine
from core.supply_chain_engine import SupplyChainEngine
from core.carbon_tax_engine import CarbonTaxEngine
from core.llm_provider import BaseLLMProvider, MockLLMProvider
from core.rag_engine import RAGEngine, create_rag_engine


class ExAnteAgent:
    """
    عامل ارزیابی پیشینی جامع:
    هماهنگ‌کننده و مجری زنجیره‌ای موتورهای شبیه‌سازی مالی، ریسک‌های عملیاتی (انرژی، MTBF، لجستیک)
    و مشوق‌های مالیاتی جهت تجمیع شاخص‌های تعدیل‌شده پروژه.
    """
    def __init__(self, llm_provider: BaseLLMProvider, rag_engine: Optional[RAGEngine] = None):
        """
        :param llm_provider: نمونه ارائه‌دهنده سرویس هوش مصنوعی (Mock یا Real)
        :param rag_engine: Optional RAG engine for context retrieval (injected for testing).
        """
        self.llm_provider = llm_provider
        self.mc_engine = MonteCarloEngine(num_simulations=10000)
        self.energy_engine = EnergyRiskEngine(num_simulations=10000)
        self.reliability_engine = ReliabilityEngine(num_simulations=10000)
        self.supply_chain_engine = SupplyChainEngine(num_simulations=10000)
        self.tax_engine = CarbonTaxEngine(num_simulations=10000)
        self.rag_engine = rag_engine or create_rag_engine(provider_type="mock")

    def evaluate_comprehensive_proposal(
        self,
        proposal_data: Dict[str, Any],
        raw_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        ارزیابی یکپارچه پروپوزال با اجرای تمام ماژول‌های ریسک و تجمیع نتایج.

        Phase 2 contract enforcement: the loose input dict is validated through
        the Pydantic ProposalInput model at the boundary; the returned dict is
        validated through EvaluationResult before leaving the agent.
        """
        # -- Contract enforcement at the boundary (raises ValidationError with
        #    precise field errors on inverted triplets / non-positive values) --
        validated = ProposalInput.model_validate(proposal_data)
        proposal_data = validated.model_dump()
        years = validated.years

        # ۱. اجرای شبیه‌سازی مالی و تورم فیشر
        cost_triplet = (
            float(proposal_data.get("cost_p10", 8000.0)),
            float(proposal_data.get("cost_p50", 10000.0)),
            float(proposal_data.get("cost_p90", 14000.0))
        )
        benefit_triplet = (
            float(proposal_data.get("benefit_p10", 3000.0)),
            float(proposal_data.get("benefit_p50", 5000.0)),
            float(proposal_data.get("benefit_p90", 8000.0))
        )
        
        mc_results = self.mc_engine.run_roi_simulation(
            cost_p10_p50_p90=cost_triplet,
            benefit_p10_p50_p90=benefit_triplet,
            inflation_p10_p50_p90=proposal_data.get("inflation_triplet", (0.35, 0.50, 0.70)),
            fx_growth_p10_p50_p90=proposal_data.get("fx_growth_triplet", (0.30, 0.45, 0.65)),
            currency_type=proposal_data.get("currency_type", "USD"),
            base_fx_rate=float(proposal_data.get("base_fx_rate", 65000.0)),
            annual_fx_savings=float(proposal_data.get("annual_fx_savings", 0.0)),
            p_success=float(proposal_data.get("p_success", 0.85)),
            years=years
        )

        # ۲. شبیه‌سازی ریسک ناترازی انرژی (برق و گاز)
        energy_loss = 0.0
        energy_outage_days = 0.0
        if proposal_data.get("enable_energy_risk", True):
            energy_res = self.energy_engine.simulate_energy_impact(
                base_annual_benefit=benefit_triplet[1],
                power_outage_days_triplet=proposal_data.get("power_outage_triplet", (10.0, 20.0, 35.0)),
                gas_outage_days_triplet=proposal_data.get("gas_outage_triplet", (15.0, 30.0, 45.0)),
                daily_downtime_loss=float(proposal_data.get("energy_daily_loss", 50.0)),
                years=years
            )
            energy_loss = energy_res["mean_annual_loss_toman"]
            energy_outage_days = energy_res["mean_outage_days_yearly"]

        # ۳. شبیه‌سازی قابلیت اطمینان و توقفات تجهیز (MTBF)
        downtime_loss = 0.0
        downtime_hours = 0.0
        if proposal_data.get("enable_reliability_risk", True):
            rel_res = self.reliability_engine.simulate_downtime_risk(
                base_annual_benefit=benefit_triplet[1],
                mtbf_hours_triplet=proposal_data.get("mtbf_triplet", (500.0, 1000.0, 1500.0)),
                mttr_hours_triplet=proposal_data.get("mttr_triplet", (2.0, 4.0, 8.0)),
                hourly_downtime_loss=float(proposal_data.get("hourly_downtime_loss", 20.0)),
                annual_operating_hours=float(proposal_data.get("annual_operating_hours", 7200.0)),
                years=years
            )
            downtime_loss = rel_res["mean_annual_downtime_loss"]
            downtime_hours = rel_res["mean_downtime_hours"]

        # ۴. شبیه‌سازی ریسک تأخیر لجستیک و زنجیره تأمین
        delay_days = 0.0
        if proposal_data.get("enable_supply_chain_risk", True):
            sc_res = self.supply_chain_engine.simulate_lead_time_risk(
                planned_lead_time_days=float(proposal_data.get("planned_lead_time", 90.0)),
                actual_lead_time_triplet=proposal_data.get("actual_lead_time_triplet", (75.0, 110.0, 180.0)),
                daily_delay_cost=float(proposal_data.get("daily_delay_cost", 12.0))
            )
            delay_days = sc_res["mean_delay_days"]

        # ۵. مشوق‌ها و معافیت‌های مالیاتی (قانون جهش تولید ایران و CBAM اروپا)
        iran_tax_credit = 0.0
        if proposal_data.get("enable_iran_tax", True):
            iran_tax_res = self.tax_engine.simulate_iran_tax_credit(
                rd_cost_triplet=cost_triplet,
                approved_cost_pct_triplet=proposal_data.get("approved_pct_triplet", (0.60, 0.80, 0.95)),
                corporate_tax_rate=float(proposal_data.get("corporate_tax_rate", 0.20))
            )
            iran_tax_credit = iran_tax_res["mean_iran_tax_credit_savings"]

        carbon_savings = 0.0
        if proposal_data.get("enable_cbam_tax", False):
            cbam_res = self.tax_engine.simulate_eu_cbam_benefit(
                annual_export_tons_triplet=proposal_data.get("export_tons_triplet", (100000.0, 200000.0, 300000.0)),
                co2_reduction_per_ton_kg=float(proposal_data.get("co2_reduction_kg", 150.0)),
                carbon_tax_per_ton_usd_triplet=proposal_data.get("carbon_tax_usd_triplet", (60.0, 85.0, 120.0)),
                usd_to_toman_rate=float(proposal_data.get("base_fx_rate", 65000.0)),
                years=years
            )
            carbon_savings = cbam_res["mean_annual_carbon_savings_toman"]

        # ۶. تجمیع ریاضی شاخص‌ها
        total_downtime = downtime_hours + (energy_outage_days * 24.0)

        # منفعت خالص تعدیل‌شده سالانه (میلیون تومان)
        base_annual_p50 = benefit_triplet[1]
        adjusted_benefit = base_annual_p50 - energy_loss - downtime_loss + iran_tax_credit + carbon_savings
        adjusted_benefit = max(adjusted_benefit, 0.0)

        # ۷. بازیابی متنی RAG در صورت وجود سند
        rag_evidence = []
        rag_context = ""
        if raw_text:
            self.rag_engine.index_document(doc_id=proposal_data.get("title", "doc"), text=raw_text)
            # جستجوی مرتبط با پرسش پیشینی
            rag_chunks = self.rag_engine.retrieve_context(
                query="ریسک هزینه فنی توقف پروژه بومی‌سازی مصرف گاز سنسور مالی فناوری",
                top_k=3
            )
            rag_evidence = [
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "similarity": chunk.similarity,
                }
                for chunk in rag_chunks
            ]
            # Build context string for prompt injection
            if rag_chunks:
                context_parts = [f"[{i+1}] {chunk.text}" for i, chunk in enumerate(rag_chunks)]
                rag_context = "<context>\n" + "\n".join(context_parts) + "\n</context>\n\n"

        # ۸. تولید پاسخ کیفی با LLM (schema-bound, with no-crash fallback)
        prompt = (
            f"{rag_context}"
            f"Evaluate proposal '{proposal_data.get('title')}' with ROI = {mc_results['mean_roi']}%, "
            f"VaR 95% = {mc_results['var_95']}%, Adjusted Benefit = {adjusted_benefit:,.0f} MToman."
        )
        llm_assessment = self._qualitative_assessment(prompt)

        status = "HIGH_CONFIDENCE" if mc_results["var_95"] > 0 else "MODERATE_RISK"

        # ساخت ساختار خروجی استاندارد
        consolidated_output = {
            "title": proposal_data.get("title", "بدون عنوان"),
            "cost_p10": cost_triplet[0],
            "cost_p50": cost_triplet[1],
            "cost_p90": cost_triplet[2],
            "benefit_p10": benefit_triplet[0],
            "benefit_p50": benefit_triplet[1],
            "benefit_p90": benefit_triplet[2],
            "p_success": float(proposal_data.get("p_success", 0.85)),
            "total_downtime_hours": round(float(total_downtime), 1),
            "energy_loss_toman": round(float(energy_loss), 2),
            "downtime_loss_toman": round(float(downtime_loss), 2),
            "lead_time_delay_days": round(float(delay_days), 1),
            "iran_tax_credit_toman": round(float(iran_tax_credit), 2),
            "carbon_savings_toman": round(float(carbon_savings), 2),
            "adjusted_net_benefit": round(float(adjusted_benefit), 2),
            "mean_roi": mc_results["mean_roi"],
            "var_95": mc_results["var_95"],
            "probability_of_loss": mc_results["probability_of_loss"],
            "risk_status": status,
            "quantitative_metrics": mc_results,
            "qualitative_assessment": llm_assessment,
            "rag_evidence": rag_evidence
        }

        # Output contract enforcement: guarantees every field the downstream
        # layers (repository, reporter, portfolio) rely on exists and is valid.
        EvaluationResult.model_validate(consolidated_output)
        return consolidated_output

    def evaluate_proposal(self, proposal_data: Dict[str, Any], raw_text: Optional[str] = None) -> Dict[str, Any]:
        return self.evaluate_comprehensive_proposal(proposal_data, raw_text=raw_text)

    def _qualitative_assessment(self, prompt: str) -> Dict[str, Any]:
        """
        Phase 3: schema-bound qualitative assessment.

        Prefers the provider's retrying generate_validated() (Ollama); any
        provider that only implements the legacy generate_json() contract is
        validated locally. On schema violation the deterministic mock path
        supplies the assessment - the evaluation NEVER crashes and NEVER
        returns unvalidated LLM output (SKILL.md no-crash guarantee).
        """
        generate_validated = getattr(self.llm_provider, "generate_validated", None)
        if callable(generate_validated):
            try:
                return generate_validated(
                    prompt,
                    schema_description="QualitativeAssessment",
                    schema_model=QualitativeAssessment,
                )
            except Exception:  # noqa: BLE001 - degrade, never crash
                pass
        else:
            try:
                raw = self.llm_provider.generate_json(
                    prompt, schema_description="QualitativeAssessment"
                )
                return QualitativeAssessment.model_validate(raw).model_dump()
            except Exception:  # noqa: BLE001 - degrade, never crash
                pass

        # Deterministic fallback: validate the mock output too, so even the
        # degraded path is contract-clean.
        return QualitativeAssessment.model_validate(
            MockLLMProvider().generate_json(prompt, "QualitativeAssessment")
        ).model_dump()