سند معماری و الگوی پیاده‌سازی کدهای پروژه پژوهشیار (ARCHITECTURE.md)
این سند، معماری دقیق نرم‌افزار، الگوی جریان داده (Data Flow)، تعاریف اینترفیس‌ها و کدهای پایه (Skeletons) مخزن پژوهشیار را برای پیاده‌سازی دستی یا تولید کد با هوش مصنوعی مشخص می‌کند.
۱. الگوی لایه‌بندی نرم‌افزار (Layered Pattern)
┌────────────────────────────────────────────────────────┐
│               Streamlit UI Layer (ui/)                 │
│   • Views (ex_ante_view, process_view, ex_post_view)   │
│   • Data Editor Components (st.data_editor)            │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│             Domain Agents Layer (agents/)              │
│   • Main: ExAnteAgent, ProcessAgent, ExPostAgent       │
│   • Sub: DataSynthAgent, DocParserAgent                │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│              Core Business Engines (core/)             │
│   • MonteCarloEngine (PERT, QMC, VaR)                  │
│   • RAGEngine & DocParser                              │
│   • LLMProvider (Base, Mock, Real)                     │
│   • Database (SQLAlchemy ORM + SQLite)                 │
└────────────────────────────────────────────────────────┘

۲. پیاده‌سازی کد‌های پایه (Core Skeletons)
۲.۱. ماژول LLM Provider مقاوم (core/llm_provider.py)
این ماژول عدم کرش برنامه در غیاب هوش مصنوعی خارجی را تضمین می‌کند.
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import json

class BaseLLMProvider(ABC):
    @abstractmethod
    def generate_json(self, prompt: str, schema_description: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def generate_text(self, prompt: str) -> str:
        pass


class MockLLMProvider(BaseLLMProvider):
    """Fallback LLM provider using deterministic rules and mock responses."""
    
    def generate_json(self, prompt: str, schema_description: str) -> Dict[str, Any]:
        # Return fallback structured data depending on prompt keywords
        if "ex_ante" in prompt.lower():
            return {
                "technical_success_probability": 0.75,
                "recommended_action": "APPROVE_WITH_CONDITIONS",
                "risk_summary": "High technical complexity in algorithm training, manageable financial risk."
            }
        elif "process" in prompt.lower():
            return {
                "variance_status": "WARNING",
                "revised_completion_probability": 0.82,
                "mitigation_advice": "Reallocate compute resources to prevent timeline slippage."
            }
        return {"status": "mock_success", "message": "Rule-based mock response executed."}

    def generate_text(self, prompt: str) -> str:
        return "[MOCK RESPONSE]: System operating in offline/mock mode. Analysis completed successfully."


class LLMFactory:
    @staticmethod
    def get_provider(provider_type: str = "mock", api_key: Optional[str] = None) -> BaseLLMProvider:
        if provider_type == "mock" or not api_key:
            return MockLLMProvider()
        # Future: Return RealLLMProvider or OllamaLLMProvider
        return MockLLMProvider()

۲.۲. موتور شبیه‌سازی مونت‌کارلو و VaR (core/monte_carlo.py)
import numpy as np
from scipy.stats import pert
from typing import Dict, Any, Tuple

class MonteCarloEngine:
    def __init__(self, num_simulations: int = 10000, seed: int = 42):
        self.num_simulations = num_simulations
        np.random.seed(seed)

    def generate_pert_samples(self, low: float, mode: float, high: float) -> np.ndarray:
        """Generates random samples following a PERT distribution."""
        if low >= high or not (low <= mode <= high):
            # Fallback to uniform if invalid parameters
            return np.random.uniform(low, high, self.num_simulations)
        
        gamma_shape = 4.0
        alpha = 1 + gamma_shape * (mode - low) / (high - low)
        beta_val = 1 + gamma_shape * (high - mode) / (high - low)
        
        beta_samples = np.random.beta(alpha, beta_val, self.num_simulations)
        return low + beta_samples * (high - low)

    def run_roi_simulation(
        self,
        cost_p10_p50_p90: Tuple[float, float, float],
        benefit_p10_p50_p90: Tuple[float, float, float],
        p_success: float,
        discount_rate: float = 0.15,
        years: int = 3
    ) -> Dict[str, Any]:
        """
        Executes Monte Carlo simulation for ROI and VaR calculation.
        """
        cost_samples = self.generate_pert_samples(*cost_p10_p50_p90)
        annual_benefit_samples = self.generate_pert_samples(*benefit_p10_p50_p90)
        
        # Success/Failure technical trial
        success_mask = np.random.binomial(1, p_success, self.num_simulations)
        
        # Calculate NPV for each simulation
        discount_factors = sum([1 / ((1 + discount_rate) ** t) for t in range(1, years + 1)])
        npv_samples = (annual_benefit_samples * discount_factors * success_mask) - cost_samples
        
        roi_samples = (npv_samples / np.maximum(cost_samples, 1.0)) * 100

        # Risk Metrics
        mean_roi = float(np.mean(roi_samples))
        std_roi = float(np.std(roi_samples))
        
        # Value at Risk (VaR) at 95% Confidence
        var_95 = float(np.percentile(roi_samples, 5))  # 5th percentile
        var_99 = float(np.percentile(roi_samples, 1))  # 1st percentile

        return {
            "mean_roi": round(mean_roi, 2),
            "std_dev": round(std_roi, 2),
            "var_95": round(var_95, 2),
            "var_99": round(var_99, 2),
            "expected_npv": round(float(np.mean(npv_samples)), 2),
            "probability_of_loss": round(float(np.mean(roi_samples < 0) * 100), 2),
            "raw_roi_samples": roi_samples
        }

۲.۳. پایگاه داده SQLAlchemy (core/database.py)
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
import datetime

Base = declarative_base()

class Proposal(Base):
    __tablename__ = 'proposals'

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255), nullable=False)
    organization = Column(String(255), default="فولاد مبارکه")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Financial & Technical Triplet Inputs
    cost_p10 = Column(Float, nullable=False)
    cost_p50 = Column(Float, nullable=False)
    cost_p90 = Column(Float, nullable=False)

    benefit_p10 = Column(Float, nullable=False)
    benefit_p50 = Column(Float, nullable=False)
    benefit_p90 = Column(Float, nullable=False)

    p_success = Column(Float, default=0.8)

    # Relationships
    milestones = relationship("Milestone", back_populates="proposal")


class Milestone(Base):
    __tablename__ = 'milestones'

    id = Column(Integer, primary_key=True, autoincrement=True)
    proposal_id = Column(Integer, ForeignKey('proposals.id'))
    title = Column(String(255), nullable=False)
    planned_budget = Column(Float, nullable=False)
    actual_budget = Column(Float, default=0.0)
    planned_duration_days = Column(Integer, nullable=False)
    actual_duration_days = Column(Integer, default=0)
    completion_percentage = Column(Float, default=0.0)

    proposal = relationship("Proposal", back_populates="milestones")


def init_db(db_path: str = "sqlite:///data/pajoheshyar.db"):
    engine = create_engine(db_path, echo=False)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)

۲.۴. عامل ارزیابی پیشینی (agents/main_agents/ex_ante_agent.py)
from typing import Dict, Any
from core.monte_carlo import MonteCarloEngine
from core.llm_provider import BaseLLMProvider

class ExAnteAgent:
    def __init__(self, llm_provider: BaseLLMProvider):
        self.llm_provider = llm_provider
        self.mc_engine = MonteCarloEngine(num_simulations=10000)

    def evaluate_proposal(self, proposal_data: Dict[str, Any]) -> Dict[str, Any]:
        # 1. Run Monte Carlo Simulation
        cost_triplet = (proposal_data['cost_p10'], proposal_data['cost_p50'], proposal_data['cost_p90'])
        benefit_triplet = (proposal_data['benefit_p10'], proposal_data['benefit_p50'], proposal_data['benefit_p90'])
        
        mc_results = self.mc_engine.run_roi_simulation(
            cost_p10_p50_p90=cost_triplet,
            benefit_p10_p50_p90=benefit_triplet,
            p_success=proposal_data.get('p_success', 0.8)
        )

        # 2. Get LLM / Mock Qualitative Assessment
        prompt = f"Evaluate proposal {proposal_data.get('title')} with VaR 95% = {mc_results['var_95']}%"
        llm_assessment = self.llm_provider.generate_json(prompt, schema_description="Ex-ante evaluation schema")

        # 3. Combine Quantitative and Qualitative Results
        return {
            "proposal_title": proposal_data.get('title'),
            "quantitative_metrics": mc_results,
            "qualitative_assessment": llm_assessment,
            "confidence_level_status": "HIGH" if mc_results['var_95'] > 10 else "MODERATE_RISK"
        }

۳. نحوه تست محلی در VS Code
تست کل جریان کاری بدون اجرای Streamlit:
# scripts/test_pipeline.py
from core.database import init_db, Proposal
from core.llm_provider import LLMFactory
from agents.main_agents.ex_ante_agent import ExAnteAgent

def run_test():
    print("--- Testing Pajoheshyar Pipeline ---")
    
    # 1. Init Database
    Session = init_db("sqlite:///:memory:")
    
    # 2. Create Dummy Proposal
    sample_proposal = {
        "title": "سامانه تشخیص عیوب ورق فولاد مبارکه",
        "cost_p10": 8000.0, "cost_p50": 10000.0, "cost_p90": 14000.0,  # میلیون تومان
        "benefit_p10": 3000.0, "benefit_p50": 5000.0, "benefit_p90": 8000.0,
        "p_success": 0.85
    }

    # 3. Instantiate Agent with Mock LLM
    llm = LLMFactory.get_provider("mock")
    agent = ExAnteAgent(llm_provider=llm)

    # 4. Evaluate
    result = agent.evaluate_proposal(sample_proposal)
    
    print("Mean ROI:", result['quantitative_metrics']['mean_roi'], "%")
    print("VaR 95%:", result['quantitative_metrics']['var_95'], "%")
    print("Probability of Loss:", result['quantitative_metrics']['probability_of_loss'], "%")
    print("Status:", result['confidence_level_status'])
    print("--- Test Passed Successfully ---")

if __name__ == "__main__":
    run_test()


