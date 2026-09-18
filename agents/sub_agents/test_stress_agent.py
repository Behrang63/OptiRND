import json
import logging
import traceback
import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

from core.llm_provider import BaseLLMProvider, LLMFactory
from agents.main_agents.ex_ante_agent import ExAnteAgent


class StressTestSubAgent:
    """
    عامل فرعی شبیه‌سازی خودکار سناریوهای تستی و ارزیابی پایداری ماشین (Stress Tester)
    همراه با سیستم ثبت لاگ ساختاریافته خطاها جهت تسهیل فرآیند دیباگ.
    """
    def __init__(self, llm_provider: BaseLLMProvider = None, log_dir: str = "logs"):
        self.llm = llm_provider or LLMFactory.get_provider("mock")
        self.agent = ExAnteAgent(llm_provider=self.llm)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.error_log_file = self.log_dir / "test_agent_errors.log"
        self._setup_logger()

    def _setup_logger(self):
        """تنظیم سیستم لاگینگ برای ثبت جزئیات دقیق خطاها"""
        self.logger = logging.getLogger("StressTestSubAgent")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.FileHandler(self.error_log_file, encoding="utf-8")
            formatter = logging.Formatter('{"time": "%(asctime)s", "level": "%(levelname)s", "message": %(message)s}')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def generate_synthetic_scenarios(self) -> List[Dict[str, Any]]:
        """
        تولید ۴ سناریوی تستی متنوع برای به چالش کشیدن الگوریتم‌های محاسباتی
        """
        scenarios = [
            # ۱. سناریوی پایه و پایدار
            {
                "scenario_name": "پروژه نرمال و کم‌ریسک",
                "title": "سامانه اتوماسیون انبارداری قطعات",
                "cost_p10": 2000.0, "cost_p50": 3000.0, "cost_p90": 4000.0,
                "benefit_p10": 4500.0, "benefit_p50": 6000.0, "benefit_p90": 8000.0,
                "p_success": 0.90, "years": 3,
                "enable_energy_risk": True, "power_outage_triplet": (5.0, 10.0, 15.0),
                "gas_outage_triplet": (5.0, 10.0, 15.0), "energy_daily_loss": 20.0,
                "enable_reliability_risk": True, "mtbf_triplet": (800.0, 1200.0, 1600.0),
                "mttr_triplet": (1.0, 2.0, 4.0), "hourly_downtime_loss": 10.0,
                "enable_supply_chain_risk": True, "planned_lead_time": 60.0,
                "actual_lead_time_triplet": (50.0, 65.0, 90.0), "daily_delay_cost": 5.0,
                "enable_iran_tax": True, "corporate_tax_rate": 0.20
            },
            # ۲. سناریوی بحران شدید عملیاتی و توقفات
            {
                "scenario_name": "بحران ناترازی و خرابی حاد تجهیز",
                "title": "بهینه‌سازی مشعل‌های کوره بلند در شرایط بحران گاز",
                "cost_p10": 5000.0, "cost_p50": 7000.0, "cost_p90": 11000.0,
                "benefit_p10": 3000.0, "benefit_p50": 4500.0, "benefit_p90": 6000.0,
                "p_success": 0.60, "years": 3,
                "enable_energy_risk": True, "power_outage_triplet": (30.0, 50.0, 75.0),
                "gas_outage_triplet": (40.0, 60.0, 90.0), "energy_daily_loss": 80.0,
                "enable_reliability_risk": True, "mtbf_triplet": (200.0, 400.0, 600.0),
                "mttr_triplet": (8.0, 16.0, 32.0), "hourly_downtime_loss": 50.0,
                "enable_supply_chain_risk": True, "planned_lead_time": 90.0,
                "actual_lead_time_triplet": (100.0, 150.0, 220.0), "daily_delay_cost": 25.0,
                "enable_iran_tax": False
            },
            # ۳. سناریوی جهش اقتصادی و ارزی
            {
                "scenario_name": "شوک تورمی و صرفه‌جویی ارزی کلان",
                "title": "بومی‌سازی پمپ‌های اسلاری با فناوری داخلی",
                "cost_p10": 4000.0, "cost_p50": 6000.0, "cost_p90": 9000.0,
                "benefit_p10": 8000.0, "benefit_p50": 12000.0, "benefit_p90": 18000.0,
                "p_success": 0.80, "years": 5,
                "inflation_triplet": (0.50, 0.75, 1.10),
                "currency_type": "EUR", "base_fx_rate": 72000.0, "annual_fx_savings": 80000.0,
                "enable_energy_risk": False, "enable_reliability_risk": True,
                "mtbf_triplet": (600.0, 900.0, 1200.0), "mttr_triplet": (3.0, 5.0, 8.0),
                "hourly_downtime_loss": 20.0, "enable_iran_tax": True,
                "enable_cbam_tax": True, "export_tons_triplet": (100000.0, 150000.0, 200000.0),
                "co2_reduction_kg": 150.0, "carbon_tax_usd_triplet": (70.0, 90.0, 130.0)
            },
            # ۴. سناریوی داده‌های کرانه‌ای و مرزی
            {
                "scenario_name": "داده‌های مرزی و احتمال موفقیت اندک",
                "title": "پروژه آزمایشی بازیابی حرارت تلف‌شده",
                "cost_p10": 10000.0, "cost_p50": 15000.0, "cost_p90": 25000.0,
                "benefit_p10": 500.0, "benefit_p50": 1000.0, "benefit_p90": 1500.0,
                "p_success": 0.15, "years": 2,
                "enable_energy_risk": True, "power_outage_triplet": (0.0, 0.0, 0.0),
                "gas_outage_triplet": (0.0, 0.0, 0.0), "energy_daily_loss": 0.0,
                "enable_reliability_risk": False, "enable_supply_chain_risk": False,
                "enable_iran_tax": True, "approved_pct_triplet": (0.0, 0.1, 0.2)
            }
        ]
        return scenarios

    def execute_and_evaluate_all(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        اجرای آزمایشی تمام سناریوها روی ماشین و ثبت خطاها در لاگ
        """
        scenarios = self.generate_synthetic_scenarios()
        successful_evaluations = []
        error_records = []

        for item in scenarios:
            name = item.get("scenario_name", "Unknown")
            try:
                result = self.agent.evaluate_comprehensive_proposal(item)
                successful_evaluations.append({
                    "scenario_name": name,
                    "title": item.get("title"),
                    "status": "SUCCESS",
                    "adjusted_benefit": result.get("adjusted_net_benefit"),
                    "total_downtime_hours": result.get("total_downtime_hours"),
                    "mean_roi": result.get("mean_roi"),
                    "risk_status": result.get("risk_status")
                })
            except Exception as e:
                error_payload = {
                    "scenario_name": name,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": traceback.format_exc(),
                    "input_data": item
                }
                error_records.append(error_payload)
                self.logger.error(json.dumps(error_payload, ensure_ascii=False))

        return successful_evaluations, error_records