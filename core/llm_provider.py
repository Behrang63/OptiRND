from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type

from pydantic import BaseModel, ValidationError


class SchemaValidationError(Exception):
    """
    Raised when an LLM payload cannot be validated against its Pydantic
    schema after all retries. Callers should treat this as a signal to use
    the deterministic mock path - never to crash the UI.
    """

    def __init__(self, message: str, last_error: Optional[ValidationError] = None):
        super().__init__(message)
        self.last_error = last_error


class BaseLLMProvider(ABC):
    """
    اینترفیس انتزاعی پایه برای تمامی ارائه‌دهندگان هوش مصنوعی در پلتفرم پژوهشیار.
    """
    @abstractmethod
    def generate_json(self, prompt: str, schema_description: str) -> Dict[str, Any]:
        """تولید پاسخ ساختاریافته به فرمت JSON"""
        pass

    @abstractmethod
    def generate_text(self, prompt: str) -> str:
        """تولید پاسخ متنی ساده"""
        pass

    @abstractmethod
    def interpret_monte_carlo(self, metrics: Dict[str, Any], p_success: float) -> str:
        """تفسیر مدیریتی و هوشمند نتایج شبیه‌سازی مونت‌کارلو"""
        pass

    def generate_validated(
        self,
        prompt: str,
        schema_description: str,
        schema_model: Type[BaseModel],
    ) -> Dict[str, Any]:
        """
        Schema-bound generation contract: generate JSON, validate it against
        ``schema_model`` (Pydantic v2), return the validated dict.

        Default implementation: delegate to generate_json() and validate once.
        Providers that support retries natively (OllamaLLMProvider) override
        this behavior; the base class raises SchemaValidationError on violation
        so callers can route to the mock path deterministically.
        """
        payload = self.generate_json(prompt, schema_description)
        try:
            return schema_model.model_validate(payload).model_dump()
        except ValidationError as exc:
            raise SchemaValidationError(
                f"LLM payload failed {schema_model.__name__} validation: "
                f"{exc.error_count()} error(s)",
                last_error=exc,
            ) from exc


class MockLLMProvider(BaseLLMProvider):
    """
    ارائه‌دهنده آفلاین و مبتنی بر قوانین ایمن (Fallback).
    تضمین‌کننده پایداری سیستم بدون وابستگی به سرویس‌های ابری یا محلی طبق ضوابط SKILL.md.
    """
    
    def generate_json(self, prompt: str, schema_description: str) -> Dict[str, Any]:
        prompt_lower = prompt.lower()
        
        if "ex_ante" in prompt_lower or "ارزیابی" in prompt_lower:
            return {
                "technical_success_probability": 0.85,
                "recommended_action": "APPROVE_WITH_CONDITIONS",
                "risk_summary": "پیچیدگی فنی الگوریتم متناسب است؛ ریسک مالی در حد قابل مدیریت ارزیابی می‌شود."
            }
        elif "process" in prompt_lower or "پایش" in prompt_lower:
            return {
                "variance_status": "WARNING",
                "revised_completion_probability": 0.82,
                "mitigation_advice": "تخصیص مجدد منابع محاسباتی جهت جلوگیری از تاخیر زمانی."
            }
        
        return {
            "technical_success_probability": 0.85,
            "recommended_action": "APPROVE_WITH_CONDITIONS",
            "risk_summary": "پیچیدگی فنی الگوریتم متناسب است."
        }

    def generate_text(self, prompt: str) -> str:
        return "[پاسخ آفلاین]: تحلیل سیستم با موفقیت اجرا شد."

    def interpret_monte_carlo(self, metrics: Dict[str, Any], p_success: float) -> str:
        p_fail_pct = round((1 - p_success) * 100, 1)
        mean_roi = metrics.get("mean_roi", 0.0)
        var_95 = metrics.get("var_95", -100.0)
        
        return (
            f"📖 **تفسیر هوشمند مدیریتی (تولیدشده توسط LLM):**\n\n"
            f"* **ریسک شکست فنی:** این پروژه دارای **{p_fail_pct}% احتمال عدم موفقیت کامل** است که منجر به زیان کل سرمایه می‌شود.\n"
            f"* **بازدهی در صورت موفقیت:** در صورت عبور از چالش‌های فنی، میانگین نرخ بازگشت سرمایه برابر با **{mean_roi}%** خواهد بود.\n"
            f"* **ارزیابی VaR 95% ({var_95}%):** به دلیل سهم {p_fail_pct} درصدی احتمال عدم موفقیت، شاخص ارزش در معرض ریسک ۵ درصد بدترین سناریوها روی حداکثر زیان تنظیم شده است.\n"
            f"* **توصیه استراتژیک:** تصویب مشروط به تعریف مایلستون‌های ارزیابی ریسک فنی در فاز اول."
        )


class LLMFactory:
    """
    کارخانه مدیریت و نمونه‌سازی ارائه‌دهندگان هوش مصنوعی.

    Phase 3 routing:
      * "mock"   -> MockLLMProvider (deterministic, offline - the default).
      * "ollama" -> OllamaLLMProvider (local server); constructed lazily so
        importing this module never requires Ollama to be installed/running.
      * anything else -> MockLLMProvider (fail-safe default).

    The mock fallback contract (SKILL.md) is preserved: an unavailable or
    misconfigured real provider NEVER breaks the application - Ollama's own
    methods degrade to mock behavior, and factory errors are logged and
    swallowed into a mock instance.
    """

    @staticmethod
    def get_provider(provider_type: str = "mock", api_key: Optional[str] = None) -> BaseLLMProvider:
        if provider_type == "ollama":
            try:
                # Local import: keeps Ollama optional at import time.
                from agents.ollama_provider import OllamaLLMProvider
                return OllamaLLMProvider()
            except Exception as exc:  # noqa: BLE001 - no-crash guarantee
                import logging
                logging.getLogger("optirnd.ollama").warning(
                    "Ollama provider construction failed; using mock",
                    extra={"error_class": type(exc).__name__},
                )
                return MockLLMProvider()

        # "mock", unknown values, and legacy api_key-only calls all resolve
        # to the deterministic offline provider.
        return MockLLMProvider()