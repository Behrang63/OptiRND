import streamlit as st
from typing import Any, Dict, Optional
from core.contracts import EvaluationResponse, ErrorResponse, ProposalRequest


class StateManager:
    """
    مدیریت متمرکز وضعیت نشست (Session State) برای انتقال داده بین صفحات مختلف.
    ذخیره‌سازی پاسخ‌های API و خطاهای شبکه بدون فراخوانی منطق دامنه.
    """
    @staticmethod
    def initialize_state():
        """مقداردهی اولیه متغیرهای مورد نیاز در کل سامانه"""
        defaults: Dict[str, Any] = {
            "current_proposal": None,
            "last_simulation_result": None,
            "selected_currency": "USD",
            "last_evaluation_response": None,
            "last_proposal_request": None,
            "last_api_error": None,
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        return st.session_state.get(key, default)

    @staticmethod
    def set(key: str, value: Any):
        st.session_state[key] = value

    @staticmethod
    def set_evaluation_response(response: EvaluationResponse):
        """Store API evaluation response payload."""
        st.session_state["last_evaluation_response"] = response

    @staticmethod
    def get_evaluation_response() -> Optional[EvaluationResponse]:
        """Retrieve stored API evaluation response."""
        return st.session_state.get("last_evaluation_response")

    @staticmethod
    def set_proposal_request(request: ProposalRequest):
        """Store the proposal request sent to API."""
        st.session_state["last_proposal_request"] = request

    @staticmethod
    def get_proposal_request() -> Optional[ProposalRequest]:
        """Retrieve stored proposal request."""
        return st.session_state.get("last_proposal_request")

    @staticmethod
    def set_api_error(error: ErrorResponse):
        """Store API network/validation error."""
        st.session_state["last_api_error"] = error

    @staticmethod
    def get_api_error() -> Optional[ErrorResponse]:
        """Retrieve stored API error."""
        return st.session_state.get("last_api_error")

    @staticmethod
    def clear_evaluation_state():
        """Clear all evaluation-related state."""
        for key in ["last_evaluation_response", "last_proposal_request", "last_api_error"]:
            if key in st.session_state:
                del st.session_state[key]