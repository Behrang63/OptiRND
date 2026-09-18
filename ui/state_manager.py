import streamlit as st
from typing import Any, Dict

class StateManager:
    """
    مدیریت متمرکز وضعیت نشست (Session State) برای انتقال داده بین صفحات مختلف.
    """
    @staticmethod
    def initialize_state():
        """مقداردهی اولیه متغیرهای مورد نیاز در کل سامانه"""
        defaults: Dict[str, Any] = {
            "current_proposal": None,
            "last_simulation_result": None,
            "selected_currency": "USD"
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