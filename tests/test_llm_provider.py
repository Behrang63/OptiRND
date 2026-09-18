import sys
from pathlib import Path

# افزودن مسیر ریشه پروژه به پایتون
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from core.llm_provider import LLMFactory

def test_mock_llm_provider():
    """تست عملکرد خروجی آفلاین هوش مصنوعی"""
    llm = LLMFactory.get_provider("mock")
    
    # تست خروجی ساختاریافته JSON
    json_response = llm.generate_json("ارزیابی ex_ante پروپوزال فولاد", "schema")
    assert "technical_success_probability" in json_response
    
    # تست خروجی متنی
    text_response = llm.generate_text("تحلیل ریسک")
    assert "[پاسخ آفلاین]" in text_response
    
    print("\n✅ [تست ماژول هوش مصنوعی آفلاین با موفقیت انجام شد]")
    print(f"📄 خروجی نمونه: {json_response['risk_summary']}")

if __name__ == "__main__":
    test_mock_llm_provider()