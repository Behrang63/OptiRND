import sys
import tempfile
import shutil
from pathlib import Path

# افزودن مسیر ریشه پروژه به پایتون
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from core.rag_engine import create_rag_engine

def test_rag_retrieval():
    """
    تست نمایه‌سازی و بازیابی متون در موتور RAG محلی
    """
    # Use a temp directory to avoid dimension mismatch with existing persisted embeddings
    temp_dir = tempfile.mkdtemp()
    try:
        rag = create_rag_engine(provider_type="mock", store_dir=temp_dir, chunk_size=20)
        
        sample_text = (
            "پروژه سامانه تشخیص عیوب ورق فولاد مبارکه شامل دو فاز اصلی است. "
            "فاز اول مربوط به طراحی الگوریتم پردازش تصویر و فاز دوم مربوط به پیاده‌سازی سخت‌افزار است. "
            "ریسک اصلی پروژه تامین دوربین‌های صنعتی با سرعت بالا می‌باشد."
        )
        
        # نمایه‌سازی سند نمونه
        rag.index_document(doc_id="prop_01", text=sample_text)
        
        # جستجوی کلیدواژه ریسک
        results = rag.retrieve_relevant_chunks(query="ریسک دوربین سخت‌افزار", top_k=1)
        
        assert len(results) > 0
        print("\n✅ [تست موتور RAG محلی با موفقیت انجام شد]")
        print(f"📄 تکه متن استخراج‌شده: {results[0]['text']}")
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    test_rag_retrieval()