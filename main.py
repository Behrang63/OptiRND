import os
import subprocess
from pathlib import Path
from core.database import init_db

def setup_environment():
    """
    آماده‌سازی پیش‌نیازهای برنامه، ساخت پوشه‌های داده و راه‌اندازی اولیه دیتابیس.
    """
    # ساخت پوشه data برای ذخیره فایل‌ها و دیتابیس در صورت عدم وجود
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # راه‌اندازی اولیه دیتابیس
    init_db()
    print("✅ [محیط اجرا و پایگاه داده با موفقیت آماده‌سازی شد]")

def run_app():
    """
    اجرای وب‌اپلیکیشن Streamlit.
    """
    setup_environment()
    print("🚀 در حال راه‌اندازی رابط کاربری سامانه پژوهشیار...")
    subprocess.run(["streamlit", "run", "app.py"])

if __name__ == "__main__":
    run_app()