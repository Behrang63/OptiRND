import unittest

class CodeAssistant:
    """
    کلاس شبیه‌ساز دستیار کدنویسی بر اساس اهداف و وظایف تعیین‌شده
    """
    def __init__(self):
        self.goals = []
        self.conversation_context = []

    def set_goal(self, goal_description: str):
        """ثبت هدف جدید کاربر"""
        self.goals.append(goal_description)
        return True

    def generate_code(self, task_name: str) -> str:
        """ساخت کد بر اساس تسک تعریف‌شده"""
        return f"# Code generated for: {task_name}\nprint('Executing {task_name}')"

    def get_learning_steps(self, task_name: str) -> list:
        """ارائه مراحل آموزشی برای درک و پیاده‌سازی ساده"""
        return [
            f"مرحله ۱: درک ساختار ورودی برای {task_name}",
            f"مرحله ۲: نوشتن منطق اصلی برنامه",
            f"مرحله ۳: اجرای تست و اعتبارسنجی خروجی"
        ]

    def document_step(self, step_name: str, details: str) -> dict:
        """مستندسازی دقیق هر بخش از کد"""
        doc_entry = {
            "step": step_name,
            "details": details,
            "status": "Documented"
        }
        self.conversation_context.append(doc_entry)
        return doc_entry


class TestCodeAssistantActions(unittest.TestCase):
    """
    تست‌های خودکار برای بررسی عملکرد اهداف و اقدامات دستیار
    """
    def setUp(self):
        self.assistant = CodeAssistant()

    def test_goal_registration(self):
        """تست هدف ۱: ثبت صحیح اهداف کاربر"""
        result = self.assistant.set_goal("نوشتن اسکریپت پردازش داده")
        self.assertTrue(result)
        self.assertIn("نوشتن اسکریپت پردازش داده", self.assistant.goals)

    def test_code_generation(self):
        """تست هدف ۲: تولید کد کامل و معتبر"""
        code = self.assistant.generate_code("data_cleaning")
        self.assertIn("print('Executing data_cleaning')", code)

    def test_educational_steps(self):
        """تست هدف ۳: ارائه مراحل آموزشی به زبان ساده"""
        steps = self.assistant.get_learning_steps("تحلیل ریسک")
        self.assertEqual(len(steps), 3)
        self.assertTrue(steps[0].startswith("مرحله ۱"))

    def test_documentation(self):
        """تست هدف ۴: مستندسازی شفاف و کامل"""
        doc = self.assistant.document_step("ماژول بهینه‌سازی", "توضیح متغیرها و نحوه فراخوانی تابع")
        self.assertEqual(doc["status"], "Documented")
        self.assertEqual(len(self.assistant.conversation_context), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)