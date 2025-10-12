"""LLM Handler"""
import sys
from pathlib import Path

# เพิ่ม root folder เข้า path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from langchain_community.llms import Ollama

try:
    from config import LLM_MODEL, LLM_TEMPERATURE, SYSTEM_PROMPT
except ImportError:
    LLM_MODEL = "llama3.1:8b"
    LLM_TEMPERATURE = 0.7
    SYSTEM_PROMPT = """คุณเป็น AI ผู้ช่วยที่ฉลาดและเป็นมิตร
    
กฎสำคัญ:
1. ตอบเป็นภาษาไทยเสมอ
2. ใช้ "ครับ/ค่ะ" ให้เหมาะสม
3. ตอบสั้น กระชับ ชัดเจน
4. ถ้าไม่รู้ บอกตรงๆ"""


class LLMHandler:
    def __init__(self):
        print(f"🤖 Initializing LLM: {LLM_MODEL}")
        self.llm = Ollama(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            system=SYSTEM_PROMPT
        )
    
    def generate(self, prompt: str) -> str:
        """Generate response from LLM"""
        try:
            # เปลี่ยนจาก self.llm(prompt) เป็น invoke
            return self.llm.invoke(prompt)  # ← เปลี่ยนตรงนี้!
        except Exception as e:
            return f"เกิดข้อผิดพลาด: {str(e)}"
    
    def get_model_name(self) -> str:
        return LLM_MODEL


if __name__ == "__main__":
    print("Testing LLM Handler...")
    llm = LLMHandler()
    print(f"✅ Model: {llm.get_model_name()}")
    
    # ทดสอบภาษาไทย
    print("\n📝 ทดสอบ 1:")
    response = llm.generate("1+1=?")
    print(f"Q: 1+1=?")
    print(f"A: {response}")
    
    print("\n📝 ทดสอบ 2:")
    response2 = llm.generate("AI คืออะไร")
    print(f"Q: AI คืออะไร")
    print(f"A: {response2}")