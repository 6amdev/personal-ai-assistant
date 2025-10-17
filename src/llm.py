"""LLM Handler - With Device Support"""
import sys
import os
from pathlib import Path

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
    def __init__(self, model_name: str = None, device: str = "GPU"):
        """
        Initialize LLM Handler
        
        Args:
            model_name: ชื่อ model (ถ้าไม่ระบุจะใช้จาก config)
            device: "GPU" หรือ "CPU"
        """
        self.model_name = model_name or LLM_MODEL
        self.device = device.upper()
        
        print(f"🤖 Initializing LLM: {self.model_name}")
        print(f"   Device: {self.device}")
        
        # 🔥 Ollama ใช้ GPU โดย default
        # ถ้าต้องการบังคับใช้ CPU ให้ตั้ง env var
        if self.device == "CPU":
            os.environ['OLLAMA_NUM_GPU'] = '0'
            print("   ⚠️ LLM will use CPU only")
        else:
            # ใช้ GPU (ลบ env var ถ้ามี)
            if 'OLLAMA_NUM_GPU' in os.environ:
                del os.environ['OLLAMA_NUM_GPU']
            print("   ✅ LLM will use GPU")
        
        try:
            self.llm = Ollama(
                model=self.model_name,
                temperature=LLM_TEMPERATURE,
                system=SYSTEM_PROMPT
            )
            print(f"✅ LLM ready: {self.model_name} on {self.device}")
        except Exception as e:
            print(f"❌ Error initializing LLM: {e}")
            raise
    
    def generate(self, prompt: str) -> str:
        """Generate response from LLM"""
        try:
            return self.llm.invoke(prompt)
        except Exception as e:
            return f"เกิดข้อผิดพลาด: {str(e)}"
    
    def get_model_name(self) -> str:
        """Get current model name"""
        return self.model_name


if __name__ == "__main__":
    print("Testing LLM Handler...")
    
    # Test GPU
    print("\n=== Test GPU ===")
    llm_gpu = LLMHandler(device="GPU")
    print(f"✅ Model: {llm_gpu.get_model_name()}")
    response = llm_gpu.generate("1+1=?")
    print(f"Response: {response[:100]}")
    
    # Test CPU
    print("\n=== Test CPU ===")
    llm_cpu = LLMHandler(device="CPU")
    print(f"✅ Model: {llm_cpu.get_model_name()}")
    response2 = llm_cpu.generate("สวัสดี")
    print(f"Response: {response2[:100]}")