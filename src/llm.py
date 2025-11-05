"""
LLM Handler - รองรับหลาย Providers
Supports: Ollama, OpenAI
"""
import sys
import os
from pathlib import Path
from typing import Optional, Literal

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI

try:
    from config import (
        LLM_MODEL, 
        LLM_TEMPERATURE, 
        SYSTEM_PROMPT,
        OPENAI_API_KEY
    )
except ImportError:
    LLM_MODEL = "llama3.1:8b"
    LLM_TEMPERATURE = 0.7
    OPENAI_API_KEY = ""
    SYSTEM_PROMPT = """คุณเป็น AI ผู้ช่วยที่ฉลาดและเป็นมิตร
    
กฎสำคัญ:
1. ตอบเป็นภาษาไทยเสมอ
2. ใช้ "ครับ/ค่ะ" ให้เหมาะสม
3. ตอบสั้น กระชับ ชัดเจน
4. ถ้าไม่รู้ บอกตรงๆ"""


class LLMHandler:
    """
    LLM Handler รองรับหลาย providers
    
    Providers:
    - ollama: Local LLM (default)
    - openai: OpenAI API (GPT-3.5, GPT-4, etc.)
    """
    
    def __init__(
        self, 
        model_name: Optional[str] = None,
        provider: Literal["ollama", "openai"] = "ollama",
        device: str = "GPU",
        api_key: Optional[str] = None
    ):
        """
        Initialize LLM Handler
        
        Args:
            model_name: ชื่อ model 
                - Ollama: "llama3.1:8b", "mistral:7b", etc.
                - OpenAI: "gpt-3.5-turbo", "gpt-4", "gpt-4o", etc.
            provider: "ollama" หรือ "openai"
            device: "GPU" หรือ "CPU" (สำหรับ Ollama only)
            api_key: OpenAI API key (ถ้าใช้ OpenAI)
        """
        self.provider = provider.lower()
        self.device = device.upper()
        
        # Auto-detect provider from model name if not specified
        if model_name:
            if model_name.startswith("gpt"):
                self.provider = "openai"
            elif ":" in model_name or model_name in ["llama", "mistral", "gemma"]:
                self.provider = "ollama"
        
        # Set model name
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = "gpt-3.5-turbo" if self.provider == "openai" else LLM_MODEL
        
        # Set API key for OpenAI
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)
        
        print(f"🤖 Initializing LLM Handler")
        print(f"   Provider: {self.provider.upper()}")
        print(f"   Model: {self.model_name}")
        
        if self.provider == "ollama":
            self._init_ollama()
        elif self.provider == "openai":
            self._init_openai()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _init_ollama(self):
        """Initialize Ollama LLM"""
        print(f"   Device: {self.device}")
        
        # Configure device
        if self.device == "CPU":
            os.environ['OLLAMA_NUM_GPU'] = '0'
            print("   ⚠️  LLM will use CPU only")
        else:
            if 'OLLAMA_NUM_GPU' in os.environ:
                del os.environ['OLLAMA_NUM_GPU']
            print("   ✅ LLM will use GPU")
        
        try:
            self.llm = Ollama(
                model=self.model_name,
                temperature=LLM_TEMPERATURE,
                system=SYSTEM_PROMPT
            )
            print(f"✅ Ollama LLM ready: {self.model_name} on {self.device}")
        except Exception as e:
            print(f"❌ Error initializing Ollama: {e}")
            raise
    
    def _init_openai(self):
        """Initialize OpenAI LLM"""
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required! "
                "Set OPENAI_API_KEY in .env or pass api_key parameter"
            )
        
        print(f"   API Key: {self.api_key[:8]}...")
        
        try:
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=LLM_TEMPERATURE,
                api_key=self.api_key,
                # Add system message
                model_kwargs={"messages": [{"role": "system", "content": SYSTEM_PROMPT}]}
            )
            print(f"✅ OpenAI LLM ready: {self.model_name}")
        except Exception as e:
            print(f"❌ Error initializing OpenAI: {e}")
            print(f"   Make sure API key is valid and model '{self.model_name}' is accessible")
            raise
    
    def generate(self, prompt: str) -> str:
        """
        Generate response from LLM
        
        Args:
            prompt: User prompt/query
            
        Returns:
            Generated text response
        """
        try:
            if self.provider == "openai":
                # OpenAI returns AIMessage object
                response = self.llm.invoke(prompt)
                return response.content if hasattr(response, 'content') else str(response)
            else:
                # Ollama returns string directly
                return self.llm.invoke(prompt)
        except Exception as e:
            error_msg = f"เกิดข้อผิดพลาด: {str(e)}"
            print(f"❌ Generation error: {e}")
            return error_msg
    
    def get_model_name(self) -> str:
        """Get current model name"""
        return self.model_name
    
    def get_provider(self) -> str:
        """Get current provider"""
        return self.provider
    
    def get_info(self) -> dict:
        """Get LLM information"""
        info = {
            "provider": self.provider,
            "model": self.model_name,
            "temperature": LLM_TEMPERATURE
        }
        
        if self.provider == "ollama":
            info["device"] = self.device
        elif self.provider == "openai":
            info["api_key_set"] = bool(self.api_key)
        
        return info


# ===== Convenience Functions =====

def create_ollama_llm(model_name: str = None, device: str = "GPU") -> LLMHandler:
    """Create Ollama LLM instance"""
    return LLMHandler(
        model_name=model_name,
        provider="ollama",
        device=device
    )


def create_openai_llm(
    model_name: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None
) -> LLMHandler:
    """Create OpenAI LLM instance"""
    return LLMHandler(
        model_name=model_name,
        provider="openai",
        api_key=api_key
    )


# ===== Testing =====

if __name__ == "__main__":
    print("=" * 60)
    print("Testing LLM Handler with Multiple Providers")
    print("=" * 60)
    
    # Test 1: Ollama with GPU
    print("\n" + "=" * 60)
    print("Test 1: Ollama LLM (GPU)")
    print("=" * 60)
    try:
        llm_ollama = LLMHandler(
            model_name="llama3.1:8b",
            provider="ollama",
            device="GPU"
        )
        print(f"\n✅ Info: {llm_ollama.get_info()}")
        
        response = llm_ollama.generate("1+1=?")
        print(f"💬 Response: {response[:150]}")
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    # Test 2: Ollama with CPU
    print("\n" + "=" * 60)
    print("Test 2: Ollama LLM (CPU)")
    print("=" * 60)
    try:
        llm_cpu = LLMHandler(
            model_name="llama3.2:3b",
            provider="ollama",
            device="CPU"
        )
        print(f"\n✅ Info: {llm_cpu.get_info()}")
        
        response = llm_cpu.generate("สวัสดีครับ")
        print(f"💬 Response: {response[:150]}")
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    # Test 3: OpenAI (if API key available)
    print("\n" + "=" * 60)
    print("Test 3: OpenAI LLM")
    print("=" * 60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            llm_openai = LLMHandler(
                model_name="gpt-3.5-turbo",
                provider="openai",
                api_key=api_key
            )
            print(f"\n✅ Info: {llm_openai.get_info()}")
            
            response = llm_openai.generate("สวัสดีครับ")
            print(f"💬 Response: {response[:150]}")
        except Exception as e:
            print(f"❌ Test failed: {e}")
    else:
        print("⚠️  Skipped: OPENAI_API_KEY not set")
    
    # Test 4: Auto-detect provider
    print("\n" + "=" * 60)
    print("Test 4: Auto-detect Provider")
    print("=" * 60)
    
    # Should detect as ollama
    llm_auto1 = LLMHandler(model_name="mistral:7b")
    print(f"✅ Model: {llm_auto1.model_name}")
    print(f"   Provider: {llm_auto1.provider} (auto-detected)")
    
    # Should detect as openai
    if api_key:
        llm_auto2 = LLMHandler(model_name="gpt-4")
        print(f"✅ Model: {llm_auto2.model_name}")
        print(f"   Provider: {llm_auto2.provider} (auto-detected)")
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)
