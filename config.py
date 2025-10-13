"""Configuration - ตั้งค่าระบบ"""
import os
from dotenv import load_dotenv

# โหลด .env
load_dotenv()

# ===== App Settings =====
APP_TITLE = "💬 Personal AI Assistant"
APP_DESCRIPTION = """
🤖 AI Chatbot ส่วนตัวที่เรียนรู้จากเอกสารของคุณ  
✨ ใช้ Llama 3.1 + RAG • ฟรี 100% • รันบนเครื่องคุณเอง
"""

# ===== LLM Settings =====
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

# ===== Embeddings Settings ===== 🔥
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence-transformers")
# Options: 
# - "sentence-transformers" (เร็ว, ฟรี, Local)
# - "ollama" (ช้า, ฟรี, Local)
# - "openai" (เร็ว, เสียตัง, Cloud)

# Sentence Transformers (Local - แนะนำ!)
SENTENCE_TRANSFORMER_MODEL = os.getenv(
    "SENTENCE_TRANSFORMER_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
# Models ที่แนะนำ:
# - paraphrase-multilingual-MiniLM-L12-v2 (เร็ว, รองรับไทย) ⭐
# - intfloat/multilingual-e5-large (ช้ากว่า แต่แม่นกว่า)
# - BAAI/bge-m3 (ดีที่สุดสำหรับหลายภาษา)

# Ollama (Local - ถ้าไม่มี GPU)
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "llama3.1:8b")

# OpenAI (Cloud - ทดสอบทีหลัง)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Device
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cuda")  # "cuda" or "cpu"

# ===== Database Settings =====
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./data/chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "personal_assistant")

# ===== Document Processing =====
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# ===== System Prompt =====
SYSTEM_PROMPT = """คุณเป็น AI ผู้ช่วยส่วนตัวที่ชาญฉลาดและเป็นมิตร

ความสามารถของคุณ:
- ตอบคำถามด้วยภาษาไทยที่เข้าใจง่าย
- ใช้ข้อมูลจากเอกสารที่ผู้ใช้อัพโหลด
- ให้คำแนะนำที่เป็นประโยชน์
- อธิบายอย่างชัดเจนและกระชับ

หลักการตอบ:
1. ตอบตรงประเด็น
2. ใช้ข้อมูลจากเอกสาร (ถ้ามี)
3. ระบุแหล่งที่มา
4. ถ้าไม่แน่ใจ บอกตรงๆ
5. ใช้ภาษาที่เป็นมิตร

จำไว้: คุณคือผู้ช่วยที่น่าเชื่อถือ!
"""