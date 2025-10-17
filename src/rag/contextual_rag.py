"""Contextual RAG - เน้น context รอบๆ chunk"""
from .base_rag import BaseRAG
from typing import List, Dict

class ContextualRAG(BaseRAG):
    """
    Contextual RAG - เน้นให้ LLM ใช้ context รอบๆ
    
    ความแตกต่างจาก Naive RAG:
    - Prompt ที่ดีกว่า เน้นการใช้ context
    - ผลลัพธ์แม่นกว่าเพราะ LLM เข้าใจบริบทดีขึ้น
    """
    
    def __init__(self, llm_handler, memory_handler):
        super().__init__(llm_handler, memory_handler, name="Contextual RAG")
    
    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """ค้นหาปกติ (เหมือน Naive RAG)"""
        return self.memory.search(query, k=k)
    
    def generate(self, query: str, context: str) -> str:
        """Generate พร้อมเน้น context"""
        if not context.strip():
            prompt = query
        else:
            # Prompt ใหม่ - เน้น context มากกว่า
            prompt = f"""ข้อมูลต่อไปนี้มี context ก่อนหน้าและหลังรวมอยู่ด้วย:

{context}

คำถาม: {query}

วิธีตอบ:
1. อ่าน context รอบๆ ข้อความเพื่อเข้าใจบริบทให้ดี
2. ตอบคำถามโดยอิงจากข้อมูลที่เกี่ยวข้องทั้งหมด
3. ใช้ข้อมูลจากหลายส่วนประกอบกันถ้าจำเป็น
4. ระบุแหล่งที่มาของข้อมูล
5. ตอบเป็นภาษาไทยที่เข้าใจง่าย

คำตอบ:"""
        
        return self.llm.generate(prompt)