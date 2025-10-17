"""Naive RAG - แบบพื้นฐาน"""
from .base_rag import BaseRAG
from typing import List, Dict

class NaiveRAG(BaseRAG):
    """RAG แบบพื้นฐาน"""
    
    def __init__(self, llm_handler, memory_handler):
        super().__init__(llm_handler, memory_handler, name="Naive RAG")
    
    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """ค้นหาด้วย vector similarity"""
        return self.memory.search(query, k=k)
    
    def generate(self, query: str, context: str) -> str:
        """Generate คำตอบ"""
        if not context.strip():
            prompt = query
        else:
            prompt = f"""{context}

คำถาม: {query}

วิธีตอบ:
1. ตอบคำถามโดยอิงจากข้อมูลที่เกี่ยวข้องข้างต้น
2. ถ้ามีรูปภาพที่เกี่ยวข้อง จะแสดงให้ดูด้านล่าง
3. ระบุแหล่งที่มาของข้อมูล
4. ตอบเป็นภาษาไทยที่เข้าใจง่าย

คำตอบ:"""
        
        return self.llm.generate(prompt)