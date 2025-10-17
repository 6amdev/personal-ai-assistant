"""Rerank RAG - จัดอันดับใหม่ด้วย keyword matching"""
from .base_rag import BaseRAG
from typing import List, Dict

class RerankRAG(BaseRAG):
    """
    Rerank RAG - ค้นหาเยอะ แล้วจัดอันดับใหม่
    
    วิธีทำงาน:
    1. Retrieve เยอะหน่อย (k*2)
    2. Re-rank ด้วย keyword overlap
    3. เลือก top k
    
    ดีกว่า Naive RAG ตรงที่:
    - กรองเอกสารที่ไม่เกี่ยวข้องออก
    - ให้น้ำหนักกับคำสำคัญ
    """
    
    def __init__(self, llm_handler, memory_handler):
        super().__init__(llm_handler, memory_handler, name="Rerank RAG")
    
    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """ค้นหา + re-rank"""
        # 1. ค้นหาเยอะหน่อย
        results = self.memory.search(query, k=k*2)
        
        # 2. Re-rank ด้วย keyword matching
        query_words = set(query.lower().split())
        
        for result in results:
            content = result['content'].lower()
            content_words = set(content.split())
            
            # Keyword overlap
            overlap = len(query_words & content_words)
            
            # Combined score (0.7 vector + 0.3 keyword)
            vector_score = 1 - result['score']  # แปลง distance → similarity
            keyword_score = overlap / len(query_words) if query_words else 0
            
            result['rerank_score'] = (0.7 * vector_score) + (0.3 * keyword_score)
        
        # 3. เรียงใหม่
        results.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        return results[:k]
    
    def generate(self, query: str, context: str) -> str:
        """Generate ปกติ"""
        if not context.strip():
            prompt = query
        else:
            prompt = f"""ข้อมูลต่อไปนี้ผ่านการคัดกรอง (re-ranked) แล้ว:

{context}

คำถาม: {query}

วิธีตอบ:
1. ตอบคำถามโดยอิงจากข้อมูลที่เกี่ยวข้องข้างต้น
2. ข้อมูลเหล่านี้ถูกคัดกรองมาแล้วว่าเกี่ยวข้องกับคำถาม
3. ระบุแหล่งที่มาของข้อมูล
4. ตอบเป็นภาษาไทยที่เข้าใจง่าย

คำตอบ:"""
        
        return self.llm.generate(prompt)