"""Query Rewrite RAG - เขียนคำถามใหม่หลายแบบ"""
from .base_rag import BaseRAG
from typing import List, Dict

class QueryRewriteRAG(BaseRAG):
    """
    Query Rewrite RAG - เขียนคำถามใหม่
    
    วิธีทำงาน:
    1. เขียนคำถามใหม่ 2-3 แบบ (synonym expansion)
    2. ค้นหาแต่ละแบบ
    3. รวมผลลัพธ์ + de-duplicate
    
    ดีสำหรับคำถามที่คลุมเครือ
    """
    
    def __init__(self, llm_handler, memory_handler):
        super().__init__(llm_handler, memory_handler, name="Query Rewrite RAG")
    
    def _rewrite_query(self, query: str) -> List[str]:
        """เขียนคำถามใหม่แบบง่าย (ไม่ใช้ LLM)"""
        queries = [query]
        
        # Synonym expansion
        synonyms = {
            'อธิบาย': ['บอก', 'คืออะไร', 'หมายถึง'],
            'วิธี': ['วิธีการ', 'ทำอย่างไร', 'แนวทาง', 'ขั้นตอน'],
            'สรุป': ['สรุป', 'ย่อความ', 'โดยสรุป', 'ประเด็นสำคัญ'],
            'ข้อดี': ['ข้อดี', 'ประโยชน์', 'คุณสมบัติ', 'จุดเด่น'],
            'ข้อเสีย': ['ข้อเสีย', 'ปัญหา', 'ข้อจำกัด', 'จุดอ่อน'],
            'ทำไม': ['ทำไม', 'เพราะอะไร', 'สาเหตุ'],
            'เปรียบเทียบ': ['เปรียบเทียบ', 'ต่างกัน', 'แตกต่าง'],
        }
        
        for word, syns in synonyms.items():
            if word in query.lower():
                for syn in syns[:2]:  # เอา 2 คำแรก
                    new_query = query.lower().replace(word, syn)
                    if new_query not in [q.lower() for q in queries]:
                        queries.append(new_query)
        
        return queries[:3]  # สูงสุด 3 queries
    
    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """ค้นหาหลาย query แล้วรวม"""
        # Rewrite
        queries = self._rewrite_query(query)
        
        # ค้นหาแต่ละ query
        all_results = []
        seen = set()
        
        for q in queries:
            results = self.memory.search(q, k=k)
            for r in results:
                content = r['content']
                if content not in seen:
                    seen.add(content)
                    all_results.append(r)
        
        # เรียงตาม score
        all_results.sort(key=lambda x: x['score'])
        
        return all_results[:k]
    
    def generate(self, query: str, context: str) -> str:
        """Generate ปกติ"""
        if not context.strip():
            prompt = query
        else:
            prompt = f"""ข้อมูลจากการค้นหาหลายรูปแบบคำถาม:

{context}

คำถามเดิม: {query}

วิธีตอบ:
1. ตอบโดยอิงจากข้อมูลที่เกี่ยวข้อง
2. ข้อมูลมาจากการค้นหาหลายแบบ
3. ระบุแหล่งที่มา
4. ตอบเป็นภาษาไทยที่เข้าใจง่าย

คำตอบ:"""
        
        return self.llm.generate(prompt)