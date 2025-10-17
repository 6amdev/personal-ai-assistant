"""Hybrid RAG - BM25 + Vector Search"""
from .base_rag import BaseRAG
from typing import List, Dict

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    print("⚠️ ติดตั้ง BM25: pip install rank-bm25")

class HybridRAG(BaseRAG):
    """
    Hybrid RAG - รวม BM25 (keyword) + Vector (semantic)
    
    วิธีทำงาน:
    1. BM25 search (ดีกับ keyword exact match)
    2. Vector search (ดีกับความหมาย)
    3. รวมคะแนน (weighted average)
    
    ดีที่สุดสำหรับภาษาไทย!
    - BM25 จับ keyword ไทยได้ดี
    - Vector จับความหมายได้ดี
    """
    
    def __init__(self, llm_handler, memory_handler, alpha: float = 0.5):
        super().__init__(llm_handler, memory_handler, name="Hybrid RAG")
        self.alpha = alpha  # น้ำหนัก (0=BM25 only, 1=Vector only)
        self.bm25 = None
        self.documents = []
        if HAS_BM25:
            self._init_bm25()
    
    def _init_bm25(self):
        """สร้าง BM25 index"""
        try:
            collection = self.memory.vectorstore._collection
            results = collection.get()
            
            if results and 'documents' in results:
                self.documents = results['documents']
                tokenized = [doc.lower().split() for doc in self.documents]
                self.bm25 = BM25Okapi(tokenized)
                print(f"✅ BM25 indexed {len(self.documents)} docs")
        except Exception as e:
            print(f"⚠️ BM25 init failed: {e}")
    
    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """Hybrid search"""
        if not HAS_BM25 or not self.bm25:
            # Fallback to vector only
            return self.memory.search(query, k=k)
        
        # 1. BM25 Search
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # Normalize (0-1)
        if bm25_scores.max() > bm25_scores.min():
            bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())
        
        # 2. Vector Search
        vector_results = self.memory.search(query, k=k*2)
        
        # 3. Combine scores
        vector_scores = {}
        for r in vector_results:
            try:
                idx = self.documents.index(r['content'])
                vector_scores[idx] = 1 - r['score']  # distance → similarity
            except:
                pass
        
        # 4. Hybrid scoring
        combined = []
        for idx, doc in enumerate(self.documents):
            bm25_s = bm25_scores[idx]
            vector_s = vector_scores.get(idx, 0)
            
            # Weighted average
            final_score = (1 - self.alpha) * bm25_s + self.alpha * vector_s
            
            combined.append({
                'content': doc,
                'score': 1 - final_score,  # กลับเป็น distance
                'metadata': {},
                'hybrid_score': final_score,
                'bm25_score': bm25_s,
                'vector_score': vector_s
            })
        
        # Sort by hybrid score
        combined.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return combined[:k]
    
    def generate(self, query: str, context: str) -> str:
        """Generate ปกติ"""
        if not context.strip():
            prompt = query
        else:
            prompt = f"""ข้อมูลจาก Hybrid Search (BM25 + Vector):

{context}

คำถาม: {query}

วิธีตอบ:
1. ข้อมูลนี้ผ่านการค้นหา 2 วิธี (keyword + ความหมาย)
2. ตอบโดยอิงจากข้อมูลที่เกี่ยวข้อง
3. ระบุแหล่งที่มา
4. ตอบเป็นภาษาไทยที่เข้าใจง่าย

คำตอบ:"""
        
        return self.llm.generate(prompt)