"""Base RAG Class - Template สำหรับ RAG ทุกแบบ"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseRAG(ABC):
    """Base class สำหรับ RAG ทุกแบบ"""
    
    def __init__(self, llm_handler, memory_handler, name: str = "BaseRAG"):
        """
        Initialize RAG
        
        Args:
            llm_handler: LLM Handler (จาก src/llm.py)
            memory_handler: Memory Handler (จาก src/memory.py)
            name: ชื่อ RAG
        """
        self.llm = llm_handler
        self.memory = memory_handler
        self.name = name
    
    @abstractmethod
    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """
        Retrieve relevant documents
        
        Args:
            query: คำถาม
            k: จำนวน documents ที่ต้องการ
            
        Returns:
            List of {'content': str, 'score': float, 'metadata': dict}
        """
        pass
    
    @abstractmethod
    def generate(self, query: str, context: str) -> str:
        """
        Generate answer from context
        
        Args:
            query: คำถาม
            context: ข้อมูลที่ retrieve มา
            
        Returns:
            คำตอบ (string)
        """
        pass
    
    def query(self, query: str, k: int = 3) -> Dict[str, Any]:
        """
        Main method: Query → Retrieve → Generate
        
        Args:
            query: คำถาม
            k: จำนวน documents
            
        Returns:
            {
                'answer': str,           # คำตอบ
                'context': str,          # Context ที่ใช้
                'sources': List[str],    # แหล่งที่มา
                'rag_type': str          # ชื่อ RAG
            }
        """
        # 1. Retrieve
        results = self.retrieve(query, k=k)
        
        # 2. Build context
        contexts = [r['content'] for r in results]
        sources = [r.get('metadata', {}).get('source', 'unknown') for r in results]
        context = "\n\n".join([f"[{i+1}] {c}" for i, c in enumerate(contexts)])
        
        # 3. Generate
        answer = self.generate(query, context)
        
        return {
            'answer': answer,
            'context': context,
            'sources': list(set(sources)),
            'rag_type': self.name
        }