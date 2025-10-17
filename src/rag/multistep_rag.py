"""Multi-step RAG - ค้นหาหลายรอบถ้าข้อมูลไม่พอ"""
from .base_rag import BaseRAG
from typing import List, Dict

class MultiStepRAG(BaseRAG):
    """
    Multi-step RAG - ค้นหาหลายรอบ
    
    วิธีทำงาน:
    1. ค้นหารอบแรก
    2. เช็คว่าข้อมูลพอไหม
    3. ถ้าไม่พอ → สร้างคำถามย่อย → ค้นหาเพิ่ม
    4. รวมข้อมูลทั้งหมด → ตอบ
    
    ดีสำหรับคำถามซับซ้อน
    """
    
    def __init__(self, llm_handler, memory_handler, max_steps: int = 3):
        super().__init__(llm_handler, memory_handler, name="Multi-step RAG")
        self.max_steps = max_steps
    
    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """ค้นหารอบแรก"""
        return self.memory.search(query, k=k)
    
    def generate(self, query: str, context: str) -> str:
        """Generate แบบ multi-step"""
        all_contexts = [context] if context else []
        current_query = query
        
        for step in range(self.max_steps):
            # สร้าง prompt
            step_context = "\n\n".join(all_contexts)
            
            if step == 0:
                # รอบแรก - ลองตอบเลย
                check_prompt = f"""ข้อมูล:
{step_context}

คำถาม: {current_query}

ตอบคำถาม หรือ ถ้าข้อมูลไม่พอให้บอก "ต้องการข้อมูลเพิ่มเติมเกี่ยวกับ: [หัวข้อ]"

คำตอบ:"""
            else:
                # รอบถัดไป - ตอบจากข้อมูลที่รวมแล้ว
                check_prompt = f"""ข้อมูลรวม:
{step_context}

คำถามเดิม: {query}

ตอบคำถามโดยอิงจากข้อมูลทั้งหมด:"""
            
            partial = self.llm.generate(check_prompt)
            
            # เช็คว่าต้องการข้อมูลเพิ่มไหม
            if "ต้องการข้อมูลเพิ่มเติม" not in partial:
                # พอแล้ว
                return partial
            
            # ดึงหัวข้อใหม่
            try:
                new_topic = partial.split("ต้องการข้อมูลเพิ่มเติมเกี่ยวกับ:")[1].split("\n")[0].strip()
                
                # ค้นหาเพิ่ม
                new_results = self.memory.search(new_topic, k=2)
                new_context = "\n\n".join([r['content'] for r in new_results])
                
                if new_context.strip():
                    all_contexts.append(f"[ข้อมูลเพิ่มเติมเกี่ยวกับ {new_topic}]\n{new_context}")
                
                current_query = new_topic
            except:
                break
        
        # Final answer
        final_prompt = f"""ข้อมูลทั้งหมด:
{chr(10).join(all_contexts)}

คำถามเดิม: {query}

สรุปคำตอบจากข้อมูลทั้งหมด:"""
        
        return self.llm.generate(final_prompt)