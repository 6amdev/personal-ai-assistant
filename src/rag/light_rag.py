"""LightRAG - RAG ที่ใช้ Knowledge Graph

Features:
- ดึง Entity และ Relationship
- สร้าง Knowledge Graph
- การให้เหตุผลแบบ Multi-hop
- 4 query modes: naive, local, global, hybrid

เหมาะกับ:
- คำถามซับซ้อนที่ต้องการ multi-hop reasoning
- ต้องการความสัมพันธ์ระหว่าง entities
- Domain ที่มี entities ชัดเจน (คน, สถานที่, องค์กร)

FIXED:
1. embedding_dim attribute issue
2. self.memory.model vs self.memory.embedding_model
3. Event loop conflict in FastAPI
4. Storage initialization in background thread
"""

import os
import asyncio
from pathlib import Path
from typing import List, Dict, Optional
import logging
import threading

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import base RAG
try:
    from .base_rag import BaseRAG
except ImportError:
    from base_rag import BaseRAG

# Try to import LightRAG
try:
    from lightrag import LightRAG, QueryParam
    from lightrag.kg.shared_storage import initialize_pipeline_status
    HAS_LIGHTRAG = True
    logger.info("✅ LightRAG imported successfully")
except ImportError as e:
    HAS_LIGHTRAG = False
    logger.warning(f"⚠️ LightRAG not available: {e}")
    logger.info("💡 ติดตั้งด้วยคำสั่ง: pip install lightrag-hku networkx nano-vectordb")


class LightRAGWrapper(BaseRAG):
    """
    LightRAG Wrapper - RAG ที่ใช้ Knowledge Graph
    
    วิธีการทำงาน:
    1. สร้าง Knowledge Graph จากเอกสาร
    2. ดึง entities และ relationships
    3. ค้นหาแบบ graph-based (local, global, hybrid)
    4. การให้เหตุผลแบบ Multi-hop
    
    Query Modes:
    - naive: ค้นหา vector แบบธรรมดา
    - local: ใช้ entities ใกล้เคียงกับคำถาม
    - global: ใช้ภาพรวมของ graph ทั้งหมด
    - hybrid: รวม local + global (ดีที่สุด)
    """
    
    def __init__(self, llm_handler, memory_handler, query_mode: str = "hybrid"):
        """
        Initialize LightRAG
        
        พารามิเตอร์:
            llm_handler: LLM Handler
            memory_handler: Memory Handler
            query_mode: Query mode (naive, local, global, hybrid)
        """
        super().__init__(llm_handler, memory_handler, name="LightRAG")
        
        if not HAS_LIGHTRAG:
            raise ImportError(
                "LightRAG ยังไม่ได้ติดตั้ง ติดตั้งด้วยคำสั่ง:\n"
                "pip install lightrag-hku networkx nano-vectordb"
            )
        
        self.query_mode = query_mode
        self.working_dir = "./data/lightrag_db"
        os.makedirs(self.working_dir, exist_ok=True)
        
        # Initialize LightRAG instance
        self.light_rag = None
        self._initialize_lightrag()
        
        logger.info(f"✅ เตรียม LightRAG ที่ {self.working_dir}")
        logger.info(f"🔍 โหมดการค้นหา: {self.query_mode}")
    
    def _initialize_lightrag(self):
        """เตรียม LightRAG ด้วย Ollama backend"""
        
        # ✅ Wrapper for LLM - Create as callable class
        class OllamaLLMWrapper:
            """
            Wrapper class สำหรับ Ollama LLM ที่เรียกได้
            
            LightRAG อาจคาดหวัง callable object ที่มี attributes
            มากกว่าแค่ฟังก์ชัน
            """
            def __init__(self, llm_handler):
                self.llm_handler = llm_handler
            
            async def __call__(
                self,
                prompt: str,
                system_prompt: Optional[str] = None,
                history_messages: List = [],
                **kwargs
            ) -> str:
                """
                เมธอด async สำหรับสร้าง LLM
                
                LightRAG จะเรียกเมธอดนี้
                """
                try:
                    # รวม system prompt กับ user prompt
                    full_prompt = prompt
                    if system_prompt:
                        full_prompt = f"{system_prompt}\n\n{prompt}"
                    
                    # สร้างคำตอบโดยใช้ LLM handler ที่มี
                    # ห่อฟังก์ชัน sync ด้วย run_in_executor
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(
                        None,  # ใช้ executor ตัวเริ่มต้น
                        self.llm_handler.generate,
                        full_prompt
                    )
                    return response
                    
                except Exception as e:
                    logger.error(f"ข้อผิดพลาดในการสร้าง LLM: {e}")
                    return f"Error: {str(e)}"
        
        # Create LLM wrapper instance
        ollama_wrapper = OllamaLLMWrapper(self.llm)
        
        # ✅ FIX: Create EmbeddingWrapper class with embedding_dim attribute
        class EmbeddingWrapper:
            """
            Wrapper class สำหรับ embeddings ที่ LightRAG ต้องการ
            
            LightRAG ต้องการ:
            - embedding_dim: ตัวแปร int ที่สำคัญมาก!
            - เมธอด __call__ สำหรับแปลงข้อความเป็น embeddings
            
            แก้ปัญหา: function ไม่มี attribute embedding_dim
            """
            
            def __init__(self, embedding_model):
                """
                เตรียม wrapper ด้วย embedding model
                
                พารามิเตอร์:
                    embedding_model: SentenceTransformer or compatible model
                """
                self.embedding_model = embedding_model
                
                # ดึงขนาด embedding จาก model (สำคัญมาก!)
                try:
                    # Method 1: ลองใช้ method ตัวติดตั้งของ SentenceTransformer
                    if hasattr(embedding_model, 'get_sentence_embedding_dimension'):
                        self.embedding_dim = embedding_model.get_sentence_embedding_dimension()
                        logger.info(f"✅ Embedding dimension (method 1): {self.embedding_dim}")
                    
                    # Method 2: ลองเอาจาก model config
                    elif hasattr(embedding_model, 'config') and hasattr(embedding_model.config, 'hidden_size'):
                        self.embedding_dim = embedding_model.config.hidden_size
                        logger.info(f"✅ Embedding dimension (method 2): {self.embedding_dim}")
                    
                    # Method 3: encode ข้อความทดสอบเพื่อหาขนาด
                    else:
                        test_embed = embedding_model.encode(
                            ["test"], 
                            convert_to_numpy=True
                        )
                        self.embedding_dim = test_embed.shape[1]
                        logger.info(f"✅ Embedding dimension (method 3): {self.embedding_dim}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ ไม่สามารถหา embedding_dim ได้เอง: {e}")
                    logger.warning("⚠️ ใช้ขนาดเริ่มต้น: 384")
                    self.embedding_dim = 384
                
                logger.info(f"📊 ขนาด embedding สุดท้าย: {self.embedding_dim}")
            
            async def __call__(self, texts: List[str]):
                """
                สร้าง embeddings จากข้อความ (แบบ ASYNC)
                
                สำคัญมาก: ต้องเป็น async เพราะ LightRAG จะ await มัน!
                ข้อผิดพลาด "An asyncio.Future, a coroutine or an awaitable is required"
                เกิดขึ้นเมื่อฟังก์ชันนี้เป็น sync
                
                เมธอดนี้ถูกเรียกโดย LightRAG เพื่อสร้าง embeddings
                
                พารามิเตอร์:
                    texts: รายการ string หรือ string เดี่ยว
                    
                ส่งคืน:
                    numpy array ของ embeddings ที่มีรูปร่าง (len(texts), embedding_dim)
                """
                try:
                    import numpy as np
                    
                    # แปลง string เดี่ยวเป็น list
                    if isinstance(texts, str):
                        texts = [texts]
                    
                    # จัดการข้อมูลที่เป็นเปล่า
                    if not texts:
                        logger.warning("⚠️ ข้อความเปล่าจากการป้อนข้อมูล")
                        return np.zeros((0, self.embedding_dim))
                    
                    # ✅ รัน encode แบบ sync ใน thread pool เพื่อไม่ให้ block event loop
                    loop = asyncio.get_event_loop()
                    embeddings = await loop.run_in_executor(
                        None,  # ใช้ ThreadPoolExecutor ตัวเริ่มต้น
                        lambda: self.embedding_model.encode(
                            texts,
                            normalize_embeddings=True,
                            convert_to_numpy=True,
                            show_progress_bar=False  # ปิดแสดง progress เพื่อให้ logs สะอาด
                        )
                    )
                    
                    # ตรวจสอบรูปร่าง output
                    expected_shape = (len(texts), self.embedding_dim)
                    if embeddings.shape != expected_shape:
                        logger.warning(
                            f"⚠️ รูปร่าง embedding ไม่คาดคิด: {embeddings.shape} "
                            f"(expected: {expected_shape})"
                        )
                    
                    return embeddings
                    
                except Exception as e:
                    logger.error(f"❌ ข้อผิดพลาดในการสร้าง embedding: {e}")
                    import numpy as np
                    # ส่ง embedding ที่เป็น 0 หากเกิดข้อผิดพลาด (backup plan)
                    return np.zeros((len(texts) if isinstance(texts, list) else 1, self.embedding_dim))
        
        # ✅ FIX: ใช้ self.memory.model แทน self.memory.embedding_model
        logger.info("🔧 Creating EmbeddingWrapper instance...")
        
        # ตรวจสอบว่า memory มี attribute ไหน
        if hasattr(self.memory, 'model'):
            embedding_model = self.memory.model
            logger.info("✅ Using self.memory.model")
        elif hasattr(self.memory, 'embedding_model'):
            embedding_model = self.memory.embedding_model
            logger.info("✅ Using self.memory.embedding_model")
        else:
            # Debug: แสดง attributes ที่มี
            available_attrs = [attr for attr in dir(self.memory) if not attr.startswith('_')]
            raise AttributeError(
                f"MemoryHandler does not have 'model' or 'embedding_model' attribute.\n"
                f"Available attributes: {available_attrs}"
            )
        
        # สร้าง embedding wrapper instance
        embedding_func = EmbeddingWrapper(embedding_model)
        
        try:
            # Initialize LightRAG (sync initialization)
            logger.info("🔧 Creating LightRAG instance...")
            self.light_rag = LightRAG(
                working_dir=self.working_dir,
                llm_model_func=ollama_wrapper,  # ✅ Use wrapper instance
                embedding_func=embedding_func,  # ✅ Now has embedding_dim attribute!
            )
            
            # Initialize storages (async required by LightRAG)
            logger.info("🔧 Initializing LightRAG storages...")
            self._run_async_init()
            
            logger.info("✅ LightRAG initialization complete")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize LightRAG: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"LightRAG initialization failed: {e}")
    
    def _run_async_init(self):
        """Run async initialization in sync context"""
        try:
            # Get or create event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, create a new one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run async initialization
            async def init():
                try:
                    await self.light_rag.initialize_storages()
                    logger.info("✅ Initialized storages")
                except Exception as e:
                    logger.warning(f"⚠️ Storage init: {e}")
                
                try:
                    await initialize_pipeline_status()
                    logger.info("✅ Initialized pipeline status")
                except Exception as e:
                    logger.warning(f"⚠️ Pipeline init: {e}")
            
            loop.run_until_complete(init())
            
        except Exception as e:
            logger.warning(f"⚠️ Async init warning: {e}")
            # Continue anyway - storages might already be initialized
    
    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """
        ค้นหาโดยใช้ LightRAG graph search
        
        พารามิเตอร์:
            query: คำถาม
            k: จำนวนผลลัพธ์ (ไม่ใช้ใน LightRAG - ให้ LightRAG จัดการเอง)
            
        ส่งคืน:
            รายการผลลัพธ์ในรูปแบบมาตรฐาน
        """
        try:
            # ✅ Check if LightRAG is initialized
            if not self.light_rag:
                logger.error("❌ LightRAG ยังไม่ได้ initialize")
                return [{
                    'content': "❌ LightRAG ยังไม่ได้ initialize กรุณารอสักครู่",
                    'score': 0.0,
                    'metadata': {'source': 'lightrag_not_initialized', 'type': 'error'}
                }]
            
            # ✅ Check if database has any data
            graph_dir = Path(self.working_dir)
            if not graph_dir.exists() or not any(graph_dir.glob("*")):
                logger.warning("⚠️ ฐานข้อมูล LightRAG ว่างเปล่า! กรุณาอัพโหลดเอกสารก่อน")
                return [{
                    'content': "⚠️ ยังไม่มีข้อมูลใน Knowledge Graph กรุณา Upload เอกสารก่อนครับ",
                    'score': 0.0,
                    'metadata': {
                        'source': 'lightrag_empty',
                        'mode': self.query_mode,
                        'type': 'empty_database'
                    }
                }]
            
            logger.info(f"🔍 ค้นหา LightRAG (โหมด: {self.query_mode}): {query[:50]}...")
            
            # Create query param
            query_param = QueryParam(mode=self.query_mode)
            
            # ✅ Query LightRAG with timeout
            async def query_with_timeout(timeout=30):
                """Query with timeout protection"""
                return await asyncio.wait_for(
                    self.light_rag.aquery(query, param=query_param),
                    timeout=timeout
                )
            
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Create new loop if current is running
                    import nest_asyncio
                    nest_asyncio.apply()
                    result = asyncio.run(query_with_timeout(timeout=30))
                else:
                    result = loop.run_until_complete(query_with_timeout(timeout=30))
            except asyncio.TimeoutError:
                logger.error("❌ หมดเวลาการค้นหา LightRAG (30 วินาที)")
                return [{
                    'content': "⏱️ การค้นหาใช้เวลานานเกินไป กรุณาลองใหม่อีกครั้ง",
                    'score': 0.0,
                    'metadata': {'source': 'timeout', 'error': 'Query timeout'}
                }]
            except RuntimeError as e:
                logger.error(f"❌ RuntimeError in query: {e}")
                # Fallback: create new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(query_with_timeout(timeout=30))
                except asyncio.TimeoutError:
                    logger.error("❌ หมดเวลาการค้นหา LightRAG (30 วินาที)")
                    return [{
                        'content': "⏱️ การค้นหาใช้เวลานานเกินไป กรุณาลองใหม่อีกครั้ง",
                        'score': 0.0,
                        'metadata': {'source': 'timeout', 'error': 'Query timeout'}
                    }]
            
            logger.info(f"✅ เสร็จสิ้นการค้นหา LightRAG: {len(result) if result else 0} ตัวอักษร")
            
            # Format result to match BaseRAG interface
            return [{
                'content': result if result else "ไม่พบข้อมูลที่เกี่ยวข้อง",
                'score': 1.0,
                'metadata': {
                    'source': 'lightrag_graph',
                    'mode': self.query_mode,
                    'type': 'graph_search'
                }
            }]
            
        except Exception as e:
            logger.error(f"❌ ข้อผิดพลาดในการค้นหา LightRAG: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return [{
                'content': f"Error querying LightRAG: {str(e)}",
                'score': 0.0,
                'metadata': {'source': 'error', 'error': str(e)}
            }]
    
    def generate(self, query: str, context: str) -> str:
        """
        สร้างคำตอบจาก LightRAG context
        
        พารามิเตอร์:
            query: คำถาม
            context: Context from retrieve()
            
        ส่งคืน:
            คำตอบที่จัดรูปแบบแล้ว
        """
        try:
            if context and context.strip():
                answer = context
                if "[1]" in answer:
                    lines = answer.split("\n")
                    clean_lines = []
                    for line in lines:
                        if line.strip() and not line.strip().startswith("[") or "]" not in line[:5]:
                            clean_lines.append(line)
                    answer = "\n".join(clean_lines)
                
                return answer.strip()
            
            # Fallback: Use LLM directly if context is empty
            logger.warning("⚠️ Empty context from LightRAG, using LLM fallback")
            prompt = f"""คำถาม: {query}

ตอบคำถามนี้เป็นภาษาไทยอย่างกระชับ:"""
            
            return self.llm.generate(prompt)
            
        except Exception as e:
            logger.error(f"❌ Generation error: {e}")
            return f"เกิดข้อผิดพลาดในการสร้างคำตอบ: {str(e)}"
    
    def insert_documents(self, texts: List[str], metadatas: Optional[List[dict]] = None):
        """
        Insert documents into LightRAG Knowledge Graph
        
        สร้าง entities ความสัมพันธ์ และ graph structure จากเอกสาร
        
        พารามิเตอร์:
            texts: รายการของข้อความที่ต้องการเพิ่ม
            metadatas: รายการ metadata (ไม่จำเป็น)
        """
        if not texts:
            logger.warning("⚠️ ไม่มีข้อความที่ต้องการเพิ่ม")
            return
        
        if not self.light_rag:
            logger.error("❌ LightRAG ยังไม่ได้ initialize")
            return
        
        try:
            logger.info(f"📚 เพิ่ม {len(texts)} เอกสารเข้า LightRAG...")
            combined_text = "\n\n".join(texts)
            
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            if loop.is_running():
                logger.info("⚙️ Event loop กำลังรัน (FastAPI) กำลังเพิ่มข้อมูลใน background thread...")
                
                def insert_in_thread():
                    try:
                        thread_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(thread_loop)
                        
                        async def do_insert():
                            try:
                                await self.light_rag.initialize_storages()
                                await initialize_pipeline_status()
                                logger.info("✅ Re-initialized storages in thread")
                            except Exception as e:
                                logger.warning(f"⚠️ Storage re-init in thread: {e}")
                            
                            await self.light_rag.ainsert(combined_text)
                        
                        thread_loop.run_until_complete(do_insert())
                        thread_loop.close()
                        logger.info("✅ เพิ่มเอกสารเข้า LightRAG Knowledge Graph เรียบร้อย!")
                    except Exception as e:
                        logger.error(f"❌ ข้อผิดพลาดในการเพิ่มข้อมูลใน thread: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                
                insert_thread = threading.Thread(target=insert_in_thread, daemon=False)
                insert_thread.start()
                logger.info("ℹ️ เริ่มเพิ่มข้อมูลใน background thread แล้ว...")
            else:
                logger.info("⚙️ Event loop พร้อมใช้ กำลังเพิ่มข้อมูล...")
                
                async def do_insert():
                    await self.light_rag.ainsert(combined_text)
                
                loop.run_until_complete(do_insert())
                logger.info(f"✅ เพิ่มเอกสารเข้า LightRAG Knowledge Graph เรียบร้อย!")
            
        except Exception as e:
            logger.error(f"❌ ข้อผิดพลาดในการเพิ่มเอกสาร: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def set_query_mode(self, mode: str):
        """
        Change query mode
        
        พารามิเตอร์:
            mode: Query mode (naive, local, global, hybrid)
        """
        valid_modes = ["naive", "local", "global", "hybrid"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode. Must be one of: {valid_modes}")
        
        self.query_mode = mode
        logger.info(f"🔄 Query mode changed to: {mode}")
    
    def get_query_mode(self) -> str:
        """Get current query mode"""
        return self.query_mode
    
    def get_graph_stats(self) -> Dict:
        """
        Get knowledge graph statistics
        
        ส่งคืน:
            Dict with graph statistics (if available)
        """
        try:
            stats = {
                "working_dir": self.working_dir,
                "query_mode": self.query_mode,
                "status": "initialized"
            }
            
            graph_dir = Path(self.working_dir)
            if graph_dir.exists():
                files = list(graph_dir.glob("*"))
                stats["files_count"] = len(files)
                stats["storage_exists"] = True
            else:
                stats["storage_exists"] = False
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting graph stats: {e}")
            return {"error": str(e)}


def is_lightrag_available() -> bool:
    """
    Check if LightRAG is installed and available
    
    ส่งคืน:
        True if LightRAG is available, False otherwise
    """
    return HAS_LIGHTRAG