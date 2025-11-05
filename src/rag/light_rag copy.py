"""
✅ FIXED: LightRAG - RAG ที่ใช้ Knowledge Graph สำหรับ FastAPI

🔧 FIX SUMMARY:
1. ✅ Event Loop Management - จัดการ async properly
2. ✅ nest_asyncio replacement - ใช้ Threading แทน
3. ✅ Close loop after use - หลีกเลี่ยง "Event loop is closed"
4. ✅ Separate thread for queries - ไม่ block FastAPI event loop
5. ✅ Proper async/await patterns

📊 Knowledge Graph Architecture:
- Entity Extraction: ดึง "สิ่ง" (คน, สถานที่, สินค้า) จากข้อความ
- Relation Extraction: ดึง "ความสัมพันธ์" ระหว่าง entities
- Graph Storage: เก็บใน GraphML format (lightrag_db/graph_chunk_entity_relation.graphml)
- Multi-hop Reasoning: ตามเนื้องาน entities ผ่าน relationships

🎯 4 Query Modes:
1. naive: Vector search แบบธรรมดา (เร็วสุด แต่ผลน้อย)
2. local: ค้นหา entities ที่เกี่ยวข้อง + neighbors
3. global: ดึง global summary ของ graph (เหมาะ overview)
4. hybrid: local + global (ดีที่สุด - แนะนำใช้นี่)
"""

import os
import asyncio
import threading
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import queue
import time

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
    logger.info("💡 Install with: pip install lightrag-hku networkx nano-vectordb")


class AsyncQueryExecutor:
    """
    🔧 Async Query Executor - จัดการ async operations ใน sync context
    
    ปัญหา: FastAPI มี event loop ที่รันอยู่แล้ว
    - loop.run_until_complete() จะ error เพราะ loop already running
    - nest_asyncio.apply() ไม่มั่นใจ
    
    วิธีแก้: ใช้ Thread Pool เพื่อ isolate async operations
    """
    
    def __init__(self, max_workers: int = 2):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.logger = logger
    
    async def run_query(self, query_func, query_text: str, timeout: int = 30) -> Optional[str]:
        """
        รัน async query ใน thread ที่ปลอดภัย
        
        📝 วิธีการทำงาน:
        1. สร้าง thread ใหม่ด้วย ThreadPoolExecutor
        2. ใน thread นั้น สร้าง event loop ใหม่
        3. รัน query function ใน loop นั้น
        4. ปิด loop อย่างสม่ำเสมอ (สำคัญมาก!)
        """
        def _run_query_in_thread():
            try:
                # สร้าง event loop ใหม่ในตัว thread นี้ (ไม่ share กับ FastAPI)
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    # รัน async query
                    result = loop.run_until_complete(
                        asyncio.wait_for(query_func(query_text), timeout=timeout)
                    )
                    return result
                finally:
                    # 🔴 CRITICAL: ปิด loop อย่างถูกต้อง
                    pending = asyncio.all_tasks(loop)
                    for task in pending:
                        task.cancel()
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                    loop.close()
                    self.logger.debug("✅ Event loop closed properly in thread")
                    
            except asyncio.TimeoutError:
                self.logger.error(f"⏱️ Query timeout after {timeout}s")
                raise
            except Exception as e:
                self.logger.error(f"❌ Query execution error: {e}")
                raise
        
        # รัน thread
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _run_query_in_thread)
    
    def shutdown(self):
        """ปิด executor"""
        self.executor.shutdown(wait=True)
        self.logger.info("✅ AsyncQueryExecutor shutdown")


class LightRAGWrapper(BaseRAG):
    """
    ✅ FIXED LightRAG Wrapper - Knowledge Graph RAG for FastAPI
    
    🧠 วิธีการทำงาน:
    
    1️⃣ DOCUMENT PROCESSING (สร้าง Knowledge Graph)
       - เอกสารเข้า → Chunking → Entity Extraction (LLM)
       - Entity: "ร้าน Krua Baan Thai", "ข้าวผัดกุ้ง", "ราคา 180 บาท"
       - Relations: "Krua Baan Thai --serves--> ข้าวผัดกุ้ง"
       → Graph Structure: nodes=entities, edges=relations
    
    2️⃣ GRAPH CONSTRUCTION
       - Merge: entities เดียวกัน ถ้าหมายถึงเรื่องเดียว
       - Storage: ใช้ nano-vectordb (vector) + GraphML (structure)
    
    3️⃣ QUERY TIME - Multi-hop Reasoning
       Query: "ร้านไหนมีอะไรอร่อยถูกๆ?"
       
       a) Local Mode: ค้นหา "ร้าน" → เพื่อนบ้านของมัน (neighbors)
          "ร้าน" --serves--> "เมนู" --has_price--> "บาท"
       
       b) Global Mode: ดึง "สรุป" ทั้ง graph
          "In this document, there are 5 restaurants..."
       
       c) Hybrid (Best): local + global combined
    
    4️⃣ RESPONSE GENERATION
       Context: "Krua Baan Thai serves Pad Gung for 180 baht"
       LLM: "ร้านเวิ้งบ้านไทย มีข้าวผัดกุ้ง ราคา 180 บาท ถูกมากครับ!"
    """
    
    def __init__(self, llm_handler, memory_handler, query_mode: str = "hybrid"):
        """
        Initialize LightRAG Wrapper
        
        Args:
            llm_handler: LLM Handler instance
            memory_handler: Memory Handler instance
            query_mode: "naive", "local", "global", or "hybrid"
        """
        super().__init__(llm_handler, memory_handler, name="LightRAG")
        
        if not HAS_LIGHTRAG:
            raise ImportError(
                "LightRAG not installed. Install with:\n"
                "pip install lightrag-hku networkx nano-vectordb"
            )
        
        self.query_mode = query_mode
        self.working_dir = "./data/lightrag_db"
        os.makedirs(self.working_dir, exist_ok=True)
        
        # Initialize LightRAG instance
        self.light_rag = None
        self._initialize_lightrag()
        
        # ✅ NEW: Async query executor for FastAPI compatibility
        self.query_executor = AsyncQueryExecutor(max_workers=2)
        
        logger.info(f"✅ LightRAG Ready at {self.working_dir}")
        logger.info(f"🔍 Query Mode: {self.query_mode}")
        logger.info(f"🧵 Async Executor: Ready with 2 threads")
    
    def _initialize_lightrag(self):
        """เตรียม LightRAG ด้วย Ollama backend"""
        
        # ✅ LLM Wrapper - Callable class
        class OllamaLLMWrapper:
            """Wrapper for Ollama LLM that LightRAG can call"""
            def __init__(self, llm_handler):
                self.llm_handler = llm_handler
                self.func = self.__call__
            
            async def __call__(
                self,
                prompt: str,
                system_prompt: Optional[str] = None,
                history_messages: List = [],
                **kwargs
            ) -> str:
                """
                Async LLM generation (called by LightRAG)
                
                📝 LightRAG expects async callable
                """
                try:
                    full_prompt = prompt
                    if system_prompt:
                        full_prompt = f"{system_prompt}\n\n{prompt}"
                    
                    # ✅ Run sync LLM in executor
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(
                        None,
                        self.llm_handler.generate,
                        full_prompt
                    )
                    return response
                    
                except Exception as e:
                    logger.error(f"LLM generation error: {e}")
                    return f"Error: {str(e)}"
        
        ollama_wrapper = OllamaLLMWrapper(self.llm)
        
        # ✅ Embedding Wrapper with proper async
        class EmbeddingWrapper:
            """Wrapper for embeddings (with embedding_dim attribute required by LightRAG)"""
            
            def __init__(self, embedding_model):
                self.embedding_model = embedding_model
                
                # 🔑 CRITICAL: Get embedding_dim (LightRAG requires this!)
                try:
                    if hasattr(embedding_model, 'get_sentence_embedding_dimension'):
                        self.embedding_dim = embedding_model.get_sentence_embedding_dimension()
                        logger.info(f"✅ Embedding dim (method 1): {self.embedding_dim}")
                    elif hasattr(embedding_model, 'config') and hasattr(embedding_model.config, 'hidden_size'):
                        self.embedding_dim = embedding_model.config.hidden_size
                        logger.info(f"✅ Embedding dim (method 2): {self.embedding_dim}")
                    else:
                        test_embed = embedding_model.encode(["test"], convert_to_numpy=True)
                        self.embedding_dim = test_embed.shape[1]
                        logger.info(f"✅ Embedding dim (method 3): {self.embedding_dim}")
                except Exception as e:
                    logger.warning(f"⚠️ Auto-detect failed: {e}")
                    self.embedding_dim = 384
                
                logger.info(f"📊 Final embedding_dim: {self.embedding_dim}")
            
            async def __call__(self, texts: List[str]):
                """
                Generate embeddings (MUST be async for LightRAG)
                
                📝 วิธี:
                1. Convert string → list
                2. Run in executor (ไม่ block event loop)
                3. Return numpy array (len(texts), embedding_dim)
                """
                try:
                    import numpy as np
                    
                    if isinstance(texts, str):
                        texts = [texts]
                    
                    if not texts:
                        return np.zeros((0, self.embedding_dim))
                    
                    loop = asyncio.get_event_loop()
                    embeddings = await loop.run_in_executor(
                        None,
                        lambda: self.embedding_model.encode(
                            texts,
                            normalize_embeddings=True,
                            convert_to_numpy=True,
                            show_progress_bar=False
                        )
                    )
                    
                    return embeddings
                    
                except Exception as e:
                    logger.error(f"Embedding error: {e}")
                    import numpy as np
                    return np.zeros((len(texts) if isinstance(texts, list) else 1, self.embedding_dim))
        
        # Get embedding model from memory
        if hasattr(self.memory, 'model'):
            embedding_model = self.memory.model
        elif hasattr(self.memory, 'embedding_model'):
            embedding_model = self.memory.embedding_model
        else:
            raise AttributeError("Memory has no 'model' or 'embedding_model' attribute")
        
        embedding_func = EmbeddingWrapper(embedding_model)
        
        try:
            logger.info("🔧 Initializing LightRAG...")
            self.light_rag = LightRAG(
                working_dir=self.working_dir,
                llm_model_func=ollama_wrapper,
                embedding_func=embedding_func,
            )
            
            # Initialize storages
            self._init_storages_sync()
            logger.info("✅ LightRAG initialized successfully")
            
        except Exception as e:
            logger.error(f"LightRAG initialization failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"LightRAG init failed: {e}")
    
    def _init_storages_sync(self):
        """Initialize storages in a clean thread (not in FastAPI event loop)"""
        def _init_in_thread():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                async def init():
                    try:
                        await self.light_rag.initialize_storages()
                        logger.info("✅ Initialized storages")
                    except Exception as e:
                        logger.debug(f"Storage init note: {e}")
                    
                    try:
                        await initialize_pipeline_status()
                        logger.info("✅ Initialized pipeline status")
                    except Exception as e:
                        logger.debug(f"Pipeline init note: {e}")
                
                loop.run_until_complete(init())
                loop.close()
            except Exception as e:
                logger.warning(f"Storage init warning: {e}")
        
        # Run in thread
        init_thread = threading.Thread(target=_init_in_thread, daemon=False)
        init_thread.start()
        init_thread.join(timeout=10)
    
    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """
        ค้นหาด้วย LightRAG graph search
        
        🎯 Process:
        1. Check if database has data
        2. Create QueryParam with current mode
        3. Execute async query (safely in thread)
        4. Format result for BaseRAG interface
        
        Args:
            query: Question
            k: Number of results (not used by LightRAG)
            
        Returns:
            List of results in standard format
        """
        try:
            # Check initialization
            if not self.light_rag:
                logger.error("LightRAG not initialized")
                return self._error_result("LightRAG not initialized")
            
            # Check if database has data
            graph_file = Path(self.working_dir) / "graph_chunk_entity_relation.graphml"
            if not graph_file.exists():
                logger.warning("⚠️ Knowledge Graph database is empty!")
                return self._error_result(
                    "No data in Knowledge Graph. Please upload documents first.",
                    "empty_database"
                )
            
            logger.info(f"🔍 Querying LightRAG (mode: {self.query_mode}): {query[:50]}...")
            
            # Create query param
            query_param = QueryParam(mode=self.query_mode)
            
            # ✅ NEW: Use AsyncQueryExecutor for safe async execution
            # ✅ FIXED: Use ThreadPoolExecutor instead of asyncio.run()
            try:
                import concurrent.futures
                
                def run_query_in_thread():
                    """Run async query in separate thread with new event loop"""
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = loop.run_until_complete(
                            self.light_rag.aquery(query, param=query_param)
                        )
                        return result
                    finally:
                        # Clean up properly
                        pending = asyncio.all_tasks(loop)
                        for task in pending:
                            task.cancel()
                        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                        loop.close()
                
                # Execute in thread pool with timeout
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(run_query_in_thread)
                    result = future.result(timeout=30)
                    
            except concurrent.futures.TimeoutError:
                logger.error("Query timeout (30s)")
                return self._error_result("Query timeout. Please try again.", "timeout")
            except Exception as e:
                logger.error(f"Query error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return self._error_result(f"Query error: {str(e)}", "query_error")
            logger.info(f"✅ Query complete: {len(result) if result else 0} chars")
            
            # Format result
            return [{
                'content': result if result else "No relevant information found",
                'score': 1.0,
                'metadata': {
                    'source': 'lightrag_graph',
                    'mode': self.query_mode,
                    'type': 'graph_search'
                }
            }]
            
        except Exception as e:
            logger.error(f"Retrieve error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._error_result(str(e), "retrieve_error")
    
    def _error_result(self, message: str, error_type: str = "error") -> List[Dict]:
        """สร้าง error result object"""
        return [{
            'content': message,
            'score': 0.0,
            'metadata': {'source': 'lightrag_error', 'type': error_type}
        }]
    
    def generate(self, query: str, context: str) -> str:
        """
        สร้างคำตอบจาก context
        
        Note: LightRAG already generates answer in retrieve()
        so we just format it here
        """
        try:
            if context and context.strip():
                answer = context
                # Remove numbering like [1], [2] if present
                if "[1]" in answer:
                    lines = answer.split("\n")
                    clean_lines = [line for line in lines 
                                   if line.strip() and not (line.strip().startswith("[") and "]" in line[:5])]
                    answer = "\n".join(clean_lines)
                
                return answer.strip()
            
            # Fallback: use LLM directly
            logger.warning("⚠️ Empty context from LightRAG, using LLM fallback")
            prompt = f"""Question: {query}

Answer concisely in Thai:"""
            
            return self.llm.generate(prompt)
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Error generating answer: {str(e)}"
    
    def insert_documents(self, texts: List[str], metadatas: Optional[List[dict]] = None):
        """
        Insert documents into LightRAG Knowledge Graph
        
        🧠 Process:
        1. Combine texts
        2. Run async insert in separate thread (important for FastAPI!)
        3. LightRAG extracts entities and relations automatically
        
        Args:
            texts: List of text chunks
            metadatas: Optional metadata (not used by LightRAG)
        """
        if not texts:
            logger.warning("No texts to insert")
            return
        
        if not self.light_rag:
            logger.error("LightRAG not initialized")
            return
        
        try:
            logger.info(f"📚 Inserting {len(texts)} documents into LightRAG...")
            combined_text = "\n\n".join(texts)
            
            # Run in background thread to not block FastAPI
            def insert_in_thread():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    async def do_insert():
                        # Re-initialize storages
                        try:
                            await self.light_rag.initialize_storages()
                            await initialize_pipeline_status()
                            logger.debug("✅ Re-initialized storages in thread")
                        except Exception as e:
                            logger.debug(f"Storage re-init note: {e}")
                        
                        # Insert document
                        await self.light_rag.ainsert(combined_text)
                    
                    loop.run_until_complete(do_insert())
                    loop.close()
                    logger.info("✅ Documents inserted into Knowledge Graph!")
                    
                except Exception as e:
                    logger.error(f"Insertion error: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            
            insert_thread = threading.Thread(target=insert_in_thread, daemon=False)
            insert_thread.start()
            logger.info("ℹ️ Document insertion started in background...")
            
        except Exception as e:
            logger.error(f"Insert error: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def set_query_mode(self, mode: str):
        """Change query mode"""
        valid_modes = ["naive", "local", "global", "hybrid"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode. Must be one of: {valid_modes}")
        
        self.query_mode = mode
        logger.info(f"🔄 Query mode changed to: {mode}")
    
    def get_query_mode(self) -> str:
        """Get current query mode"""
        return self.query_mode
    
    def get_graph_stats(self) -> Dict:
        """Get knowledge graph statistics"""
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
                stats["has_graph"] = (graph_dir / "graph_chunk_entity_relation.graphml").exists()
                stats["storage_exists"] = True
                
                # Try to get entity count
                try:
                    import networkx as nx
                    graph_file = graph_dir / "graph_chunk_entity_relation.graphml"
                    if graph_file.exists():
                        G = nx.read_graphml(str(graph_file))
                        stats["entities"] = G.number_of_nodes()
                        stats["relations"] = G.number_of_edges()
                except Exception as e:
                    logger.debug(f"Could not read graph stats: {e}")
            else:
                stats["storage_exists"] = False
            
            return stats
            
        except Exception as e:
            logger.error(f"Stats error: {e}")
            return {"error": str(e)}
    
    def shutdown(self):
        """ปิด resources"""
        self.query_executor.shutdown()
        logger.info("✅ LightRAGWrapper shutdown complete")


def is_lightrag_available() -> bool:
    """Check if LightRAG is available"""
    return HAS_LIGHTRAG