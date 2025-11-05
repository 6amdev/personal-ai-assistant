"""LightRAG - Simple Version (ไม่มี class wrapper ซับซ้อน)"""
import os
import asyncio
import threading
import logging
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from .base_rag import BaseRAG
except ImportError:
    from base_rag import BaseRAG

try:
    from lightrag import LightRAG, QueryParam
    from lightrag.kg.shared_storage import initialize_pipeline_status
    HAS_LIGHTRAG = True
except ImportError:
    HAS_LIGHTRAG = False
    logger.warning("⚠️ ติดตั้ง LightRAG: pip install lightrag-hku networkx nano-vectordb")


class LightRAGWrapper(BaseRAG):
    """LightRAG - Knowledge Graph RAG (แบบง่าย)"""
    
    def __init__(self, llm_handler, memory_handler, query_mode: str = "hybrid"):
        super().__init__(llm_handler, memory_handler, name="LightRAG")
        
        if not HAS_LIGHTRAG:
            raise ImportError("LightRAG not installed")
        
        self.query_mode = query_mode
        self.working_dir = "./data/lightrag_db"
        os.makedirs(self.working_dir, exist_ok=True)
        
        self.light_rag = None
        self._initialize_lightrag()
        
        logger.info(f"✅ LightRAG Ready at {self.working_dir}")
        logger.info(f"🔍 Query Mode: {self.query_mode}")
    
    def _initialize_lightrag(self):
        """Initialize LightRAG - แบบง่าย ไม่ซับซ้อน"""
        
        # ✅ LLM Function wrapper - มี .func attribute
        class LLMFunc:
            def __init__(self, llm_handler):
                self.llm_handler = llm_handler
            
            async def __call__(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
                try:
                    full_prompt = prompt
                    if system_prompt:
                        full_prompt = f"{system_prompt}\n\n{prompt}"
                    
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(
                        None,
                        self.llm_handler.generate,
                        full_prompt
                    )
                    return response
                except Exception as e:
                    logger.error(f"LLM error: {e}")
                    return f"Error: {str(e)}"
        
        llm_func = LLMFunc(self.llm)
        llm_func.func = llm_func  # LightRAG ต้องการ .func
        
        # ✅ Embedding Function wrapper - มี .embedding_dim attribute
        class EmbeddingFunc:
            def __init__(self, memory_handler):
                # Get model
                if hasattr(memory_handler, 'model'):
                    self.model = memory_handler.model
                elif hasattr(memory_handler, 'embedding_model'):
                    self.model = memory_handler.embedding_model
                else:
                    raise AttributeError("No embedding model")
                
                # Get embedding_dim
                try:
                    if hasattr(self.model, 'get_sentence_embedding_dimension'):
                        self.embedding_dim = self.model.get_sentence_embedding_dimension()
                    else:
                        test = self.model.encode(["test"], convert_to_numpy=True)
                        self.embedding_dim = test.shape[1]
                    logger.info(f"📊 Embedding dim: {self.embedding_dim}")
                except Exception as e:
                    logger.warning(f"⚠️ Using default dim=384: {e}")
                    self.embedding_dim = 384
            
            async def __call__(self, texts):
                try:
                    import numpy as np
                    
                    if isinstance(texts, str):
                        texts = [texts]
                    
                    if not texts:
                        return np.zeros((0, self.embedding_dim))
                    
                    loop = asyncio.get_event_loop()
                    embeddings = await loop.run_in_executor(
                        None,
                        lambda: self.model.encode(
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
                    return np.zeros((len(texts), self.embedding_dim))
        
        embedding_func = EmbeddingFunc(self.memory)
        
        try:
            logger.info("🔧 Initializing LightRAG...")
            self.light_rag = LightRAG(
                working_dir=self.working_dir,
                llm_model_func=llm_func,
                embedding_func=embedding_func,
            )
            
            # Initialize storages
            self._init_storages_sync()
            logger.info("✅ LightRAG initialized!")
            
        except Exception as e:
            logger.error(f"❌ LightRAG init failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _init_storages_sync(self):
        """Initialize storages in separate thread"""
        def init_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def init():
                try:
                    await self.light_rag.initialize_storages()
                    await initialize_pipeline_status()
                    logger.info("✅ Storages initialized")
                except Exception as e:
                    logger.debug(f"Storage init: {e}")
            
            loop.run_until_complete(init())
            loop.close()
        
        thread = threading.Thread(target=init_in_thread, daemon=False)
        thread.start()
        thread.join(timeout=10)
    
    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """Query LightRAG"""
        try:
            if not self.light_rag:
                return self._error_result("LightRAG not initialized")
            
            # Check database
            graph_file = Path(self.working_dir) / "graph_chunk_entity_relation.graphml"
            if not graph_file.exists():
                return self._error_result("No data in Knowledge Graph")
            
            logger.info(f"🔍 Querying: {query[:50]}...")
            
            # Query in separate thread
            def query_in_thread():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    query_param = QueryParam(mode=self.query_mode)
                    result = loop.run_until_complete(
                        self.light_rag.aquery(query, param=query_param)
                    )
                    return result
                finally:
                    pending = asyncio.all_tasks(loop)
                    for task in pending:
                        task.cancel()
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                    loop.close()
            
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(query_in_thread)
                result = future.result(timeout=30)
            
            logger.info(f"✅ Query complete: {len(result) if result else 0} chars")
            
            return [{
                'content': result if result else "No information found",
                'score': 1.0,
                'metadata': {
                    'source': 'lightrag_graph',
                    'mode': self.query_mode
                }
            }]
            
        except Exception as e:
            logger.error(f"Query error: {e}")
            import traceback
            traceback.print_exc()
            return self._error_result(str(e))
    
    def _error_result(self, message: str) -> List[Dict]:
        """Error result"""
        return [{
            'content': message,
            'score': 0.0,
            'metadata': {'source': 'lightrag_error'}
        }]
    
    def generate(self, query: str, context: str) -> str:
        """Generate answer"""
        if context and context.strip():
            return context.strip()
        
        # Fallback
        prompt = f"Question: {query}\n\nAnswer in Thai:"
        return self.llm.generate(prompt)
    
    def insert_documents(self, texts: List[str], metadatas=None):
        """Insert to graph"""
        if not texts or not self.light_rag:
            return
        
        logger.info(f"📚 Inserting {len(texts)} docs...")
        combined = "\n\n".join(texts)
        
        def insert_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def do_insert():
                try:
                    await self.light_rag.initialize_storages()
                    await initialize_pipeline_status()
                except:
                    pass
                await self.light_rag.ainsert(combined)
            
            loop.run_until_complete(do_insert())
            loop.close()
            logger.info("✅ Inserted!")
        
        thread = threading.Thread(target=insert_in_thread, daemon=False)
        thread.start()
    
    def set_query_mode(self, mode: str):
        """Change mode"""
        if mode not in ["naive", "local", "global", "hybrid"]:
            raise ValueError(f"Invalid mode: {mode}")
        self.query_mode = mode
    
    def get_query_mode(self) -> str:
        return self.query_mode
    
    def get_graph_stats(self) -> Dict:
        """Get stats"""
        try:
            stats = {"working_dir": self.working_dir, "query_mode": self.query_mode}
            graph_file = Path(self.working_dir) / "graph_chunk_entity_relation.graphml"
            
            if graph_file.exists():
                import networkx as nx
                G = nx.read_graphml(str(graph_file))
                stats["entities"] = G.number_of_nodes()
                stats["relations"] = G.number_of_edges()
            
            return stats
        except Exception as e:
            return {"error": str(e)}
    
    def shutdown(self):
        """Cleanup"""
        pass


def is_lightrag_available() -> bool:
    return HAS_LIGHTRAG