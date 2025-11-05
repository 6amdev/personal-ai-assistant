"""
✅ COMPLETE: FastAPI with LightRAG - Production Ready
Fixed all import paths and initialization issues
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from pathlib import Path
import os
import logging
from datetime import datetime

# ===== Logging Setup =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===== Import Components =====
try:
    from src.llm import LLMHandler
    from src.memory import MemoryHandler
    from src.document_processor import DocumentProcessor
    
    from src.rag.naive_rag import NaiveRAG
    from src.rag.contextual_rag import ContextualRAG
    from src.rag.rerank_rag import RerankRAG
    from src.rag.hybrid_rag import HybridRAG
    from src.rag.query_rewrite_rag import QueryRewriteRAG
    from src.rag.multistep_rag import MultiStepRAG
    
    try:
        from src.rag.light_rag import LightRAGWrapper
        HAS_LIGHTRAG = True
        logger.info("✅ LightRAG imported successfully")
    except ImportError as e:
        HAS_LIGHTRAG = False
        LightRAGWrapper = None
        logger.warning(f"⚠️ LightRAG not available: {e}")
    
    logger.info("✅ All components imported successfully")
    
except ImportError as e:
    logger.error(f"❌ Failed to import components: {e}")
    import traceback
    traceback.print_exc()
    raise

# ===== Configuration =====
try:
    from config import LLM_MODEL
except ImportError:
    logger.warning("⚠️ config.py not found, using defaults")
    LLM_MODEL = "llama3.2:3b"

# API Configuration
API_VERSION = "2.0.0"
API_TITLE = "Personal AI Assistant API with LightRAG"
API_DESCRIPTION = """
🤖 Advanced AI Chat API with Knowledge Graph RAG

✨ Features:
- 7 RAG types (Naive, Contextual, Rerank, Hybrid, Query Rewrite, Multi-step, **LightRAG**)
- **Knowledge Graph** reasoning (LightRAG)
- Document upload & management
- Multi-format support (TXT, PDF, DOCX, JSON, MD)
- Persistent memory (ChromaDB + Graph)
- 100% Local & Private
"""

# Paths Configuration
BASE_DATA_DIR = Path("./data")
CHROMA_DB_PATH = BASE_DATA_DIR / "chroma_db"
LIGHTRAG_DB_PATH = BASE_DATA_DIR / "lightrag_db"
UPLOAD_DIR = BASE_DATA_DIR / "uploads"

# Create directories
for path in [BASE_DATA_DIR, CHROMA_DB_PATH, LIGHTRAG_DB_PATH, UPLOAD_DIR]:
    path.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Directory ready: {path}")

# Security
API_KEY = os.getenv("API_KEY", "your-secret-key-here")
ENABLE_AUTH = os.getenv("ENABLE_AUTH", "false").lower() == "true"

# ===== FastAPI App =====
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# ===== CORS Middleware =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Global State =====
llm = None
memory = None
rag_systems = {}
lightrag_system = None
startup_time = None

# ===== Request/Response Models =====

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    rag_type: str = Field(default="hybrid")
    k: int = Field(default=3, ge=1, le=10)
    include_context: bool = Field(default=False)


class LightRAGQueryRequest(BaseModel):
    query: str = Field(...)
    mode: str = Field(default="hybrid")
    include_context: bool = Field(default=False)


class LightRAGUploadRequest(BaseModel):
    text: str = Field(...)


class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = []
    rag_type: str
    context: Optional[str] = None
    metadata: Dict[str, Any] = {}


class DocumentUploadResponse(BaseModel):
    status: str
    filename: str
    chunks: int
    message: str
    processing_time: Optional[float] = None
    lightrag_processed: bool = False


class HealthResponse(BaseModel):
    status: str
    model: str
    rag_types: List[str]
    documents_count: int
    uptime: Optional[str] = None
    lightrag_available: bool
    lightrag_stats: Optional[Dict] = None


# ===== Authentication =====

async def verify_api_key(x_api_key: str = Header(None)):
    if not ENABLE_AUTH:
        return True
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


# ===== Startup/Shutdown Events =====

@app.on_event("startup")
async def startup_event():
    global llm, memory, rag_systems, lightrag_system, startup_time
    
    startup_time = datetime.now()
    
    logger.info("=" * 60)
    logger.info(f"🚀 Starting {API_TITLE} v{API_VERSION}")
    logger.info("=" * 60)
    
    try:
        # Initialize LLM
        logger.info("🔧 Initializing LLM Handler...")
        llm = LLMHandler(model_name=LLM_MODEL, device="GPU")
        logger.info(f"✅ LLM Ready: {llm.get_model_name()}")
        
        # Initialize Memory - ✅ FIXED: ไม่ส่ง persist_directory
        logger.info("🔧 Initializing Memory Handler...")
        memory = MemoryHandler(device="cuda")
        doc_count = memory.count_documents()
        logger.info(f"✅ Memory Ready: {doc_count} documents")
        
        # Initialize standard RAG systems
        logger.info("🔧 Initializing RAG Systems...")
        rag_systems = {
            "naive": NaiveRAG(llm, memory),
            "contextual": ContextualRAG(llm, memory),
            "rerank": RerankRAG(llm, memory),
            "hybrid": HybridRAG(llm, memory),
            "query_rewrite": QueryRewriteRAG(llm, memory),
            "multistep": MultiStepRAG(llm, memory)
        }
        logger.info(f"✅ Initialized {len(rag_systems)} standard RAG systems")
        
        # Try to initialize LightRAG
        if HAS_LIGHTRAG:
            try:
                logger.info("🔧 Initializing LightRAG...")
                lightrag_system = LightRAGWrapper(llm, memory, query_mode="hybrid")
                lightrag_system.working_dir = str(LIGHTRAG_DB_PATH)
                rag_systems["lightrag"] = lightrag_system
                logger.info(f"✅ LightRAG initialized at {LIGHTRAG_DB_PATH}")
            except Exception as e:
                logger.error(f"⚠️ LightRAG initialization failed: {e}")
                import traceback
                traceback.print_exc()
                lightrag_system = None
        
        logger.info("=" * 60)
        logger.info("✅ All systems ready!")
        logger.info(f"📚 Total RAG types: {len(rag_systems)}")
        logger.info(f"🧠 LightRAG: {'Enabled' if lightrag_system else 'Disabled'}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        import traceback
        traceback.print_exc()
        raise


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Shutting down...")
    if lightrag_system:
        try:
            lightrag_system.shutdown()
        except:
            pass
    logger.info("👋 Goodbye!")


# ===== Health Check =====

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """System health check"""
    try:
        uptime = str(datetime.now() - startup_time) if startup_time else None
        
        lightrag_stats = None
        if lightrag_system:
            try:
                lightrag_stats = lightrag_system.get_graph_stats()
            except:
                pass
        
        return HealthResponse(
            status="healthy",
            model=llm.get_model_name() if llm else "unknown",
            rag_types=list(rag_systems.keys()),
            documents_count=memory.count_documents() if memory else 0,
            uptime=uptime,
            lightrag_available=lightrag_system is not None,
            lightrag_stats=lightrag_stats
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===== Chat Endpoints =====

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest, authenticated: bool = Depends(verify_api_key)):
    """Main chat endpoint - supports all RAG types including LightRAG"""
    try:
        logger.info(f"📨 Chat: '{request.query}' (type: {request.rag_type})")
        
        if request.rag_type not in rag_systems:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid RAG type. Available: {list(rag_systems.keys())}"
            )
        
        rag_system = rag_systems[request.rag_type]
        
        start_time = datetime.now()
        result = rag_system.query(request.query, k=request.k)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        context_parts = []
        sources = []
        
        if isinstance(result.get('context'), list):
            for item in result['context']:
                if isinstance(item, dict):
                    context_parts.append(item.get('content', ''))
                    if 'metadata' in item and 'source' in item['metadata']:
                        sources.append(item['metadata']['source'])
        
        context_text = "\n\n".join(context_parts) if context_parts else ""
        
        response = ChatResponse(
            answer=result.get('answer', 'No answer generated'),
            sources=list(set(sources)),
            rag_type=request.rag_type,
            context=context_text if request.include_context else None,
            metadata={'processing_time': processing_time, 'timestamp': datetime.now().isoformat()}
        )
        
        logger.info(f"✅ Response in {processing_time:.2f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Chat error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ===== LightRAG Endpoints =====

@app.post("/lightrag/query", response_model=ChatResponse, tags=["LightRAG"])
async def lightrag_query(request: LightRAGQueryRequest, authenticated: bool = Depends(verify_api_key)):
    """Query LightRAG Knowledge Graph"""
    if not lightrag_system:
        raise HTTPException(status_code=503, detail="LightRAG not available")
    
    try:
        logger.info(f"🧠 LightRAG: '{request.query}' (mode: {request.mode})")
        
        original_mode = lightrag_system.get_query_mode()
        lightrag_system.set_query_mode(request.mode)
        
        start_time = datetime.now()
        result = lightrag_system.query(request.query, k=3)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        lightrag_system.set_query_mode(original_mode)
        
        context_text = ""
        if isinstance(result.get('context'), list) and result['context']:
            context_text = result['context'][0].get('content', '')
        
        return ChatResponse(
            answer=result.get('answer', 'No answer'),
            sources=['lightrag_knowledge_graph'],
            rag_type="lightrag",
            context=context_text if request.include_context else None,
            metadata={'processing_time': processing_time, 'mode': request.mode}
        )
        
    except Exception as e:
        logger.error(f"❌ LightRAG error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/lightrag/insert", tags=["LightRAG"])
async def lightrag_insert(
    request: LightRAGUploadRequest,
    background_tasks: BackgroundTasks,
    authenticated: bool = Depends(verify_api_key)
):
    """Insert text into LightRAG Knowledge Graph"""
    if not lightrag_system:
        raise HTTPException(status_code=503, detail="LightRAG not available")
    
    try:
        logger.info(f"📚 Inserting: {len(request.text)} chars")
        
        def process():
            try:
                lightrag_system.insert_documents([request.text])
                logger.info("✅ Insert complete")
            except Exception as e:
                logger.error(f"❌ Insert error: {e}")
        
        background_tasks.add_task(process)
        
        return {
            "status": "accepted",
            "message": "Text queued for processing",
            "text_length": len(request.text)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/lightrag/upload", tags=["LightRAG"])
async def lightrag_upload(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    authenticated: bool = Depends(verify_api_key)
):
    """Upload document to LightRAG"""
    if not lightrag_system:
        raise HTTPException(status_code=503, detail="LightRAG not available")
    
    try:
        temp_path = UPLOAD_DIR / file.filename
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        processor = DocumentProcessor()
        documents = processor.process_file(str(temp_path))
        
        if not documents:
            raise HTTPException(status_code=400, detail="No content")
        
        texts = [doc['content'] for doc in documents]
        
        def process():
            try:
                lightrag_system.insert_documents(texts)
                logger.info(f"✅ Processed {len(texts)} chunks")
            except Exception as e:
                logger.error(f"❌ Processing error: {e}")
        
        if background_tasks:
            background_tasks.add_task(process)
        
        if temp_path.exists():
            temp_path.unlink()
        
        return {
            "status": "accepted",
            "filename": file.filename,
            "chunks": len(texts),
            "message": f"Queued {len(texts)} chunks"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/lightrag/stats", tags=["LightRAG"])
async def lightrag_stats():
    """Get LightRAG statistics"""
    if not lightrag_system:
        raise HTTPException(status_code=503, detail="LightRAG not available")
    
    try:
        stats = lightrag_system.get_graph_stats()
        return {"status": "success", "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/lightrag/mode", tags=["LightRAG"])
async def set_lightrag_mode(mode: str, authenticated: bool = Depends(verify_api_key)):
    """Change LightRAG query mode"""
    if not lightrag_system:
        raise HTTPException(status_code=503, detail="LightRAG not available")
    
    try:
        lightrag_system.set_query_mode(mode)
        return {"status": "success", "mode": mode}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ===== Document Management =====

@app.post("/upload", response_model=DocumentUploadResponse, tags=["Documents"])
async def upload_document(
    file: UploadFile = File(...),
    add_to_lightrag: bool = False,
    background_tasks: BackgroundTasks = None,
    authenticated: bool = Depends(verify_api_key)
):
    """Upload document to standard RAG"""
    try:
        temp_path = UPLOAD_DIR / file.filename
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        start_time = datetime.now()
        processor = DocumentProcessor()
        documents = processor.process_file(str(temp_path))
        
        if not documents:
            raise HTTPException(status_code=400, detail="No content")
        
        memory.add_documents(documents)
        
        lightrag_processed = False
        if add_to_lightrag and lightrag_system:
            texts = [doc['content'] for doc in documents]
            
            def add_to_graph():
                try:
                    lightrag_system.insert_documents(texts)
                except Exception as e:
                    logger.error(f"⚠️ LightRAG add error: {e}")
            
            if background_tasks:
                background_tasks.add_task(add_to_graph)
                lightrag_processed = True
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        if temp_path.exists():
            temp_path.unlink()
        
        return DocumentUploadResponse(
            status="success",
            filename=file.filename,
            chunks=len(documents),
            message=f"Processed {len(documents)} chunks",
            processing_time=processing_time,
            lightrag_processed=lightrag_processed
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/info", tags=["System"])
async def system_info():
    """Get system information"""
    return {
        "api": {"version": API_VERSION, "title": API_TITLE},
        "paths": {
            "chroma_db": str(CHROMA_DB_PATH),
            "lightrag_db": str(LIGHTRAG_DB_PATH),
            "uploads": str(UPLOAD_DIR)
        },
        "rag": {
            "available_types": list(rag_systems.keys()),
            "lightrag_enabled": lightrag_system is not None
        },
        "documents": {"count": memory.count_documents() if memory else 0}
    }


# ===== Main =====

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print(f"🚀 Starting {API_TITLE}")
    print("=" * 60)
    print(f"📖 Docs: http://localhost:8000/docs")
    print(f"🔍 Health: http://localhost:8000/health")
    print(f"🧠 LightRAG: {'Enabled' if HAS_LIGHTRAG else 'Disabled'}")
    print("=" * 60)
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )