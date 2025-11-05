"""
✅ COMPLETE: FastAPI with LightRAG - Production Ready with API Key Security + Multi-Model Support
Fixed all import paths and initialization issues + API Key authentication + Dynamic Model Selection
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends, BackgroundTasks, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from pathlib import Path
import os
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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

# Get default model from environment or config
DEFAULT_MODEL = os.getenv("LLM_MODEL", LLM_MODEL)

# Available models (customize this list based on your Ollama installation)
AVAILABLE_MODELS = os.getenv("AVAILABLE_MODELS", "llama3.2:3b,llama3.1:8b,llama3:8b,mistral:7b,gemma2:9b,phi3:14b").split(",")

# API Configuration
API_VERSION = "2.2.0"
API_TITLE = "Personal AI Assistant API with LightRAG (Secured + Multi-Model)"
API_DESCRIPTION = """
🤖 Advanced AI Chat API with Knowledge Graph RAG - **Secured with API Key + Multi-Model Support**

🔒 **Security:**
- API Key authentication required
- Configure via environment variables

🎯 **Multi-Model Support:**
- Choose from multiple LLM models
- Dynamic model switching per request
- Available models: """ + ", ".join(AVAILABLE_MODELS) + """

✨ Features:
- 7 RAG types (Naive, Contextual, Rerank, Hybrid, Query Rewrite, Multi-step, **LightRAG**)
- **Knowledge Graph** reasoning (LightRAG)
- Document upload & management
- Multi-format support (TXT, PDF, DOCX, JSON, MD)
- Persistent memory (ChromaDB + Graph)
- 100% Local & Private

🔑 **Authentication:**
Add header: `X-API-Key: your-api-key-here`

🤖 **Model Selection:**
Add to request body: `"model": "llama3.1:8b"`
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

# ===== Security Configuration =====
API_KEY = os.getenv("API_KEY", "")
ENABLE_AUTH = os.getenv("ENABLE_AUTH", "false").lower() == "true"

if not API_KEY:
    logger.warning("⚠️ API_KEY not set in environment! Using insecure default.")
    API_KEY = "insecure-default-key-please-change"

if ENABLE_AUTH:
    logger.info(f"🔒 Authentication ENABLED (Key: {API_KEY[:8]}...)")
else:
    logger.warning("⚠️ Authentication DISABLED - API is publicly accessible!")

# API Key Header Security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

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

# Cache for multiple LLM instances
llm_cache = {}  # {model_name: LLMHandler}

# ===== Request/Response Models =====

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    rag_type: str = Field(default="hybrid")
    k: int = Field(default=3, ge=1, le=10)
    include_context: bool = Field(default=False)
    model: Optional[str] = Field(default=None, description="LLM model to use (e.g., 'llama3.1:8b')")


class LightRAGQueryRequest(BaseModel):
    query: str = Field(...)
    mode: str = Field(default="hybrid")
    include_context: bool = Field(default=False)
    model: Optional[str] = Field(default=None, description="LLM model to use")


class LightRAGUploadRequest(BaseModel):
    text: str = Field(...)


class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = []
    rag_type: str
    context: Optional[str] = None
    metadata: Dict[str, Any] = {}
    model_used: str = ""


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
    available_models: List[str]
    rag_types: List[str]
    documents_count: int
    uptime: Optional[str] = None
    lightrag_available: bool
    lightrag_stats: Optional[Dict] = None
    auth_enabled: bool


class ModelListResponse(BaseModel):
    default_model: str
    available_models: List[str]
    cached_models: List[str]


# ===== Helper Functions =====

def get_llm_instance(model_name: Optional[str] = None) -> LLMHandler:
    """
    Get or create LLM instance for specified model
    Uses caching to avoid re-initialization
    """
    global llm, llm_cache
    
    # Use default if no model specified
    if not model_name:
        return llm
    
    # Check if model is available
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' not available. Available models: {', '.join(AVAILABLE_MODELS)}"
        )
    
    # Return cached instance if exists
    if model_name in llm_cache:
        logger.info(f"♻️ Using cached LLM: {model_name}")
        return llm_cache[model_name]
    
    # Create new instance
    try:
        logger.info(f"🔧 Creating new LLM instance: {model_name}")
        new_llm = LLMHandler(model_name=model_name, device="GPU")
        llm_cache[model_name] = new_llm
        logger.info(f"✅ LLM instance created: {model_name}")
        return new_llm
    except Exception as e:
        logger.error(f"❌ Failed to create LLM instance for {model_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize model '{model_name}': {str(e)}"
        )


def create_rag_with_model(llm_instance: LLMHandler, rag_type: str):
    """Create RAG system with specified LLM instance"""
    rag_classes = {
        "naive": NaiveRAG,
        "contextual": ContextualRAG,
        "rerank": RerankRAG,
        "hybrid": HybridRAG,
        "query_rewrite": QueryRewriteRAG,
        "multistep": MultiStepRAG
    }
    
    if rag_type not in rag_classes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid RAG type. Available: {', '.join(rag_classes.keys())}"
        )
    
    return rag_classes[rag_type](llm_instance, memory)


# ===== Authentication Functions =====

async def verify_api_key(api_key: str = Security(api_key_header)):
    """
    Verify API Key from header
    Returns True if valid or authentication is disabled
    Raises HTTPException if invalid
    """
    if not ENABLE_AUTH:
        return True
    
    if api_key is None:
        logger.warning("🚫 Missing API key in request")
        raise HTTPException(
            status_code=401, 
            detail="API key required. Add header: X-API-Key",
            headers={"WWW-Authenticate": "ApiKey"}
        )
    
    if api_key != API_KEY:
        logger.warning(f"🚫 Invalid API key attempt: {api_key[:8]}...")
        raise HTTPException(
            status_code=403, 
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"}
        )
    
    logger.debug("✅ API key validated")
    return True


# ===== Startup/Shutdown Events =====

@app.on_event("startup")
async def startup_event():
    global llm, memory, rag_systems, lightrag_system, startup_time, llm_cache
    
    startup_time = datetime.now()
    
    logger.info("=" * 60)
    logger.info(f"🚀 Starting {API_TITLE} v{API_VERSION}")
    logger.info("=" * 60)
    
    try:
        # Initialize default LLM
        logger.info(f"🔧 Initializing default LLM: {DEFAULT_MODEL}...")
        llm = LLMHandler(model_name=DEFAULT_MODEL, device="GPU")
        llm_cache[DEFAULT_MODEL] = llm
        logger.info(f"✅ Default LLM Ready: {llm.get_model_name()}")
        
        # Initialize Memory
        logger.info("🔧 Initializing Memory Handler...")
        memory = MemoryHandler(device="cuda")
        doc_count = memory.count_documents()
        logger.info(f"✅ Memory Ready: {doc_count} documents")
        
        # Initialize standard RAG systems with default LLM
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
        logger.info(f"🤖 Default Model: {DEFAULT_MODEL}")
        logger.info(f"🎯 Available Models: {', '.join(AVAILABLE_MODELS)}")
        logger.info(f"📚 Total RAG types: {len(rag_systems)}")
        logger.info(f"🧠 LightRAG: {'Enabled' if lightrag_system else 'Disabled'}")
        logger.info(f"🔒 Auth: {'Enabled' if ENABLE_AUTH else 'Disabled'}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        import traceback
        traceback.print_exc()
        raise


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Shutting down...")
    
    # Cleanup LLM cache
    global llm_cache
    llm_cache.clear()
    
    if lightrag_system:
        try:
            lightrag_system.shutdown()
        except:
            pass
    
    logger.info("👋 Goodbye!")


# ===== Public Endpoints (No Auth Required) =====

@app.get("/", tags=["System"])
async def root():
    """API root endpoint"""
    return {
        "message": "Personal AI Assistant API with LightRAG",
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health",
        "models": "/models",
        "auth_required": ENABLE_AUTH
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """System health check - No authentication required"""
    try:
        uptime = None
        if startup_time:
            uptime_delta = datetime.now() - startup_time
            hours, remainder = divmod(uptime_delta.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            uptime = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        
        lightrag_stats = None
        if lightrag_system:
            try:
                lightrag_stats = lightrag_system.get_graph_stats()
            except:
                pass
        
        return HealthResponse(
            status="healthy",
            model=llm.get_model_name() if llm else "unknown",
            available_models=AVAILABLE_MODELS,
            rag_types=list(rag_systems.keys()),
            documents_count=memory.count_documents() if memory else 0,
            uptime=uptime,
            lightrag_available=lightrag_system is not None,
            lightrag_stats=lightrag_stats,
            auth_enabled=ENABLE_AUTH
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models", response_model=ModelListResponse, tags=["System"])
async def list_models():
    """List available models - No authentication required"""
    return ModelListResponse(
        default_model=DEFAULT_MODEL,
        available_models=AVAILABLE_MODELS,
        cached_models=list(llm_cache.keys())
    )


# ===== Protected Endpoints (Auth Required) =====

@app.post("/chat", response_model=ChatResponse, tags=["Chat"], dependencies=[Depends(verify_api_key)])
async def chat(request: ChatRequest):
    """
    Chat with AI using specified RAG system and optional model
    🔒 Requires API Key
    🤖 Supports custom model selection
    """
    try:
        logger.info(f"📨 Query: '{request.query}' (RAG: {request.rag_type}, Model: {request.model or 'default'})")
        
        # Get appropriate LLM instance
        llm_instance = get_llm_instance(request.model)
        model_used = llm_instance.get_model_name()
        
        # Create or get RAG system
        if request.model:
            # Create temporary RAG with custom model
            rag = create_rag_with_model(llm_instance, request.rag_type)
        else:
            # Use cached RAG with default model
            if request.rag_type not in rag_systems:
                available = ", ".join(rag_systems.keys())
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid RAG type. Available: {available}"
                )
            rag = rag_systems[request.rag_type]
        
        start_time = datetime.now()
        result = rag.query(request.query, k=request.k)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        sources = []
        context_text = None
        
        if isinstance(result.get('context'), list):
            sources = [ctx.get('metadata', {}).get('source', 'unknown') 
                      for ctx in result['context']]
            if request.include_context:
                context_text = "\n\n---\n\n".join(
                    ctx.get('content', '') for ctx in result['context']
                )
        
        response = ChatResponse(
            answer=result.get('answer', 'No answer available'),
            sources=sources,
            rag_type=request.rag_type,
            context=context_text,
            metadata={
                'processing_time': processing_time,
                'k': request.k
            },
            model_used=model_used
        )
        
        logger.info(f"✅ Response in {processing_time:.2f}s using {model_used}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Chat error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ===== LightRAG Endpoints (Auth Required) =====

@app.post("/lightrag/query", response_model=ChatResponse, tags=["LightRAG"], dependencies=[Depends(verify_api_key)])
async def lightrag_query(request: LightRAGQueryRequest):
    """
    Query LightRAG Knowledge Graph with optional custom model
    🔒 Requires API Key
    🤖 Supports custom model selection
    """
    if not lightrag_system:
        raise HTTPException(status_code=503, detail="LightRAG not available")
    
    try:
        logger.info(f"🧠 LightRAG: '{request.query}' (mode: {request.mode}, model: {request.model or 'default'})")
        
        # Get LLM instance
        llm_instance = get_llm_instance(request.model)
        model_used = llm_instance.get_model_name()
        
        # If custom model, create temporary LightRAG
        if request.model:
            temp_lightrag = LightRAGWrapper(llm_instance, memory, query_mode=request.mode)
            temp_lightrag.working_dir = str(LIGHTRAG_DB_PATH)
            rag_instance = temp_lightrag
        else:
            rag_instance = lightrag_system
            original_mode = rag_instance.get_query_mode()
            rag_instance.set_query_mode(request.mode)
        
        start_time = datetime.now()
        result = rag_instance.query(request.query, k=3)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Restore original mode if using default
        if not request.model:
            rag_instance.set_query_mode(original_mode)
        
        context_text = ""
        if isinstance(result.get('context'), list) and result['context']:
            context_text = result['context'][0].get('content', '')
        
        return ChatResponse(
            answer=result.get('answer', 'No answer'),
            sources=['lightrag_knowledge_graph'],
            rag_type="lightrag",
            context=context_text if request.include_context else None,
            metadata={
                'processing_time': processing_time,
                'mode': request.mode
            },
            model_used=model_used
        )
        
    except Exception as e:
        logger.error(f"❌ LightRAG error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/lightrag/insert", tags=["LightRAG"], dependencies=[Depends(verify_api_key)])
async def lightrag_insert(
    request: LightRAGUploadRequest,
    background_tasks: BackgroundTasks
):
    """
    Insert text into LightRAG Knowledge Graph
    🔒 Requires API Key
    """
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


@app.post("/lightrag/upload", tags=["LightRAG"], dependencies=[Depends(verify_api_key)])
async def lightrag_upload(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload document to LightRAG
    🔒 Requires API Key
    """
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
    """Get LightRAG statistics - No authentication required for stats"""
    if not lightrag_system:
        raise HTTPException(status_code=503, detail="LightRAG not available")
    
    try:
        stats = lightrag_system.get_graph_stats()
        return {"status": "success", "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/lightrag/mode", tags=["LightRAG"], dependencies=[Depends(verify_api_key)])
async def set_lightrag_mode(mode: str):
    """
    Change LightRAG query mode
    🔒 Requires API Key
    """
    if not lightrag_system:
        raise HTTPException(status_code=503, detail="LightRAG not available")
    
    try:
        lightrag_system.set_query_mode(mode)
        return {"status": "success", "mode": mode}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ===== Document Management (Auth Required) =====

@app.post("/upload", response_model=DocumentUploadResponse, tags=["Documents"], dependencies=[Depends(verify_api_key)])
async def upload_document(
    file: UploadFile = File(...),
    add_to_lightrag: bool = False,
    background_tasks: BackgroundTasks = None
):
    """
    Upload document to standard RAG
    🔒 Requires API Key
    """
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
    """Get system information - No authentication required"""
    return {
        "api": {
            "version": API_VERSION,
            "title": API_TITLE
        },
        "security": {
            "auth_enabled": ENABLE_AUTH,
            "auth_method": "API Key (X-API-Key header)"
        },
        "models": {
            "default": DEFAULT_MODEL,
            "available": AVAILABLE_MODELS,
            "cached": list(llm_cache.keys())
        },
        "paths": {
            "chroma_db": str(CHROMA_DB_PATH),
            "lightrag_db": str(LIGHTRAG_DB_PATH),
            "uploads": str(UPLOAD_DIR)
        },
        "rag": {
            "available_types": list(rag_systems.keys()),
            "lightrag_enabled": lightrag_system is not None
        },
        "documents": {
            "count": memory.count_documents() if memory else 0
        }
    }


# ===== Main =====

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print(f"🚀 Starting {API_TITLE}")
    print("=" * 60)
    print(f"📖 Docs: http://localhost:8000/docs")
    print(f"🔍 Health: http://localhost:8000/health")
    print(f"🤖 Models: http://localhost:8000/models")
    print(f"🧠 LightRAG: {'Enabled' if HAS_LIGHTRAG else 'Disabled'}")
    print(f"🔒 Auth: {'Enabled' if ENABLE_AUTH else 'Disabled'}")
    if ENABLE_AUTH:
        print(f"🔑 API Key: {API_KEY[:8]}...")
    print(f"🎯 Default Model: {DEFAULT_MODEL}")
    print(f"🎯 Available Models: {', '.join(AVAILABLE_MODELS)}")
    print("=" * 60)
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )