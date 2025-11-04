"""
FastAPI Wrapper for n8n Integration (With LightRAG Support)

Complete API server for Personal AI Assistant with:
- 7 RAG types (including LightRAG)
- Document upload/management
- Health checks
- Error handling
- Optional authentication
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import tempfile
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
    from src.utils import ImageHandler
    
    # Import all RAG types
    from src.rag import (
        NaiveRAG,
        ContextualRAG,
        RerankRAG,
        HybridRAG,
        QueryRewriteRAG,
        MultiStepRAG,
        is_lightrag_available
    )
    
    # Try to import LightRAG
    try:
        from src.rag import LightRAGWrapper
        HAS_LIGHTRAG = True
    except ImportError:
        HAS_LIGHTRAG = False
        LightRAGWrapper = None
        logger.warning("LightRAG not available (optional)")
    
    logger.info("✅ All components imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import components: {e}")
    raise

# ===== Configuration =====
from config import LLM_MODEL

# API Configuration
API_VERSION = "1.0.0"
API_TITLE = "Personal AI Assistant API"
API_DESCRIPTION = """
🤖 AI Chat API with Advanced RAG Support

Features:
- 7 RAG types (Naive, Contextual, Rerank, Hybrid, Query Rewrite, Multi-step, LightRAG)
- Document upload & management
- Multi-format support (TXT, PDF, DOCX, JSON, MD)
- Image extraction
- Persistent memory (ChromaDB)
- 100% Local & Private
"""

# Security (change in production!)
API_KEY = os.getenv("API_KEY", "your-secret-key-here")
ENABLE_AUTH = os.getenv("ENABLE_AUTH", "false").lower() == "true"

# ===== FastAPI App =====
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# ===== CORS Middleware =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production: specify domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Global State =====
llm = None
memory = None
rag_systems = {}
startup_time = None

# ===== Request/Response Models =====

class ChatRequest(BaseModel):
    """Chat request model"""
    query: str = Field(..., description="คำถามที่ต้องการถาม", min_length=1)
    rag_type: str = Field(
        default="hybrid",
        description="ประเภท RAG: naive, contextual, rerank, hybrid, query_rewrite, multistep, lightrag"
    )
    k: int = Field(default=3, ge=1, le=10, description="จำนวนเอกสารที่ต้องการค้นหา")
    include_images: bool = Field(default=True, description="รวมรูปภาพในผลลัพธ์")
    include_context: bool = Field(default=False, description="รวม context ที่ใช้ในการตอบ")
    session_id: Optional[str] = Field(default=None, description="Session ID (สำหรับอนาคต)")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "RAG คืออะไร?",
                "rag_type": "hybrid",
                "k": 3,
                "include_images": True,
                "include_context": False
            }
        }


class ChatResponse(BaseModel):
    """Chat response model"""
    answer: str = Field(..., description="คำตอบจาก AI")
    sources: List[str] = Field(default=[], description="แหล่งที่มาของข้อมูล")
    rag_type: str = Field(..., description="ประเภท RAG ที่ใช้")
    images: List[Dict[str, Any]] = Field(default=[], description="รูปภาพที่เกี่ยวข้อง")
    context: Optional[str] = Field(default=None, description="Context ที่ใช้ (ถ้าเปิด)")
    metadata: Dict[str, Any] = Field(default={}, description="ข้อมูลเพิ่มเติม")


class DocumentUploadResponse(BaseModel):
    """Document upload response"""
    status: str = Field(..., description="สถานะ: success/error")
    filename: str = Field(..., description="ชื่อไฟล์")
    chunks: int = Field(..., description="จำนวน chunks ที่สร้างได้")
    message: str = Field(..., description="ข้อความ")
    processing_time: Optional[float] = Field(default=None, description="เวลาในการประมวลผล (วินาที)")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="สถานะ: healthy/unhealthy")
    model: str = Field(..., description="LLM model ที่กำลังใช้")
    rag_types: List[str] = Field(..., description="RAG types ที่พร้อมใช้งาน")
    documents_count: int = Field(..., description="จำนวนเอกสารใน memory")
    uptime: Optional[str] = Field(default=None, description="เวลาที่ server ทำงาน")
    lightrag_available: bool = Field(..., description="LightRAG พร้อมใช้งานหรือไม่")


class ErrorResponse(BaseModel):
    """Error response"""
    error: str = Field(..., description="ประเภทของ error")
    message: str = Field(..., description="รายละเอียด error")
    detail: Optional[str] = Field(default=None, description="ข้อมูลเพิ่มเติม")


# ===== Authentication =====

async def verify_api_key(x_api_key: str = Header(None)):
    """Verify API key for protected endpoints"""
    if not ENABLE_AUTH:
        return True
    
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Use X-API-Key header."
        )
    return True


# ===== Startup/Shutdown Events =====

@app.on_event("startup")
async def startup_event():
    """Initialize AI components on startup"""
    global llm, memory, rag_systems, startup_time
    
    startup_time = datetime.now()
    
    logger.info("=" * 60)
    logger.info(f"🚀 Starting {API_TITLE} v{API_VERSION}")
    logger.info("=" * 60)
    
    try:
        # Initialize LLM
        logger.info("📦 Initializing LLM Handler...")
        llm = LLMHandler(model_name=LLM_MODEL, device="GPU")
        logger.info(f"✅ LLM Ready: {llm.get_model_name()}")
        
        # Initialize Memory
        logger.info("📦 Initializing Memory Handler...")
        memory = MemoryHandler(device="cuda")
        doc_count = memory.count_documents()
        logger.info(f"✅ Memory Ready: {doc_count} documents")
        
        # Initialize RAG systems
        logger.info("📦 Initializing RAG Systems...")
        rag_systems = {
            "naive": NaiveRAG(llm, memory),
            "contextual": ContextualRAG(llm, memory),
            "rerank": RerankRAG(llm, memory),
            "hybrid": HybridRAG(llm, memory),
            "query_rewrite": QueryRewriteRAG(llm, memory),
            "multistep": MultiStepRAG(llm, memory)
        }
        logger.info(f"✅ Initialized {len(rag_systems)} RAG systems")
        
        # Try to initialize LightRAG
        if is_lightrag_available() and HAS_LIGHTRAG:
            try:
                logger.info("🌟 Initializing LightRAG...")
                rag_systems["lightrag"] = LightRAGWrapper(llm, memory)
                logger.info("✅ LightRAG ready!")
            except Exception as e:
                logger.warning(f"⚠️ LightRAG init failed: {e}")
                logger.info("📝 LightRAG is optional, continuing without it")
        else:
            logger.info("⚠️ LightRAG not installed (optional)")
        
        logger.info("=" * 60)
        logger.info("🎉 API is ready to serve!")
        logger.info(f"📖 Documentation: http://localhost:8000/docs")
        logger.info(f"🔧 RAG Systems: {list(rag_systems.keys())}")
        logger.info(f"🔐 Authentication: {'Enabled' if ENABLE_AUTH else 'Disabled'}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("=" * 60)
    logger.info("👋 Shutting down API server...")
    logger.info("=" * 60)


# ===== Exception Handlers =====

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested endpoint does not exist",
            "path": str(request.url),
            "available_endpoints": [
                "/docs", "/health", "/chat", "/documents", "/models"
            ]
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors"""
    logger.error(f"Internal error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "detail": str(exc) if not ENABLE_AUTH else "Error details hidden"
        }
    )


# ===== Endpoints =====

@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint - API information
    
    Returns basic information about the API
    """
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "documents": {
                "list": "/documents",
                "upload": "/documents/upload",
                "delete": "/documents/{source}"
            },
            "models": "/models",
            "documentation": "/docs"
        },
        "features": {
            "rag_types": list(rag_systems.keys()),
            "authentication": ENABLE_AUTH,
            "lightrag": is_lightrag_available()
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint
    
    Returns system status, model info, and statistics
    """
    uptime = None
    if startup_time:
        delta = datetime.now() - startup_time
        hours, remainder = divmod(delta.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime = f"{delta.days}d {hours}h {minutes}m {seconds}s"
    
    return HealthResponse(
        status="healthy",
        model=llm.get_model_name() if llm else "Not initialized",
        rag_types=list(rag_systems.keys()),
        documents_count=memory.count_documents() if memory else 0,
        uptime=uptime,
        lightrag_available=is_lightrag_available() and "lightrag" in rag_systems
    )


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Chat"],
    dependencies=[Depends(verify_api_key)] if ENABLE_AUTH else []
)
async def chat(request: ChatRequest):
    """
    Main chat endpoint with RAG support
    
    **RAG Types:**
    - `naive`: พื้นฐาน - Vector search (เร็วที่สุด)
    - `contextual`: เน้น context รอบๆ
    - `rerank`: ค้นหาแล้วจัดอันดับใหม่
    - `hybrid`: BM25 + Vector (ดีที่สุดสำหรับภาษาไทย) 🔥
    - `query_rewrite`: เขียนคำถามใหม่หลายแบบ
    - `multistep`: ค้นหาหลายรอบถ้าข้อมูลไม่พอ
    - `lightrag`: Graph-based RAG with Knowledge Graph 🌟
    
    **Example Request:**
    ```bash
    curl -X POST "http://localhost:8000/chat" \\
      -H "Content-Type: application/json" \\
      -d '{
        "query": "RAG คืออะไร?",
        "rag_type": "hybrid",
        "k": 3
      }'
    ```
    
    **With Authentication:**
    ```bash
    curl -X POST "http://localhost:8000/chat" \\
      -H "Content-Type: application/json" \\
      -H "X-API-Key: your-secret-key-here" \\
      -d '{"query": "test"}'
    ```
    """
    # Validate RAG type
    rag_type_lower = request.rag_type.lower()
    if rag_type_lower not in rag_systems:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid rag_type '{request.rag_type}'. Must be one of: {list(rag_systems.keys())}"
        )
    
    try:
        # Get RAG system
        rag = rag_systems[rag_type_lower]
        
        # Query RAG
        logger.info(f"Processing query with {rag_type_lower} RAG: {request.query[:50]}...")
        result = rag.query(request.query, k=request.k)
        
        # Extract images if requested
        images = []
        if request.include_images and result.get('context'):
            try:
                images = ImageHandler.extract_images_from_context(result['context'])
                logger.info(f"Found {len(images)} images")
            except Exception as e:
                logger.warning(f"Image extraction failed: {e}")
        
        # Prepare response
        return ChatResponse(
            answer=result['answer'],
            sources=result.get('sources', []),
            rag_type=result['rag_type'],
            images=images,
            context=result.get('context') if request.include_context else None,
            metadata={
                "documents_searched": request.k,
                "query_length": len(request.query),
                "response_length": len(result['answer']),
                "has_context": bool(result.get('context')),
                "timestamp": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@app.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    tags=["Documents"],
    dependencies=[Depends(verify_api_key)] if ENABLE_AUTH else []
)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document
    
    **Supported formats:** TXT, PDF, DOCX, JSON, MD
    
    **Example:**
    ```bash
    curl -X POST "http://localhost:8000/documents/upload" \\
      -F "file=@document.pdf"
    ```
    
    **With Authentication:**
    ```bash
    curl -X POST "http://localhost:8000/documents/upload" \\
      -H "X-API-Key: your-secret-key-here" \\
      -F "file=@document.pdf"
    ```
    """
    import time
    start_time = time.time()
    
    # Check file extension
    suffix = Path(file.filename).suffix.lower()
    supported = ['.txt', '.pdf', '.docx', '.json', '.md']
    
    if suffix not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {supported}"
        )
    
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Process document
        logger.info(f"Processing document: {file.filename}")
        processor = DocumentProcessor()
        chunks = processor.process_file(tmp_path)
        
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No content extracted from file. File may be empty or corrupted."
            )
        
        # Add to memory
        logger.info(f"Adding {len(chunks)} chunks to memory...")
        metadata = [{'source': file.filename} for _ in chunks]
        memory.add_documents(chunks, metadata)
        
        # If LightRAG is active, add to knowledge graph
        if "lightrag" in rag_systems:
            try:
                logger.info("Adding to LightRAG Knowledge Graph...")
                rag_systems["lightrag"].insert_documents(chunks, metadata)
                logger.info("✅ LightRAG graph updated")
            except Exception as e:
                logger.warning(f"LightRAG update failed: {e}")
        
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Document processed: {file.filename} ({len(chunks)} chunks, {processing_time:.2f}s)")
        
        return DocumentUploadResponse(
            status="success",
            filename=file.filename,
            chunks=len(chunks),
            message=f"Successfully processed and stored {len(chunks)} chunks from {file.filename}",
            processing_time=round(processing_time, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )
        
    finally:
        # Cleanup temp file
        if Path(tmp_path).exists():
            Path(tmp_path).unlink()


@app.get(
    "/documents",
    tags=["Documents"],
    dependencies=[Depends(verify_api_key)] if ENABLE_AUTH else []
)
async def list_documents():
    """
    List all uploaded documents
    
    Returns list of document sources and statistics
    
    **Example:**
    ```bash
    curl http://localhost:8000/documents
    ```
    """
    try:
        sources = memory.get_all_sources()
        total_chunks = memory.count_documents()
        
        # Group by source
        documents_info = []
        for source in sources:
            # Count chunks per source (approximate)
            documents_info.append({
                "filename": source,
                "type": Path(source).suffix[1:].upper() if Path(source).suffix else "UNKNOWN"
            })
        
        return {
            "status": "success",
            "total_documents": len(sources),
            "total_chunks": total_chunks,
            "documents": documents_info,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing documents: {str(e)}"
        )


@app.delete(
    "/documents/{source}",
    tags=["Documents"],
    dependencies=[Depends(verify_api_key)] if ENABLE_AUTH else []
)
async def delete_document(source: str):
    """
    Delete a document by source name
    
    **Parameters:**
    - `source`: ชื่อไฟล์ที่ต้องการลบ (URL encoded)
    
    **Example:**
    ```bash
    curl -X DELETE "http://localhost:8000/documents/myfile.pdf"
    ```
    """
    try:
        logger.info(f"Deleting document: {source}")
        count = memory.delete_by_source(source)
        
        if count == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Document '{source}' not found"
            )
        
        logger.info(f"✅ Deleted {count} chunks from {source}")
        
        return {
            "status": "success",
            "deleted_chunks": count,
            "source": source,
            "message": f"Deleted {count} chunks from {source}",
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting document: {str(e)}"
        )


@app.delete(
    "/documents",
    tags=["Documents"],
    dependencies=[Depends(verify_api_key)] if ENABLE_AUTH else []
)
async def clear_all_documents():
    """
    Clear all documents from memory
    
    ⚠️ **Warning:** This action cannot be undone!
    
    **Example:**
    ```bash
    curl -X DELETE "http://localhost:8000/documents"
    ```
    """
    try:
        logger.warning("⚠️ Clearing all documents...")
        success = memory.clear_all_documents()
        
        if success:
            logger.info("✅ All documents cleared")
            return {
                "status": "success",
                "message": "All documents cleared successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to clear documents"
            )
            
    except Exception as e:
        logger.error(f"Error clearing documents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing documents: {str(e)}"
        )


@app.get("/models", tags=["System"])
async def list_models():
    """
    List available models and RAG types
    
    Returns information about LLM model, RAG systems, and embeddings
    
    **Example:**
    ```bash
    curl http://localhost:8000/models
    ```
    """
    return {
        "llm": {
            "current_model": llm.get_model_name() if llm else None,
            "provider": "Ollama",
            "device": "GPU/CPU"
        },
        "rag_types": {
            "available": list(rag_systems.keys()),
            "total": len(rag_systems),
            "lightrag_enabled": "lightrag" in rag_systems
        },
        "embeddings": {
            "provider": "Sentence Transformers",
            "model": "paraphrase-multilingual-MiniLM-L12-v2",
            "dimensions": 384
        },
        "statistics": {
            "total_documents": memory.count_documents() if memory else 0,
            "available_sources": len(memory.get_all_sources()) if memory else 0
        },
        "timestamp": datetime.now().isoformat()
    }


@app.get("/stats", tags=["System"])
async def get_statistics():
    """
    Get detailed statistics
    
    Returns comprehensive statistics about the system
    """
    sources = memory.get_all_sources() if memory else []
    
    return {
        "system": {
            "status": "healthy",
            "uptime": (datetime.now() - startup_time).total_seconds() if startup_time else 0,
            "version": API_VERSION
        },
        "documents": {
            "total_sources": len(sources),
            "total_chunks": memory.count_documents() if memory else 0,
            "sources": sources
        },
        "rag": {
            "available_types": list(rag_systems.keys()),
            "total_types": len(rag_systems),
            "lightrag_available": "lightrag" in rag_systems
        },
        "model": {
            "name": llm.get_model_name() if llm else None,
            "provider": "Ollama"
        },
        "timestamp": datetime.now().isoformat()
    }


# ===== Main =====

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print(f"🚀 Starting {API_TITLE}")
    print("=" * 60)
    print(f"📖 API Docs: http://localhost:8000/docs")
    print(f"📊 Health Check: http://localhost:8000/health")
    print(f"🔐 Authentication: {'Enabled' if ENABLE_AUTH else 'Disabled'}")
    print("=" * 60)
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )