"""Memory Handler - Alternative (No HuggingFace Hub Issues)"""
import sys
from pathlib import Path
from typing import List, Dict
import os
import warnings

warnings.filterwarnings('ignore')

# ปิด ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory

try:
    from config import CHROMA_DB_DIR, COLLECTION_NAME
except ImportError:
    CHROMA_DB_DIR = "./data/chroma_db"
    COLLECTION_NAME = "personal_assistant"


class MemoryHandler:
    def __init__(self, device: str = "cuda"):
        """
        Initialize memory systems
        
        Args:
            device: "cuda", "cpu", "GPU", หรือ "CPU"
        """
        # แปลง GPU/CPU → cuda/cpu
        if device.upper() == "GPU":
            device = "cuda"
        elif device.upper() == "CPU":
            device = "cpu"
        
        device = device.lower()
        
        print(f"💾 Initializing memory with device: {device.upper()}")
        print("📦 Loading embedding model...")
        
        # 🔥 วิธีใหม่: ใช้ sentence-transformers โดยตรง (ไม่ผ่าน LangChain)
        try:
            from sentence_transformers import SentenceTransformer
            
            # โหลด model โดยตรง
            self.model = SentenceTransformer(
                'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                device=device
            )
            
            # สร้าง custom embeddings class
            self.embeddings = self._create_embeddings()
            
            print(f"✅ Embedding model ready on {device.upper()}!")
            
        except Exception as e:
            print(f"❌ Error loading embedding model: {e}")
            print("🔄 Trying fallback to Ollama embeddings...")
            
            # Fallback: ใช้ Ollama embeddings
            try:
                from langchain_community.embeddings import OllamaEmbeddings
                from config import LLM_MODEL
                
                self.embeddings = OllamaEmbeddings(model=LLM_MODEL)
                print(f"✅ Using Ollama embeddings: {LLM_MODEL}")
                
            except Exception as e2:
                print(f"❌ Fallback also failed: {e2}")
                raise Exception("Cannot initialize embeddings. Please check your setup.")
        
        # Long-term memory (ChromaDB)
        self.vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=CHROMA_DB_DIR
        )
        
        # Short-term memory (Conversation)
        self.conversation_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        print("✅ Memory ready!")
    
    def _create_embeddings(self):
        """สร้าง custom embeddings class"""
        model = self.model
        
        class CustomEmbeddings:
            def __init__(self, model):
                self.model = model
            
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                """Embed multiple documents"""
                embeddings = self.model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                return embeddings.tolist()
            
            def embed_query(self, text: str) -> List[float]:
                """Embed single query"""
                embedding = self.model.encode(
                    text,
                    normalize_embeddings=True,
                    convert_to_numpy=True
                )
                return embedding.tolist()
        
        return CustomEmbeddings(model)
    
    def add_documents(self, texts: List[str], metadatas: List[dict] = None):
        """Add documents to long-term memory (with batch processing)"""
        total = len(texts)
        batch_size = 20
        
        print(f"📚 Adding {total} chunks to memory...")
        
        for i in range(0, total, batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_metas = metadatas[i:i+batch_size] if metadatas else None
            
            self.vectorstore.add_texts(texts=batch_texts, metadatas=batch_metas)
            
            processed = min(i+batch_size, total)
            print(f"   ✅ {processed}/{total} chunks")
        
        print(f"✅ Successfully added all {total} chunks!")
    
    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search similar documents"""
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': score
            })
        
        return formatted_results
    
    def get_context(self, query: str, k: int = 3) -> str:
        """Get context for query (formatted for LLM)"""
        results = self.search(query, k=k)
        
        if not results:
            return ""
        
        context = "ข้อมูลที่เกี่ยวข้อง:\n\n"
        for i, result in enumerate(results, 1):
            context += f"{i}. {result['content']}\n"
            if 'source' in result['metadata']:
                context += f"   (แหล่งที่มา: {result['metadata']['source']})\n"
            context += "\n"
        
        return context
    
    def clear_conversation(self):
        """Clear conversation memory"""
        self.conversation_memory.clear()
        print("🗑️ Cleared conversation")
    
    def count_documents(self) -> int:
        """Count total documents in vector store"""
        try:
            collection = self.vectorstore._collection
            return collection.count()
        except:
            return 0
    
    def get_all_sources(self) -> List[str]:
        """Get list of all unique document sources"""
        try:
            collection = self.vectorstore._collection
            results = collection.get()
            
            if not results or 'metadatas' not in results:
                return []
            
            sources = set()
            for metadata in results['metadatas']:
                if metadata and 'source' in metadata:
                    sources.add(metadata['source'])
            
            return sorted(list(sources))
        except Exception as e:
            print(f"Error getting sources: {e}")
            return []
    
    def delete_by_source(self, source: str) -> int:
        """Delete all documents from a specific source"""
        try:
            collection = self.vectorstore._collection
            results = collection.get()
            
            if not results or 'ids' not in results:
                return 0
            
            ids_to_delete = []
            for i, metadata in enumerate(results['metadatas']):
                if metadata and metadata.get('source') == source:
                    ids_to_delete.append(results['ids'][i])
            
            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
                print(f"🗑️ Deleted {len(ids_to_delete)} chunks from {source}")
            
            return len(ids_to_delete)
        except Exception as e:
            print(f"Error deleting source: {e}")
            return 0
    
    def clear_all_documents(self) -> bool:
        """Clear all documents from vector store"""
        try:
            collection = self.vectorstore._collection
            results = collection.get()
            
            if results and 'ids' in results and results['ids']:
                collection.delete(ids=results['ids'])
                print(f"🗑️ Deleted {len(results['ids'])} documents")
                return True
            else:
                print("ℹ️ No documents to delete")
                return True
                
        except Exception as e:
            print(f"Error clearing documents: {e}")
            return False


if __name__ == "__main__":
    print("Testing Memory Handler...")
    
    # Test CUDA
    print("\n=== Test CUDA ===")
    try:
        mem_cuda = MemoryHandler(device="cuda")
        print("✅ CUDA works!")
    except Exception as e:
        print(f"❌ CUDA failed: {e}")
    
    # Test CPU
    print("\n=== Test CPU ===")
    try:
        mem_cpu = MemoryHandler(device="cpu")
        print("✅ CPU works!")
    except Exception as e:
        print(f"❌ CPU failed: {e}")