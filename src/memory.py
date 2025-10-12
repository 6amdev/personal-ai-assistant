"""Memory Handler - จัดการความจำ"""
import sys
from pathlib import Path
from typing import List, Dict
import os

# ปิด ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# เพิ่ม root folder เข้า path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.memory import ConversationBufferMemory

try:
    from config import CHROMA_DB_DIR, COLLECTION_NAME, LLM_MODEL
except ImportError:
    CHROMA_DB_DIR = "./data/chroma_db"
    COLLECTION_NAME = "personal_assistant"
    LLM_MODEL = "llama3.1:8b"


class MemoryHandler:
    def __init__(self):
        """Initialize memory systems"""
        print("💾 Initializing memory...")
        
        # Embeddings - ใช้ nomic-embed-text (เร็วกว่า!)
        ##self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.embeddings = OllamaEmbeddings(model="llama3.1:8b")
        
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
    
    def add_documents(self, texts: List[str], metadatas: List[dict] = None):
        """
        Add documents to long-term memory (with batch processing)
        
        Args:
            texts: List of text chunks
            metadatas: Optional metadata for each chunk
        """
        total = len(texts)
        batch_size = 20  # ประมวลผล 20 chunks ต่อครั้ง
        
        print(f"📚 Adding {total} chunks to memory...")
        
        for i in range(0, total, batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_metas = metadatas[i:i+batch_size] if metadatas else None
            
            # บันทึก batch นี้
            self.vectorstore.add_texts(texts=batch_texts, metadatas=batch_metas)
            
            # แสดงความคืบหน้า
            processed = min(i+batch_size, total)
            print(f"   ✅ {processed}/{total} chunks")
        
        print(f"✅ Successfully added all {total} chunks!")
    
    def search(self, query: str, k: int = 3) -> List[Dict]:
        """
        Search similar documents
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of documents with metadata
        """
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        # Format results
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': score
            })
        
        return formatted_results
    
    def get_context(self, query: str, k: int = 3) -> str:
        """
        Get context for query (formatted for LLM)
        
        Args:
            query: User query
            k: Number of documents to retrieve
            
        Returns:
            Formatted context string
        """
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
        """
        Get list of all unique document sources
        
        Returns:
            List of source filenames
        """
        try:
            collection = self.vectorstore._collection
            results = collection.get()
            
            if not results or 'metadatas' not in results:
                return []
            
            # Extract unique sources
            sources = set()
            for metadata in results['metadatas']:
                if metadata and 'source' in metadata:
                    sources.add(metadata['source'])
            
            return sorted(list(sources))
        except Exception as e:
            print(f"Error getting sources: {e}")
            return []
    
    def delete_by_source(self, source: str) -> int:
        """
        Delete all documents from a specific source
        
        Args:
            source: Source filename to delete
            
        Returns:
            Number of documents deleted
        """
        try:
            collection = self.vectorstore._collection
            
            # Get all documents
            results = collection.get()
            
            if not results or 'ids' not in results:
                return 0
            
            # Find IDs with matching source
            ids_to_delete = []
            for i, metadata in enumerate(results['metadatas']):
                if metadata and metadata.get('source') == source:
                    ids_to_delete.append(results['ids'][i])
            
            # Delete
            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
                print(f"🗑️ Deleted {len(ids_to_delete)} chunks from {source}")
            
            return len(ids_to_delete)
        except Exception as e:
            print(f"Error deleting source: {e}")
            return 0
    
    def clear_all_documents(self) -> bool:
        """
        Clear all documents from vector store
        ใช้ API แทนการลบไฟล์
        
        Returns:
            True if successful
        """
        try:
            collection = self.vectorstore._collection
            
            # Get all IDs
            results = collection.get()
            
            if results and 'ids' in results and results['ids']:
                # Delete all by IDs
                collection.delete(ids=results['ids'])
                print(f"🗑️ Deleted {len(results['ids'])} documents")
                return True
            else:
                print("ℹ️ No documents to delete")
                return True
                
        except Exception as e:
            print(f"Error clearing documents: {e}")
            return False


# ทดสอบ
if __name__ == "__main__":
    print("Testing Memory Handler...")
    memory = MemoryHandler()
    
    print("\n📝 Adding test documents...")
    memory.add_documents([
        "ผมชื่อ John Doe",
        "ผมเป็นนักพัฒนาซอฟต์แวร์",
        "มีประสบการณ์ 5 ปี",
        "เชี่ยวชาญ Python และ AI"
    ], metadatas=[
        {'source': 'profile.txt'},
        {'source': 'profile.txt'},
        {'source': 'profile.txt'},
        {'source': 'profile.txt'}
    ])
    
    print(f"\n📊 Total documents: {memory.count_documents()}")
    
    print("\n🔍 Searching...")
    query = "ประสบการณ์ทำงาน"
    context = memory.get_context(query, k=2)
    print(f"Query: {query}")
    print(f"\nContext:\n{context}")
    
    print("\n📄 All sources:")
    sources = memory.get_all_sources()
    for source in sources:
        print(f"  - {source}")