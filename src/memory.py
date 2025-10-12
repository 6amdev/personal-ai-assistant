"""Memory Handler"""
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.memory import ConversationBufferMemory
from config import CHROMA_DB_DIR, COLLECTION_NAME, LLM_MODEL
from typing import List

class MemoryHandler:
    def __init__(self):
        print("💾 Initializing memory...")
        self.embeddings = OllamaEmbeddings(model=LLM_MODEL)
        self.vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=CHROMA_DB_DIR
        )
        self.conversation_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        print("✅ Memory ready!")
    
    def add_documents(self, texts: List[str], metadatas: List[dict] = None):
        self.vectorstore.add_texts(texts=texts, metadatas=metadatas)
        print(f"✅ Added {len(texts)} documents")
    
    def search(self, query: str, k: int = 3) -> List[str]:
        results = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in results]
    
    def clear_conversation(self):
        self.conversation_memory.clear()
        print("🗑️ Cleared conversation")

if __name__ == "__main__":
    print("Testing Memory Handler...")
    memory = MemoryHandler()
    memory.add_documents(["แมวชื่อนำ้ตาล", "อายุ 4 เดือน", "ชอบกินขนม"])
    results = memory.search("แมว")
    print(f"✅ Results: {results}")