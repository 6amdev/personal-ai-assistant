#!/usr/bin/env python3
"""
Sync Documents from ChromaDB to LightRAG
จะดึงเอกสารจาก ChromaDB แล้วส่งเข้า LightRAG Knowledge Graph
"""

import sys
import requests
from pathlib import Path

# เพิ่ม path
sys.path.append(str(Path(__file__).parent))

from src.memory import MemoryHandler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Configuration
API_BASE = "http://localhost:8000"


def sync_to_lightrag():
    """Sync all documents from ChromaDB to LightRAG via API"""
    
    print("=" * 60)
    print("🔄 Syncing Documents to LightRAG")
    print("=" * 60)
    print("")
    
    # 1. Initialize memory to read existing documents
    print("📦 Loading ChromaDB...")
    memory = MemoryHandler(device="cpu")
    
    # 2. Get all documents from ChromaDB directly
    print("📚 Fetching all documents from ChromaDB...")
    
    try:
        # Access ChromaDB collection directly
        collection = memory.vectorstore._collection
        
        # Get all documents
        results = collection.get(
            include=['documents', 'metadatas']
        )
        
        if not results['documents']:
            print("⚠️ No documents found in ChromaDB!")
            return
        
        print(f"✅ Found {len(results['documents'])} chunks")
        print("")
        
        # 3. Group by source
        from collections import defaultdict
        docs_by_source = defaultdict(list)
        
        for doc, metadata in zip(results['documents'], results['metadatas']):
            source = metadata.get('source', 'unknown') if metadata else 'unknown'
            docs_by_source[source].append(doc)
        
    except Exception as e:
        print(f"❌ Error reading from ChromaDB: {e}")
        print("Trying alternative method...")
        
        # Alternative: Use get_all_sources and search
        sources = memory.get_all_sources()
        if not sources:
            print("⚠️ No sources found!")
            return
        
        from collections import defaultdict
        docs_by_source = defaultdict(list)
        
        # For each source, get documents
        for source in sources:
            try:
                # Search for this source
                results = memory.vectorstore.similarity_search(
                    query="",  # Empty query to get all
                    k=100,  # Get many
                    filter={"source": source}
                )
                
                for result in results:
                    docs_by_source[source].append(result.page_content)
            except:
                continue
        
        if not docs_by_source:
            print("⚠️ Could not fetch documents!")
            return
    
    print(f"📂 Grouped into {len(docs_by_source)} sources")
    print("")
    
    # 4. Check API health
    print("🏥 Checking API health...")
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        if response.status_code != 200:
            print("❌ API is not healthy!")
            return
        
        data = response.json()
        if not data.get('lightrag_available'):
            print("❌ LightRAG not available in API!")
            return
        
        print("✅ API is healthy")
        print(f"   LightRAG: {data['lightrag_available']}")
        print(f"   Current docs: {data['documents_count']}")
        print("")
        
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("   Make sure API is running: python api.py")
        return
    
    # 5. Sync to LightRAG
    print("🔄 Syncing to LightRAG...")
    print("")
    
    for idx, (source, chunks) in enumerate(docs_by_source.items(), 1):
        print(f"[{idx}/{len(docs_by_source)}] Processing: {source}")
        print(f"   Chunks: {len(chunks)}")
        
        # Combine chunks
        combined_text = "\n\n".join(chunks)
        
        # Create temp file
        import tempfile
        suffix = Path(source).suffix or '.txt'
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False, encoding='utf-8') as f:
            f.write(combined_text)
            temp_path = f.name
        
        try:
            # Upload via API
            with open(temp_path, 'rb') as f:
                files = {'file': (source, f)}
                response = requests.post(
                    f"{API_BASE}/documents/upload",
                    files=files,
                    timeout=60
                )
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Uploaded: {data['chunks']} chunks")
            else:
                print(f"   ❌ Upload failed: {response.status_code}")
                print(f"      {response.text}")
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        finally:
            # Cleanup
            Path(temp_path).unlink(missing_ok=True)
        
        print("")
    
    # 6. Final check
    print("=" * 60)
    print("🔍 Final Check")
    print("=" * 60)
    
    try:
        response = requests.get(f"{API_BASE}/health")
        data = response.json()
        
        print(f"✅ Sync Complete!")
        print(f"   Total documents in API: {data['documents_count']}")
        print(f"   LightRAG available: {data['lightrag_available']}")
        
    except Exception as e:
        print(f"❌ Cannot verify: {e}")
    
    print("")
    print("=" * 60)
    print("✅ All Done!")
    print("=" * 60)
    print("")
    print("🧪 Test with Postman:")
    print("   POST http://localhost:8000/chat")
    print("   Body: {\"query\": \"RAG คืออะไร?\", \"rag_type\": \"lightrag\", \"k\": 5}")
    print("")


if __name__ == "__main__":
    try:
        sync_to_lightrag()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()