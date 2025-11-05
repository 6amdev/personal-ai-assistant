#!/usr/bin/env python3
"""
🧪 API Testing Script with Authentication
Test all endpoints with API Key
"""

import requests
import os
from dotenv import load_dotenv
import json
from pathlib import Path

# Load environment variables
load_dotenv()

# Configuration
API_URL = "http://localhost:8000"
API_KEY = os.getenv("API_KEY", "your-api-key-here")
ENABLE_AUTH = os.getenv("ENABLE_AUTH", "false").lower() == "true"

# Headers
headers = {
    "Content-Type": "application/json"
}

if ENABLE_AUTH:
    headers["X-API-Key"] = API_KEY

def print_section(title):
    print("\n" + "=" * 60)
    print(f"🧪 {title}")
    print("=" * 60)

def print_response(response):
    print(f"Status: {response.status_code}")
    try:
        data = response.json()
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except:
        print(response.text)
    print()

# ===== Test Functions =====

def test_root():
    print_section("Test 1: Root Endpoint (Public)")
    response = requests.get(f"{API_URL}/")
    print_response(response)

def test_health():
    print_section("Test 2: Health Check (Public)")
    response = requests.get(f"{API_URL}/health")
    print_response(response)

def test_info():
    print_section("Test 3: System Info (Public)")
    response = requests.get(f"{API_URL}/info")
    print_response(response)

def test_chat_naive():
    print_section("Test 4: Chat - Naive RAG (Protected)")
    data = {
        "query": "อธิบาย RAG คืออะไร",
        "rag_type": "naive",
        "k": 3,
        "include_context": False
    }
    response = requests.post(f"{API_URL}/chat", headers=headers, json=data)
    print_response(response)

def test_chat_hybrid():
    print_section("Test 5: Chat - Hybrid RAG (Protected)")
    data = {
        "query": "ความแตกต่างระหว่าง RAG แต่ละประเภท",
        "rag_type": "hybrid",
        "k": 5,
        "include_context": True
    }
    response = requests.post(f"{API_URL}/chat", headers=headers, json=data)
    print_response(response)

def test_lightrag_stats():
    print_section("Test 6: LightRAG Stats (Public)")
    response = requests.get(f"{API_URL}/lightrag/stats")
    print_response(response)

def test_lightrag_query():
    print_section("Test 7: LightRAG Query (Protected)")
    data = {
        "query": "สรุปความรู้เกี่ยวกับ AI และ Machine Learning",
        "mode": "hybrid",
        "include_context": True
    }
    response = requests.post(f"{API_URL}/lightrag/query", headers=headers, json=data)
    print_response(response)

def test_lightrag_insert():
    print_section("Test 8: LightRAG Insert Text (Protected)")
    data = {
        "text": "Deep Learning เป็นสาขาหนึ่งของ Machine Learning ที่ใช้ Neural Networks ที่มีหลายชั้น (Deep Neural Networks) เพื่อเรียนรู้ patterns ที่ซับซ้อนจากข้อมูล"
    }
    response = requests.post(f"{API_URL}/lightrag/insert", headers=headers, json=data)
    print_response(response)

def test_invalid_key():
    print_section("Test 9: Invalid API Key (Should Fail)")
    invalid_headers = headers.copy()
    invalid_headers["X-API-Key"] = "invalid-key-123"
    
    data = {
        "query": "test",
        "rag_type": "naive"
    }
    response = requests.post(f"{API_URL}/chat", headers=invalid_headers, json=data)
    print_response(response)

def test_missing_key():
    print_section("Test 10: Missing API Key (Should Fail if auth enabled)")
    no_auth_headers = {"Content-Type": "application/json"}
    
    data = {
        "query": "test",
        "rag_type": "naive"
    }
    response = requests.post(f"{API_URL}/chat", headers=no_auth_headers, json=data)
    print_response(response)

def test_upload_document():
    print_section("Test 11: Upload Document (Protected)")
    
    # สร้างไฟล์ทดสอบ
    test_file = Path("test_document.txt")
    test_file.write_text("""
    Graph RAG คืออะไร?
    
    Graph RAG เป็นเทคนิคใหม่ในการทำ Retrieval-Augmented Generation 
    ที่ใช้ Knowledge Graph เพื่อจัดเก็บและค้นหาข้อมูลอย่างมีประสิทธิภาพ
    
    ข้อดีของ Graph RAG:
    1. เข้าใจความสัมพันธ์ระหว่างข้อมูล
    2. ค้นหาได้แม่นยำขึ้น
    3. สามารถทำ reasoning ได้ดีกว่า
    """, encoding="utf-8")
    
    try:
        with open(test_file, "rb") as f:
            files = {"file": (test_file.name, f, "text/plain")}
            # ต้องเอา Content-Type ออกเวลาส่ง multipart
            upload_headers = {"X-API-Key": API_KEY} if ENABLE_AUTH else {}
            
            response = requests.post(
                f"{API_URL}/upload",
                headers=upload_headers,
                files=files,
                data={"add_to_lightrag": "true"}
            )
            print_response(response)
    finally:
        if test_file.exists():
            test_file.unlink()

# ===== Main =====

def main():
    print("\n" + "🚀" * 30)
    print("API Testing Script with Authentication")
    print("🚀" * 30)
    print(f"\n📍 API URL: {API_URL}")
    print(f"🔒 Auth Enabled: {ENABLE_AUTH}")
    if ENABLE_AUTH:
        print(f"🔑 API Key: {API_KEY[:8]}...")
    print()
    
    try:
        # Public endpoints (no auth required)
        test_root()
        test_health()
        test_info()
        
        # Protected endpoints (auth required if enabled)
        test_chat_naive()
        test_chat_hybrid()
        test_lightrag_stats()
        test_lightrag_query()
        test_lightrag_insert()
        test_upload_document()
        
        # Security tests (only if auth is enabled)
        if ENABLE_AUTH:
            test_invalid_key()
            test_missing_key()
        
        print_section("✅ All Tests Completed!")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Cannot connect to API")
        print("Make sure the API is running: python api.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()