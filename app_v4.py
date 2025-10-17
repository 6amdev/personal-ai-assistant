"""Personal AI Assistant - Main App with RAG (Quick Fix)"""
import streamlit as st
import tempfile
import os
from pathlib import Path

from src.ui import setup_page, show_header, show_sidebar, chat_interface, clear_chat_history
from src.llm import LLMHandler
from src.memory import MemoryHandler
from src.document_processor import DocumentProcessor
from src.rag import (
    NaiveRAG, 
    ContextualRAG, 
    RerankRAG, 
    HybridRAG, 
    QueryRewriteRAG, 
    MultiStepRAG
)

def main():
    setup_page()
    
    @st.cache_resource
    def init_components():
        """Initialize LLM and Memory"""
        llm = LLMHandler()
        memory = MemoryHandler()
        return llm, memory
    
    try:
        llm, memory = init_components()
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")
        st.info("💡 ตรวจสอบว่า Ollama กำลังทำงานอยู่!")
        st.code("ollama list", language="bash")
        st.stop()
    
    show_header()
    
    # ⚠️ ต้องเรียก sidebar ก่อน เพื่อรับ rag_type
    # Sidebar - รองรับ 5 และ 6 return values
    sidebar_returns = show_sidebar(memory)
    
    if len(sidebar_returns) == 6:  # 🆕 มี rag_type
        clear_chat, clear_memory, uploaded_files, docs_to_delete, debug_mode, rag_type = sidebar_returns
    elif len(sidebar_returns) == 5:
        clear_chat, clear_memory, uploaded_files, docs_to_delete, debug_mode = sidebar_returns
        rag_type = "Naive RAG"  # default
    else:
        clear_chat, clear_memory, uploaded_files, docs_to_delete = sidebar_returns
        st.session_state.debug_mode = False
        rag_type = "Naive RAG"
    
    st.session_state.debug_mode = debug_mode
    
    # 🆕 สร้าง RAG system ตาม type ที่เลือก
    if "current_rag_type" not in st.session_state or st.session_state.current_rag_type != rag_type:
        st.session_state.current_rag_type = rag_type
        
        if rag_type == "Naive RAG":
            st.session_state.rag_system = NaiveRAG(llm, memory)
        elif rag_type == "Contextual RAG":
            st.session_state.rag_system = ContextualRAG(llm, memory)
        elif rag_type == "Rerank RAG":
            st.session_state.rag_system = RerankRAG(llm, memory)
        elif rag_type == "Hybrid RAG":
            st.session_state.rag_system = HybridRAG(llm, memory)
        elif rag_type == "Query Rewrite RAG":
            st.session_state.rag_system = QueryRewriteRAG(llm, memory)
        elif rag_type == "Multi-step RAG":
            st.session_state.rag_system = MultiStepRAG(llm, memory)
        
        st.sidebar.success(f"✅ ใช้ {rag_type}")
    
    rag = st.session_state.rag_system
    
    # Handle clear chat
    if clear_chat:
        clear_chat_history()
        memory.clear_conversation()
    
    # Handle delete specific documents
    if docs_to_delete:
        for doc_source in docs_to_delete:
            deleted_count = memory.delete_by_source(doc_source)
            if deleted_count > 0:
                st.success(f"✅ ลบ {doc_source} แล้ว ({deleted_count} chunks)")
                st.rerun()
    
    # Handle clear all documents
    if clear_memory:
        with st.spinner("🗑️ กำลังลบเอกสารทั้งหมด..."):
            success = memory.clear_all_documents()
            
            if success:
                st.success("✅ ลบเอกสารทั้งหมดแล้ว!")
                st.cache_resource.clear()
                st.rerun()
            else:
                st.error("❌ เกิดข้อผิดพลาดในการลบเอกสาร")
    
    # Handle file uploads
    if uploaded_files:
        process_uploaded_files(uploaded_files, memory)
    
    # 🔥 Main chat - ส่ง rag ไปด้วย!
    chat_interface(llm, memory, rag_system=rag)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        Made with ❤️ using Ollama, LangChain & Streamlit
    </div>
    """, unsafe_allow_html=True)


def process_uploaded_files(uploaded_files, memory):
    """Process and store uploaded files with detailed progress"""
    
    # ป้องกันการประมวลผลซ้ำ 🔥
    file_ids = [f.file_id for f in uploaded_files]
    processed_key = "processed_files"
    
    if processed_key not in st.session_state:
        st.session_state[processed_key] = set()
    
    # เช็คว่าไฟล์เหล่านี้ประมวลผลแล้วหรือยัง
    new_files = []
    for f in uploaded_files:
        if f.file_id not in st.session_state[processed_key]:
            new_files.append(f)
            st.session_state[processed_key].add(f.file_id)
    
    if not new_files:
        return  # ไฟล์ประมวลผลแล้ว
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    detail_text = st.empty()
    
    processor = DocumentProcessor()
    total_files = len(new_files)
    all_chunks = []
    all_metadatas = []
    
    # Phase 1: อ่านไฟล์ทั้งหมด
    for idx, uploaded_file in enumerate(new_files):
        try:
            status_text.text(f"📄 กำลังอ่าน {uploaded_file.name}... ({idx+1}/{total_files})")
            
            # Save temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Process
            status_text.text(f"📝 แยกข้อความจาก {uploaded_file.name}...")
            chunks = processor.process_file(tmp_path)
            
            # รวม chunks
            all_chunks.extend(chunks)
            all_metadatas.extend([{'source': uploaded_file.name} for _ in chunks])
            
            detail_text.info(f"✅ {uploaded_file.name}: {len(chunks)} chunks")
            
            # Cleanup
            os.unlink(tmp_path)
            
            # Update progress
            progress_bar.progress((idx + 1) / total_files * 0.4)
            
        except Exception as e:
            st.error(f"❌ {uploaded_file.name}: {e}")
    
    # Phase 2: บันทึกทีเดียว
    if all_chunks:
        total_chunks = len(all_chunks)
        status_text.text(f"🧠 กำลังบันทึก {total_chunks} chunks...")
        
        try:
            import time
            start_time = time.time()
            
            progress_bar.progress(0.5)
            
            # เรียกครั้งเดียว!
            memory.add_documents(all_chunks, all_metadatas)
            
            elapsed = time.time() - start_time
            
            progress_bar.progress(1.0)
            status_text.empty()
            detail_text.empty()
            
            st.success(f"✅ เรียนรู้เอกสารสำเร็จ! ({total_chunks} chunks จาก {total_files} ไฟล์)")
            st.info(f"⏱️ ใช้เวลา: {elapsed:.1f} วินาที (~{elapsed/total_chunks:.2f}s/chunk)")
            st.balloons()
            
            # ไม่ rerun! แค่รีเฟรช sidebar 🔥
            st.rerun()
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            detail_text.empty()
            st.error(f"❌ Error: {e}")


if __name__ == "__main__":
    main()