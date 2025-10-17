"""Personal AI Assistant - Main App (Final Fix)"""
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

def init_session_state():
    """Initialize all session state variables ONCE"""
    if 'app_initialized' not in st.session_state:
        from config import LLM_MODEL
        import torch
        
        st.session_state.selected_model = LLM_MODEL
        st.session_state.processing_device = "GPU" if torch.cuda.is_available() else "CPU"
        st.session_state.embedding_device = "cuda" if torch.cuda.is_available() else "cpu"
        st.session_state.app_initialized = True
        st.session_state.components_loaded = False
        
        print("🔧 Session state initialized")


def load_components():
    """Load LLM and Memory components"""
    if st.session_state.components_loaded:
        return st.session_state.llm, st.session_state.memory
    
    selected_model = st.session_state.selected_model
    llm_device = st.session_state.processing_device
    embedding_device = st.session_state.embedding_device
    
    print(f"🔄 Initializing...")
    print(f"   Model: {selected_model}")
    print(f"   LLM Device: {llm_device}")
    print(f"   Embedding Device: {embedding_device}")
    
    try:
        llm = LLMHandler(model_name=selected_model, device=llm_device)
        memory = MemoryHandler(device=embedding_device)
        
        st.session_state.llm = llm
        st.session_state.memory = memory
        st.session_state.components_loaded = True
        
        print("✅ Components initialized!")
        return llm, memory
        
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")
        st.info("💡 ตรวจสอบว่า Ollama กำลังทำงานอยู่!")
        st.code("ollama list", language="bash")
        st.stop()


def main():
    setup_page()
    
    # Initialize session state
    init_session_state()
    
    # Load components
    with st.spinner("🔄 Loading AI components..."):
        llm, memory = load_components()
    
    show_header()
    
    # 🔥 Sidebar - รับค่ากลับมา แต่ไม่ทำอะไร
    sidebar_returns = show_sidebar(memory)
    
    # Unpack returns
    if len(sidebar_returns) >= 6:
        if len(sidebar_returns) == 9:
            clear_chat, clear_memory, uploaded_files, docs_to_delete, debug_mode, rag_type, new_model, new_llm_dev, new_emb_dev = sidebar_returns
        elif len(sidebar_returns) == 6:
            clear_chat, clear_memory, uploaded_files, docs_to_delete, debug_mode, rag_type = sidebar_returns
            new_model = st.session_state.selected_model
            new_llm_dev = st.session_state.processing_device
            new_emb_dev = st.session_state.embedding_device
        else:
            st.error(f"⚠️ Unexpected sidebar returns: {len(sidebar_returns)}")
            st.stop()
    else:
        st.error(f"⚠️ Sidebar return {len(sidebar_returns)} values")
        st.stop()
    
    st.session_state.debug_mode = debug_mode
    
    # 🔥 เช็คการเปลี่ยนแปลง settings (แต่ไม่ rerun ทันที)
    settings_changed = False
    change_message = []
    
    if new_model != st.session_state.selected_model:
        st.session_state.selected_model = new_model
        st.session_state.components_loaded = False
        settings_changed = True
        change_message.append(f"Model: {new_model}")
    
    if new_llm_dev != st.session_state.processing_device:
        st.session_state.processing_device = new_llm_dev
        st.session_state.components_loaded = False
        settings_changed = True
        change_message.append(f"LLM Device: {new_llm_dev}")
    
    if new_emb_dev != st.session_state.embedding_device:
        st.session_state.embedding_device = new_emb_dev
        st.session_state.components_loaded = False
        settings_changed = True
        change_message.append(f"Embedding Device: {new_emb_dev}")
    
    # ถ้ามีการเปลี่ยนแปลง แสดงข้อความและรอ user action
    if settings_changed:
        st.sidebar.warning("⚠️ Settings changed:")
        for msg in change_message:
            st.sidebar.info(f"• {msg}")
        
        if st.sidebar.button("🔄 Apply Changes", type="primary"):
            st.rerun()
        else:
            st.sidebar.info("👆 Click to apply")
            # ใช้ค่าเดิมต่อไป
            llm, memory = load_components()
    
    # สร้าง RAG system
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
                st.rerun()
            else:
                st.error("❌ เกิดข้อผิดพลาดในการลบเอกสาร")
    
    # Handle file uploads
    if uploaded_files:
        process_uploaded_files(uploaded_files, memory)
    
    # Main chat
    chat_interface(llm, memory, rag_system=rag)
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #666;'>
        Made with ❤️ using <strong>{st.session_state.selected_model}</strong> ({st.session_state.processing_device}) • Embeddings ({st.session_state.embedding_device.upper()})
    </div>
    """, unsafe_allow_html=True)


def process_uploaded_files(uploaded_files, memory):
    """Process and store uploaded files with detailed progress"""
    
    # ป้องกันการประมวลผลซ้ำ
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
            
            st.rerun()
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            detail_text.empty()
            st.error(f"❌ Error: {e}")


if __name__ == "__main__":
    main()