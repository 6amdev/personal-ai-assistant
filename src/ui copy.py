"""UI Components"""
import streamlit as st
from config import APP_TITLE, APP_DESCRIPTION, LLM_MODEL
from src.utils import ImageHandler


def setup_page():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="💬",
        layout="wide"
    )


def show_header():
    st.title(APP_TITLE)
    st.markdown(APP_DESCRIPTION)
    st.markdown("---")


def show_sidebar(memory_handler):
    """Show sidebar with document management"""
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model info
        st.subheader("🤖 Model")
        st.info(f"Using: {LLM_MODEL}")
        
        # Document stats
        st.subheader("📚 Knowledge Base")
        doc_count = memory_handler.count_documents()
        st.metric("Total Chunks", doc_count)
        
        # Show document list
        sources = memory_handler.get_all_sources()
        docs_to_delete = []
        
        if sources:
            st.write(f"**Documents ({len(sources)}):**")
            
            with st.expander("📄 View Documents", expanded=False):
                for source in sources:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.text(f"📄 {source}")
                    with col2:
                        if st.button("🗑️", key=f"delete_{source}", help="ลบเอกสารนี้"):
                            docs_to_delete.append(source)
        else:
            st.info("ยังไม่มีเอกสาร")
        
        # File uploader
        st.subheader("📤 Upload Documents")
        uploaded_files = st.file_uploader(
            "เลือกไฟล์",
            type=['txt', 'pdf', 'docx', 'md', 'json'],
            accept_multiple_files=True,
            help="รองรับ: TXT, PDF, DOCX, MD, JSON"
        )
        
        # Options
        st.subheader("📝 Options")
        clear_chat = st.button("🗑️ Clear Chat")
        clear_memory = st.button("🗑️ Clear All Documents", 
                                help="ลบเอกสารทั้งหมด",
                                type="secondary")
        
        # About
        st.markdown("---")
        st.markdown("""
        **Personal AI Assistant**
        
        - 🤖 Local LLM (Ollama)
        - 💾 Persistent Memory
        - 📚 RAG Support
        - 🖼️ Image Support
        - 🔒 100% Private
        
        [GitHub](https://github.com/6amdev/personal-ai-assistant)
        """)
        
        return clear_chat, clear_memory, uploaded_files, docs_to_delete


def chat_interface(llm_handler, memory_handler):
    """Main chat interface with RAG and Images"""
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # แสดงรูปภาพถ้ามี
            if "images" in message and message["images"]:
                display_images(message["images"])
    
    # Chat input
    if prompt := st.chat_input("พิมพ์คำถาม..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response with RAG
        with st.chat_message("assistant"):
            with st.spinner("🤔 กำลังคิด..."):
                # Get context from documents
                context = memory_handler.get_context(prompt, k=3)
                
                # ดึงรูปภาพจาก context
                # ดึงรูปภาพจาก context
                images = ImageHandler.extract_images_from_context(context)

                # 🔍 Debug: แสดงข้อมูล
                st.write(f"🔍 Debug - Context length: {len(context) if context else 0} chars")
                st.write(f"🔍 Debug - Found {len(images)} images")
                if images:
                    st.write("🔍 Debug - Images:", images)
                
                # Create full prompt
                if context:
                    full_prompt = f"""{context}

คำถาม: {prompt}

วิธีตอบ:
1. ตอบคำถามโดยอิงจากข้อมูลที่เกี่ยวข้องข้างต้น
2. ถ้ามีรูปภาพที่เกี่ยวข้อง จะแสดงให้ดูด้านล่าง
3. ระบุแหล่งที่มาของข้อมูล
4. ตอบเป็นภาษาไทยที่เข้าใจง่าย

คำตอบ:"""
                else:
                    full_prompt = prompt
                
                # Generate response
                response = llm_handler.generate(full_prompt)
                st.markdown(response)
                
                # แสดงรูปภาพ
                if images:
                    st.markdown("---")
                    st.markdown("**🖼️ รูปภาพที่เกี่ยวข้อง:**")
                    display_images(images)
                
                # Show context used
                if context:
                    with st.expander("📚 ข้อมูลอ้างอิง"):
                        st.text(context)
        
        # Add to history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "images": images if images else []
        })


# def display_images(images: list):
#     """
#     แสดงรูปภาพทุกรูปแบบ (URL, Local, Base64)
    
#     Args:
#         images: List of image dicts [{"type": "url/local/base64", "data": "...", "caption": "..."}]
#     """
#     if not images:
#         return
    
#     # แสดงเป็นกริด (สูงสุด 3 รูปต่อแถว)
#     num_cols = min(3, len(images))
#     cols = st.columns(num_cols)
    
#     for idx, img_info in enumerate(images):
#         col_idx = idx % num_cols
        
#         with cols[col_idx]:
#             try:
#                 img_type = img_info.get("type", "url")
#                 img_data = img_info.get("data", "")
#                 caption = img_info.get("caption", "")
                
#                 if img_type == "url":
#                     # แสดงรูปจาก URL
#                     st.image(img_data, caption=caption, use_container_width=True)
                
#                 elif img_type == "local":
#                     # แสดงรูปจาก local file
#                     from pathlib import Path
#                     if Path(img_data).exists():
#                         st.image(img_data, caption=caption, use_container_width=True)
#                     else:
#                         st.error(f"❌ ไม่พบไฟล์: {img_data}")
                
#                 elif img_type == "base64":
#                     # แสดงรูปจาก base64
#                     image = ImageHandler.base64_to_image(img_data)
#                     if image:
#                         st.image(image, caption=caption, use_container_width=True)
#                     else:
#                         st.error("❌ ไม่สามารถแสดงรูป base64")
                
#             except Exception as e:
#                 st.error(f"❌ Error displaying image: {e}")

# def display_images(images: list):
#     """
#     แสดงรูปภาพทุกรูปแบบ (URL, Local, Base64) - คลิกได้!
    
#     Args:
#         images: List of image dicts [{"type": "url/local/base64", "data": "...", "caption": "..."}]
#     """
#     if not images:
#         return
    
#     # แสดงเป็นกริด (สูงสุด 3 รูปต่อแถว)
#     num_cols = min(3, len(images))
#     cols = st.columns(num_cols)
    
#     for idx, img_info in enumerate(images):
#         col_idx = idx % num_cols
        
#         with cols[col_idx]:
#             try:
#                 img_type = img_info.get("type", "url")
#                 img_data = img_info.get("data", "")
#                 caption = img_info.get("caption", "")
                
#                 if img_type == "url":
#                     # แสดงรูปจาก URL (คลิกเพื่อเปิดใหม่)
#                     st.image(img_data, caption=caption, use_container_width=True)
#                     st.markdown(f"[🔗 เปิดรูปในหน้าใหม่]({img_data})", unsafe_allow_html=True)
                
#                 elif img_type == "local":
#                     # แสดงรูปจาก local file
#                     from pathlib import Path
#                     if Path(img_data).exists():
#                         st.image(img_data, caption=caption, use_container_width=True)
                        
#                         # แปลง local เป็น base64 เพื่อให้คลิกได้
#                         import base64
#                         with open(img_data, "rb") as f:
#                             img_bytes = f.read()
#                             img_b64 = base64.b64encode(img_bytes).decode()
#                             img_ext = Path(img_data).suffix[1:]  # jpg, png, etc.
#                             data_url = f"data:image/{img_ext};base64,{img_b64}"
#                             st.markdown(f"[🔗 เปิดรูปในหน้าใหม่]({data_url})", unsafe_allow_html=True)
#                     else:
#                         st.error(f"❌ ไม่พบไฟล์: {img_data}")
                
#                 elif img_type == "base64":
#                     # แสดงรูปจาก base64
#                     image = ImageHandler.base64_to_image(img_data)
#                     if image:
#                         st.image(image, caption=caption, use_container_width=True)
                        
#                         # สร้าง data URL
#                         data_url = f"data:image/png;base64,{img_data}"
#                         st.markdown(f"[🔗 เปิดรูปในหน้าใหม่]({data_url})", unsafe_allow_html=True)
#                     else:
#                         st.error("❌ ไม่สามารถแสดงรูป base64")
                
#             except Exception as e:
#                 st.error(f"❌ Error displaying image: {e}")

# def display_images(images: list):
#     """
#     แสดงรูปภาพทุกรูปแบบ (URL, Local, Base64) - คลิกเพื่อเปิดใหม่
    
#     Args:
#         images: List of image dicts [{"type": "url/local/base64", "data": "...", "caption": "..."}]
#     """
#     if not images:
#         return
    
#     # แสดงเป็นกริด (สูงสุด 3 รูปต่อแถว)
#     num_cols = min(3, len(images))
#     cols = st.columns(num_cols)
    
#     for idx, img_info in enumerate(images):
#         col_idx = idx % num_cols
        
#         with cols[col_idx]:
#             try:
#                 img_type = img_info.get("type", "url")
#                 img_data = img_info.get("data", "")
#                 caption = img_info.get("caption", "")
                
#                 if img_type == "url":
#                     # แสดงรูปจาก URL พร้อม clickable overlay
#                     st.markdown(f"""
#                     <a href="{img_data}" target="_blank" style="text-decoration: none;">
#                         <img src="{img_data}" 
#                              style="width: 100%; border-radius: 8px; cursor: pointer; 
#                                     transition: transform 0.2s;"
#                              onmouseover="this.style.transform='scale(1.05)'"
#                              onmouseout="this.style.transform='scale(1)'">
#                     </a>
#                     <p style="text-align: center; color: #666; font-size: 14px; margin-top: 5px;">
#                         {caption} <a href="{img_data}" target="_blank">🔗</a>
#                     </p>
#                     """, unsafe_allow_html=True)
                
#                 elif img_type == "local":
#                     # แสดงรูปจาก local file
#                     from pathlib import Path
#                     if Path(img_data).exists():
#                         import base64
#                         with open(img_data, "rb") as f:
#                             img_bytes = f.read()
#                             img_b64 = base64.b64encode(img_bytes).decode()
#                             img_ext = Path(img_data).suffix[1:]
#                             data_url = f"data:image/{img_ext};base64,{img_b64}"
                        
#                         st.markdown(f"""
#                         <a href="{data_url}" target="_blank" style="text-decoration: none;">
#                             <img src="{data_url}" 
#                                  style="width: 100%; border-radius: 8px; cursor: pointer;
#                                         transition: transform 0.2s;"
#                                  onmouseover="this.style.transform='scale(1.05)'"
#                                  onmouseout="this.style.transform='scale(1)'">
#                         </a>
#                         <p style="text-align: center; color: #666; font-size: 14px; margin-top: 5px;">
#                             {caption} <a href="{data_url}" target="_blank">🔗</a>
#                         </p>
#                         """, unsafe_allow_html=True)
#                     else:
#                         st.error(f"❌ ไม่พบไฟล์: {img_data}")
                
#                 elif img_type == "base64":
#                     # แสดงรูปจาก base64
#                     data_url = f"data:image/png;base64,{img_data}"
                    
#                     st.markdown(f"""
#                     <a href="{data_url}" target="_blank" style="text-decoration: none;">
#                         <img src="{data_url}" 
#                              style="width: 100%; border-radius: 8px; cursor: pointer;
#                                     transition: transform 0.2s;"
#                              onmouseover="this.style.transform='scale(1.05)'"
#                              onmouseout="this.style.transform='scale(1)'">
#                     </a>
#                     <p style="text-align: center; color: #666; font-size: 14px; margin-top: 5px;">
#                         {caption} <a href="{data_url}" target="_blank">🔗</a>
#                     </p>
#                     """, unsafe_allow_html=True)
                
#             except Exception as e:
#                 st.error(f"❌ Error displaying image: {e}")

def display_images(images: list):
    """
    แสดงรูปภาพแบบคลิกขยายในหน้าเดียวกัน (Lightbox)
    
    Args:
        images: List of image dicts [{"type": "url/local/base64", "data": "...", "caption": "..."}]
    """
    if not images:
        return
    
    # เพิ่ม CSS และ JavaScript สำหรับ Lightbox
    st.markdown("""
    <style>
    /* Lightbox styles */
    .lightbox {
        display: none;
        position: fixed;
        z-index: 999;
        padding-top: 50px;
        left: 0;
        top: 0;
        width: 100%;
        height: 100%;
        overflow: auto;
        background-color: rgba(0,0,0,0.9);
    }
    
    .lightbox-content {
        margin: auto;
        display: block;
        max-width: 90%;
        max-height: 90%;
    }
    
    .lightbox-close {
        position: absolute;
        top: 15px;
        right: 35px;
        color: #f1f1f1;
        font-size: 40px;
        font-weight: bold;
        cursor: pointer;
    }
    
    .lightbox-close:hover,
    .lightbox-close:focus {
        color: #bbb;
    }
    
    .lightbox-caption {
        margin: auto;
        display: block;
        width: 80%;
        max-width: 700px;
        text-align: center;
        color: #ccc;
        padding: 10px 0;
        height: 150px;
    }
    
    /* Thumbnail styles */
    .thumbnail {
        cursor: pointer;
        transition: transform 0.2s;
        border-radius: 8px;
    }
    
    .thumbnail:hover {
        transform: scale(1.05);
    }
    </style>
    
    <script>
    function openLightbox(imgSrc, caption) {
        document.getElementById('lightbox').style.display = 'block';
        document.getElementById('lightbox-img').src = imgSrc;
        document.getElementById('lightbox-caption').innerHTML = caption;
    }
    
    function closeLightbox() {
        document.getElementById('lightbox').style.display = 'none';
    }
    
    // ปิดเมื่อคลิกนอกรูป
    window.onclick = function(event) {
        var lightbox = document.getElementById('lightbox');
        if (event.target == lightbox) {
            closeLightbox();
        }
    }
    
    // ปิดด้วย ESC key
    document.addEventListener('keydown', function(event) {
        if (event.key === 'Escape') {
            closeLightbox();
        }
    });
    </script>
    """, unsafe_allow_html=True)
    
    # Lightbox container
    st.markdown("""
    <div id="lightbox" class="lightbox">
        <span class="lightbox-close" onclick="closeLightbox()">&times;</span>
        <img class="lightbox-content" id="lightbox-img">
        <div class="lightbox-caption" id="lightbox-caption"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # แสดงรูปเป็นกริด
    num_cols = min(3, len(images))
    cols = st.columns(num_cols)
    
    for idx, img_info in enumerate(images):
        col_idx = idx % num_cols
        
        with cols[col_idx]:
            try:
                img_type = img_info.get("type", "url")
                img_data = img_info.get("data", "")
                caption = img_info.get("caption", "")
                
                if img_type == "url":
                    # แสดงรูปจาก URL
                    st.markdown(f"""
                    <div onclick="openLightbox('{img_data}', '{caption}')">
                        <img src="{img_data}" 
                             class="thumbnail"
                             style="width: 100%;">
                        <p style="text-align: center; color: #666; font-size: 14px; margin-top: 5px;">
                            {caption}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                elif img_type == "local":
                    # แสดงรูปจาก local file
                    from pathlib import Path
                    if Path(img_data).exists():
                        import base64
                        with open(img_data, "rb") as f:
                            img_bytes = f.read()
                            img_b64 = base64.b64encode(img_bytes).decode()
                            img_ext = Path(img_data).suffix[1:]
                            data_url = f"data:image/{img_ext};base64,{img_b64}"
                        
                        st.markdown(f"""
                        <div onclick="openLightbox('{data_url}', '{caption}')">
                            <img src="{data_url}" 
                                 class="thumbnail"
                                 style="width: 100%;">
                            <p style="text-align: center; color: #666; font-size: 14px; margin-top: 5px;">
                                {caption}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error(f"❌ ไม่พบไฟล์: {img_data}")
                
                elif img_type == "base64":
                    # แสดงรูปจาก base64
                    data_url = f"data:image/png;base64,{img_data}"
                    
                    st.markdown(f"""
                    <div onclick="openLightbox('{data_url}', '{caption}')">
                        <img src="{data_url}" 
                             class="thumbnail"
                             style="width: 100%;">
                        <p style="text-align: center; color: #666; font-size: 14px; margin-top: 5px;">
                            {caption}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error displaying image: {e}")
                
def clear_chat_history():
    """Clear chat history"""
    st.session_state.messages = []
    st.success("✅ ล้างประวัติการสนทนาแล้ว!")