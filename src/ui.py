"""UI Components - Final Version with Inline Images"""
import streamlit as st
import json
import base64
from pathlib import Path
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
        
        # Debug Mode Toggle
        debug_mode = st.checkbox("🔍 Debug Mode", value=False, help="แสดงข้อมูล debug")
        
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
        
        return clear_chat, clear_memory, uploaded_files, docs_to_delete, debug_mode


def chat_interface(llm_handler, memory_handler):
    """Main chat interface with RAG and Images (Enhanced)"""
    
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
                images = ImageHandler.extract_images_from_context(context) if context else []
                
                # Debug info (แสดงเฉพาะเมื่อเปิด debug mode)
                debug_mode = st.session_state.get("debug_mode", False)
                if debug_mode:
                    with st.expander("🔍 Debug Info", expanded=True):
                        st.write(f"**Context length:** {len(context) if context else 0} chars")
                        st.write(f"**Found images:** {len(images)}")
                        if images:
                            st.json(images)
                        if context:
                            st.text_area("Context Preview", context[:500] + "..." if len(context) > 500 else context, height=200)
                
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
                        st.text(context[:1000] + "..." if len(context) > 1000 else context)
        
        # Add to history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "images": images if images else []
        })


def display_images(images: list):
    """
    แสดงรูปภาพแบบ inline thumbnail พร้อม Lightbox
    🔥 COPY FUNCTION นี้ไปแทนที่ใน src/ui.py
    
    Args:
        images: List of image dicts [{"type": "url/local/base64", "data": "...", "caption": "..."}]
    """
    if not images:
        return
    
    import hashlib
    import time
    import base64
    import json
    from pathlib import Path
    
    # สร้าง unique ID
    lightbox_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
    
    # เตรียมข้อมูลรูปภาพ
    processed_images = []
    for img_info in images:
        img_type = img_info.get("type", "url")
        img_data = img_info.get("data", "")
        caption = img_info.get("caption", "")
        
        try:
            if img_type == "url":
                processed_images.append({
                    "src": img_data,
                    "caption": caption or "รูปภาพ"
                })
            elif img_type == "local":
                if Path(img_data).exists():
                    with open(img_data, "rb") as f:
                        img_bytes = f.read()
                        img_b64 = base64.b64encode(img_bytes).decode()
                        img_ext = Path(img_data).suffix[1:]
                        data_url = f"data:image/{img_ext};base64,{img_b64}"
                        processed_images.append({
                            "src": data_url,
                            "caption": caption or "รูปภาพ"
                        })
            elif img_type == "base64":
                data_url = f"data:image/png;base64,{img_data}"
                processed_images.append({
                    "src": data_url,
                    "caption": caption or "รูปภาพ"
                })
        except Exception as e:
            st.warning(f"⚠️ ไม่สามารถโหลดรูปภาพ: {e}")
    
    if not processed_images:
        st.warning("⚠️ ไม่สามารถแสดงรูปภาพได้")
        return
    
    # แปลงเป็น JSON
    images_json = json.dumps(processed_images)
    
    # CSS และ JavaScript
    st.markdown(f"""
    <style>
    .lightbox-{lightbox_id} {{
        display: none;
        position: fixed;
        z-index: 9999;
        padding-top: 50px;
        left: 0;
        top: 0;
        width: 100%;
        height: 100%;
        overflow: auto;
        background-color: rgba(0,0,0,0.95);
    }}
    
    .lightbox-content-{lightbox_id} {{
        margin: auto;
        display: block;
        max-width: 90%;
        max-height: 85vh;
        object-fit: contain;
        animation: zoom-{lightbox_id} 0.3s;
    }}
    
    @keyframes zoom-{lightbox_id} {{
        from {{transform: scale(0.8); opacity: 0;}}
        to {{transform: scale(1); opacity: 1;}}
    }}
    
    .lightbox-close-{lightbox_id} {{
        position: absolute;
        top: 20px;
        right: 40px;
        color: #f1f1f1;
        font-size: 45px;
        font-weight: bold;
        cursor: pointer;
        transition: 0.3s;
        z-index: 10000;
    }}
    
    .lightbox-close-{lightbox_id}:hover {{
        color: #ff4444;
    }}
    
    .lightbox-caption-{lightbox_id} {{
        margin: auto;
        display: block;
        width: 80%;
        max-width: 700px;
        text-align: center;
        color: #fff;
        padding: 15px 0;
        font-size: 18px;
    }}
    
    .lightbox-arrow-{lightbox_id} {{
        cursor: pointer;
        position: absolute;
        top: 50%;
        width: auto;
        padding: 20px;
        margin-top: -50px;
        color: white;
        font-weight: bold;
        font-size: 35px;
        transition: 0.3s;
        user-select: none;
        background-color: rgba(0, 0, 0, 0.6);
        border-radius: 5px;
        z-index: 10000;
    }}
    
    .lightbox-arrow-{lightbox_id}:hover {{
        background-color: rgba(0, 0, 0, 0.9);
    }}
    
    .arrow-left-{lightbox_id} {{
        left: 25px;
    }}
    
    .arrow-right-{lightbox_id} {{
        right: 25px;
    }}
    
    .lightbox-counter-{lightbox_id} {{
        position: absolute;
        top: 25px;
        left: 50%;
        transform: translateX(-50%);
        color: #f1f1f1;
        font-size: 20px;
        background-color: rgba(0, 0, 0, 0.7);
        padding: 10px 20px;
        border-radius: 5px;
        z-index: 10000;
    }}
    
    .image-grid-{lightbox_id} {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 15px;
        margin: 15px 0;
    }}
    
    .thumbnail-container-{lightbox_id} {{
        position: relative;
        overflow: hidden;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        cursor: pointer;
        background: #f5f5f5;
    }}
    
    .thumbnail-container-{lightbox_id}:hover {{
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }}
    
    .thumbnail-{lightbox_id} {{
        width: 100%;
        height: 200px;
        object-fit: cover;
        display: block;
    }}
    
    .thumbnail-caption-{lightbox_id} {{
        padding: 10px;
        text-align: center;
        color: #333;
        font-size: 14px;
        background: white;
        border-top: 1px solid #eee;
        min-height: 45px;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    
    .thumbnail-overlay-{lightbox_id} {{
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 45px;
        background: rgba(0,0,0,0.5);
        display: flex;
        align-items: center;
        justify-content: center;
        opacity: 0;
        transition: opacity 0.3s;
        color: white;
        font-size: 40px;
    }}
    
    .thumbnail-container-{lightbox_id}:hover .thumbnail-overlay-{lightbox_id} {{
        opacity: 1;
    }}
    </style>
    
    <script>
    (function() {{
        const images_{lightbox_id} = {images_json};
        let currentImageIndex_{lightbox_id} = 0;
        
        window.openLightbox_{lightbox_id} = function(index) {{
            currentImageIndex_{lightbox_id} = index;
            const img = images_{lightbox_id}[index];
            
            const lightbox = document.getElementById('lightbox-{lightbox_id}');
            const lightboxImg = document.getElementById('lightbox-img-{lightbox_id}');
            const caption = document.getElementById('lightbox-caption-{lightbox_id}');
            const counter = document.getElementById('lightbox-counter-{lightbox_id}');
            const arrowLeft = document.getElementById('arrow-left-{lightbox_id}');
            const arrowRight = document.getElementById('arrow-right-{lightbox_id}');
            
            if (lightbox && lightboxImg) {{
                lightbox.style.display = 'block';
                lightboxImg.src = img.src;
                if (caption) caption.innerHTML = img.caption || '';
                if (counter) counter.innerHTML = (index + 1) + ' / ' + images_{lightbox_id}.length;
                
                if (arrowLeft) arrowLeft.style.display = index > 0 ? 'block' : 'none';
                if (arrowRight) arrowRight.style.display = index < images_{lightbox_id}.length - 1 ? 'block' : 'none';
            }}
        }}
        
        window.closeLightbox_{lightbox_id} = function() {{
            const lightbox = document.getElementById('lightbox-{lightbox_id}');
            if (lightbox) lightbox.style.display = 'none';
        }}
        
        window.changeImage_{lightbox_id} = function(direction) {{
            let newIndex = currentImageIndex_{lightbox_id} + direction;
            if (newIndex >= 0 && newIndex < images_{lightbox_id}.length) {{
                window.openLightbox_{lightbox_id}(newIndex);
            }}
        }}
        
        setTimeout(function() {{
            const lightbox = document.getElementById('lightbox-{lightbox_id}');
            if (lightbox) {{
                lightbox.onclick = function(event) {{
                    if (event.target.id === 'lightbox-{lightbox_id}') {{
                        window.closeLightbox_{lightbox_id}();
                    }}
                }}
            }}
            
            const keyHandler_{lightbox_id} = function(event) {{
                const lightbox = document.getElementById('lightbox-{lightbox_id}');
                if (lightbox && lightbox.style.display === 'block') {{
                    if (event.key === 'Escape') {{
                        window.closeLightbox_{lightbox_id}();
                    }} else if (event.key === 'ArrowRight') {{
                        window.changeImage_{lightbox_id}(1);
                    }} else if (event.key === 'ArrowLeft') {{
                        window.changeImage_{lightbox_id}(-1);
                    }}
                }}
            }}
            
            document.removeEventListener('keydown', keyHandler_{lightbox_id});
            document.addEventListener('keydown', keyHandler_{lightbox_id});
        }}, 100);
    }})();
    </script>
    """, unsafe_allow_html=True)
    
    # Lightbox container
    st.markdown(f"""
    <div id="lightbox-{lightbox_id}" class="lightbox-{lightbox_id}">
        <span class="lightbox-close-{lightbox_id}" onclick="closeLightbox_{lightbox_id}()">&times;</span>
        <div class="lightbox-counter-{lightbox_id}" id="lightbox-counter-{lightbox_id}">1 / 1</div>
        <span class="lightbox-arrow-{lightbox_id} arrow-left-{lightbox_id}" id="arrow-left-{lightbox_id}" 
              onclick="changeImage_{lightbox_id}(-1)">&#10094;</span>
        <img class="lightbox-content-{lightbox_id}" id="lightbox-img-{lightbox_id}" alt="Image">
        <span class="lightbox-arrow-{lightbox_id} arrow-right-{lightbox_id}" id="arrow-right-{lightbox_id}" 
              onclick="changeImage_{lightbox_id}(1)">&#10095;</span>
        <div class="lightbox-caption-{lightbox_id}" id="lightbox-caption-{lightbox_id}"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # แสดงรูปเป็น Grid
    html_content = f'<div class="image-grid-{lightbox_id}">'
    
    for idx, img in enumerate(processed_images):
        caption_escaped = img['caption'].replace("'", "\\'").replace('"', '&quot;')
        src_escaped = img['src'].replace("'", "\\'")
        
        html_content += f"""
        <div class="thumbnail-container-{lightbox_id}" onclick="openLightbox_{lightbox_id}({idx})">
            <img src="{src_escaped}" class="thumbnail-{lightbox_id}" alt="{caption_escaped}" loading="lazy" onerror="this.src='https://via.placeholder.com/200x200?text=Image+Error'">
            <div class="thumbnail-overlay-{lightbox_id}">🔍</div>
            <div class="thumbnail-caption-{lightbox_id}">{img['caption']}</div>
        </div>
        """
    
    html_content += '</div>'
    st.markdown(html_content, unsafe_allow_html=True)

def clear_chat_history():
    """Clear chat history"""
    st.session_state.messages = []
    st.success("✅ ล้างประวัติการสนทนาแล้ว!")