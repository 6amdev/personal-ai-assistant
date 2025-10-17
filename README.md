# 💬 Personal AI Assistant

> AI Chatbot ส่วนตัวที่เรียนรู้จากเอกสารของคุณ - ฟรี 100% รันบนเครื่องเอง!

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Ollama](https://img.shields.io/badge/ollama-required-orange.svg)](https://ollama.com)
[![Made by](https://img.shields.io/badge/made%20by-6amdev-red.svg)](https://github.com/6amdev)

## ✨ Features

- 🤖 **Multi-Model Support** - เลือกใช้ LLM หลายตัว (Llama, Qwen, DeepSeek, etc.)
- 📚 **Advanced RAG** - ระบบ RAG 6 แบบ (Naive, Contextual, Rerank, Hybrid, Query Rewrite, Multi-step)
- 💾 **Persistent Memory** - จำบทสนทนาและเอกสารถาวรด้วย ChromaDB
- 🖼️ **Image Support** - แสดงรูปภาพจากเอกสาร
- 📄 **Multi-Format** - รองรับ TXT, PDF, DOCX, JSON, MD
- 🇹🇭 **Thai Language** - รองรับภาษาไทยได้ดี (โดยเฉพาะ Qwen 2.5)
- 🔒 **100% Private** - ข้อมูลไม่ออกจากเครื่อง
- 🆓 **Free & Open Source** - ใช้ฟรี ปรับแต่งได้
- ⚡ **Fast Embeddings** - Sentence Transformers (5-10x เร็วกว่า Ollama embeddings)
- 🎨 **Modern UI** - Interface สวยงามด้วย Streamlit

---

## 🚀 Quick Start

### Installation

```bash
# 1. Clone repository
git clone https://github.com/6amdev/personal-ai-assistant.git
cd personal-ai-assistant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install Ollama (if not installed)
curl -fsSL https://ollama.com/install.sh | sh

# 4. Pull AI models
ollama pull llama3.1:8b      # Default
ollama pull qwen2.5:7b        # Recommended for Thai
ollama pull deepseek-r1:8b    # Good for reasoning

# 5. Run the app
streamlit run app.py
```

เปิด browser ที่ `http://localhost:8501` 🎉

---

## 📖 Usage

### 💬 Chat
พิมพ์คำถาม → AI ตอบ (ใช้ความรู้ทั่วไป)

### 📚 Upload Documents
1. Sidebar → **📤 Upload Documents**
2. เลือกไฟล์ (TXT, PDF, DOCX, JSON, MD)
3. รอระบบเรียนรู้เอกสาร (~10-30s)
4. ถามคำถามเกี่ยวกับเอกสาร → AI ตอบจากข้อมูลที่อัปโหลด

### 🤖 Select Models
Sidebar → **🤖 LLM Model** → เลือก model ที่ต้องการ

**แนะนำ:**
- **Qwen 2.5:7b** - ดีที่สุดสำหรับภาษาไทย 🔥
- **DeepSeek R1:8b** - เก่งคณิตศาสตร์และ reasoning
- **Llama 3.1:8b** - Balanced, ใช้งานทั่วไป

### 🔍 Select RAG Type
Sidebar → **🔍 RAG Type** → เลือกแบบ RAG

**แนะนำ:**
- **Hybrid RAG** 🔥 - ดีที่สุดสำหรับภาษาไทย (BM25 + Vector)
- **Naive RAG** - เร็วที่สุด (พื้นฐาน)
- **Multi-step RAG** - เหมาะกับคำถามซับซ้อน

### 🗑️ Manage Documents
Sidebar → **📄 View Documents** → คลิก **🗑️** เพื่อลบทีละไฟล์

---

## 🛠️ Tech Stack

### Core Technologies

| Technology | Purpose | Why We Use It |
|-----------|---------|---------------|
| **[Streamlit](https://streamlit.io)** | Web UI Framework | สร้าง web app ด้วย Python ง่ายๆ ไม่ต้องเขียน HTML/CSS |
| **[LangChain](https://langchain.com)** | LLM Framework | เชื่อมต่อ LLM, Memory, และ Tools เข้าด้วยกัน |
| **[Llama 3.1 8B](https://ai.meta.com/llama/)** | Language Model | Open source LLM จาก Meta ฉลาดและรองรับภาษาไทย |
| **[Qwen 2.5](https://qwenlm.github.io/)** | Language Model | LLM จาก Alibaba ดีเยี่ยมกับภาษาไทยและหลายภาษา |
| **[DeepSeek R1](https://www.deepseek.com/)** | Language Model | เก่งด้าน reasoning และคณิตศาสตร์ |
| **[Ollama](https://ollama.com)** | LLM Runtime | รัน LLM บนเครื่องเราได้ ไม่ต้องใช้ API |
| **[ChromaDB](https://trychroma.com)** | Vector Database | เก็บ embeddings สำหรับค้นหาเอกสารที่เกี่ยวข้อง |
| **[Sentence Transformers](https://www.sbert.net/)** | Embeddings | แปลงข้อความเป็น vectors (เร็วกว่า Ollama 5-10x) |
| **Python 3.9+** | Backend Language | ภาษาหลักของโปรเจกต์ |

### Document Processing

| Library | Purpose |
|---------|---------|
| **pypdf** | อ่านไฟล์ PDF |
| **python-docx** | อ่านไฟล์ DOCX (Word) |
| **Pillow** | จัดการรูปภาพ |

### AI Components

**RAG (Retrieval-Augmented Generation):**

ระบบมี 6 แบบให้เลือก:

1. **Naive RAG** - พื้นฐาน ค้นหาด้วย vector similarity
2. **Contextual RAG** - เน้นการใช้ context รอบๆ
3. **Rerank RAG** - ค้นหาแล้วจัดอันดับใหม่
4. **Hybrid RAG** 🔥 - รวม BM25 (keyword) + Vector (semantic) - ดีที่สุดสำหรับภาษาไทย!
5. **Query Rewrite RAG** - เขียนคำถามใหม่หลายแบบ
6. **Multi-step RAG** - ค้นหาหลายรอบถ้าข้อมูลไม่พอ

**Embeddings:**
- ใช้ **Sentence Transformers** (paraphrase-multilingual-MiniLM-L12-v2)
- แปลงข้อความเป็น vectors (384 dimensions)
- เก็บใน ChromaDB สำหรับการค้นหาแบบ semantic
- รองรับภาษาไทยและ 50+ ภาษา

**Memory:**
- **Short-term:** จำบทสนทนาปัจจุบัน (ConversationBufferMemory)
- **Long-term:** จำเอกสารถาวร (ChromaDB)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│           User Interface (Streamlit)                    │
│  - Chat input/output                                    │
│  - Model selector (Llama/Qwen/DeepSeek)                │
│  - RAG type selector (6 types)                         │
│  - File uploader                                        │
│  - Document management UI                               │
│  - Image viewer (Lightbox)                             │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│      Application Layer (LangChain + Custom RAG)         │
│  - Multi-model LLM handler                              │
│  - RAG implementations (6 types)                        │
│  - Prompt engineering                                   │
│  - Context retrieval                                    │
│  - Response generation                                  │
│  - Document processing (TXT/PDF/DOCX/JSON/MD)          │
│  - Image extraction                                     │
└──────┬────────────────────────────────┬─────────────────┘
       │                                │
┌──────▼────────────────┐  ┌────────────▼─────────────────┐
│ 💾 Memory (ChromaDB) │  │  🤖 LLM (Ollama)            │
│  - Vector Database    │  │  - Llama 3.1/3.2 (Meta)     │
│  - Sentence Trans.    │  │  - Qwen 2.5 (Alibaba)       │
│  - 384D Embeddings    │  │  - DeepSeek R1              │
│  - Semantic Search    │  │  - Dynamic switching        │
│  - Persistence        │  │  - GPU/CPU support          │
│  - Multi-language     │  │  - Local runtime            │
└───────────────────────┘  └──────────────────────────────┘
```

### Data Flow

**1. Document Upload Flow:**
```
User uploads document
    ↓
Document Processor
    ├─ PDF → pypdf
    ├─ DOCX → python-docx
    ├─ JSON → json parser
    └─ TXT/MD → text reader
    ↓
Split into chunks (500 chars)
    ↓
Generate embeddings (Sentence Transformers)
    ↓
Store in ChromaDB (Vector Database)
    ↓
✅ Ready for search
```

**2. Query Flow (RAG):**
```
User asks question
    ↓
RAG System (เลือก 1 จาก 6 แบบ)
    ├─ Naive: Vector search
    ├─ Contextual: Context-aware
    ├─ Rerank: Vector + Reranking
    ├─ Hybrid: BM25 + Vector 🔥
    ├─ Query Rewrite: Multiple queries
    └─ Multi-step: Iterative search
    ↓
Retrieve relevant chunks (k=3)
    ↓
Extract images (if any)
    ↓
Build context + prompt
    ↓
Send to LLM (Llama/Qwen/DeepSeek)
    ↓
Generate answer
    ↓
Display to user + images
```

**3. Model Switching Flow:**
```
User selects new model
    ↓
Clear cache
    ↓
Initialize new LLMHandler
    ↓
Reinitialize RAG system
    ↓
✅ Ready with new model
    (Documents remain unchanged)
```

---

## 📊 Performance

### Chat Response Time

| Model | RAM Usage | Speed | Quality |
|-------|-----------|-------|---------|
| Llama 3.2:1b | ~2GB | ⚡⚡⚡ | ⭐⭐ |
| Llama 3.2:3b | ~4GB | ⚡⚡⚡ | ⭐⭐⭐ |
| Qwen 2.5:7b | ~8GB | ⚡⚡ | ⭐⭐⭐⭐ |
| Llama 3.1:8b | ~8GB | ⚡⚡ | ⭐⭐⭐ |
| DeepSeek R1:8b | ~8GB | ⚡⚡ | ⭐⭐⭐⭐ |
| Qwen 2.5:14b | ~16GB | ⚡ | ⭐⭐⭐⭐⭐ |

### Upload Performance

| Task | Time |
|------|------|
| Upload 1 file (500 KB) | 5-10s |
| Upload 5 files (2.5 MB) | 15-30s |
| Embedding generation | ~0.02s/chunk |
| Semantic Search | <0.5s |

*Tested on: RTX 4070 Ti 12GB, 64GB RAM*

### RAG Comparison

| RAG Type | Speed | Accuracy | Best For |
|----------|-------|----------|----------|
| Naive | ⚡⚡⚡ | ⭐⭐ | ทั่วไป |
| Contextual | ⚡⚡⚡ | ⭐⭐⭐ | ทั่วไป |
| Rerank | ⚡⚡ | ⭐⭐⭐ | ความแม่นยำ |
| Hybrid 🔥 | ⚡⚡ | ⭐⭐⭐⭐ | ภาษาไทย |
| Query Rewrite | ⚡⚡ | ⭐⭐⭐ | คำถามคลุมเครือ |
| Multi-step | ⚡ | ⭐⭐⭐⭐ | คำถามซับซ้อน |

---

## 🍓 Raspberry Pi Support

ระบบรองรับ Raspberry Pi! 🎉

### Recommended Models

| Raspberry Pi | RAM | Model | Performance |
|-------------|-----|-------|-------------|
| **Pi 4 (2GB)** | 2GB | llama3.2:1b | 🐌 ช้า |
| **Pi 4 (4GB)** | 4GB | llama3.2:3b | 🐢 ปานกลาง |
| **Pi 5 (8GB)** | 8GB | qwen2.5:3b | 🚶 ใช้ได้ |

### Installation on Pi

```bash
# 1. Update system
sudo apt update && sudo apt upgrade -y

# 2. Install dependencies
sudo apt install -y python3-pip python3-venv git

# 3. Increase swap (important!)
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Set: CONF_SWAPSIZE=4096
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# 4. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 5. Clone and install
git clone https://github.com/6amdev/personal-ai-assistant.git
cd personal-ai-assistant
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 6. Pull lightweight model
ollama pull llama3.2:3b

# 7. Run
streamlit run app.py --server.address 0.0.0.0
```

Access from another device: `http://[Pi-IP]:8501`

---

## 🐛 Troubleshooting

### Ollama not found
```bash
# Check if Ollama is running
ollama --version

# If not found, install
curl -fsSL https://ollama.com/install.sh | sh
```

### Model not found
```bash
# List available models
ollama list

# Pull missing model
ollama pull llama3.1:8b
```

### Import error
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Out of Memory
```bash
# Use smaller model
ollama pull llama3.2:3b

# Or increase swap (Linux)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Slow performance
**Solutions:**
- ใช้ GPU (ติดตั้ง CUDA drivers)
- ใช้ model เล็กกว่า (llama3.2:3b)
- เปลี่ยน embedding device เป็น `cpu` ใน `.env`
- ลด `CHUNK_SIZE` ใน `.env`

### CUDA not available
```bash
# Edit .env
EMBEDDING_DEVICE=cpu  # Change from cuda to cpu
```

---

## 📁 Project Structure

```
personal-ai-assistant/
├── app.py                      # Main application
├── config.py                   # Configuration
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables
├── .env.example               # Example config
├── README.md                  # This file
│
├── src/
│   ├── __init__.py
│   ├── llm.py                 # LLM Handler (Multi-model)
│   ├── memory.py              # Memory & Vector DB
│   ├── ui.py                  # Streamlit UI (Model selector)
│   ├── document_processor.py  # Document processing
│   ├── utils.py               # Image handler
│   │
│   └── rag/                   # RAG implementations
│       ├── __init__.py
│       ├── base_rag.py        # Base class
│       ├── naive_rag.py       # Naive RAG
│       ├── contextual_rag.py  # Contextual RAG
│       ├── rerank_rag.py      # Rerank RAG
│       ├── hybrid_rag.py      # Hybrid RAG 🔥
│       ├── query_rewrite_rag.py
│       └── multistep_rag.py
│
└── data/
    ├── chroma_db/             # Vector database (auto-created)
    └── images/                # Uploaded images (optional)
```

---

## 🎯 Configuration

Edit `.env` file:

```bash
# LLM Settings
LLM_MODEL=llama3.1:8b           # Default model
LLM_TEMPERATURE=0.7             # Creativity (0-1)

# Embeddings
EMBEDDING_PROVIDER=sentence-transformers
SENTENCE_TRANSFORMER_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_DEVICE=cuda           # cuda or cpu

# Database
CHROMA_DB_DIR=./data/chroma_db
COLLECTION_NAME=personal_assistant

# Document Processing
CHUNK_SIZE=500                  # Chunk size in characters
CHUNK_OVERLAP=50                # Overlap between chunks
```

---

## 🎨 Advanced Features

### Image Support
- ระบบดึงรูปภาพจาก JSON/URL โดยอัตโนมัติ
- แสดงด้วย Lightbox (คลิกขยายได้)
- รองรับ URL, Local file, Base64

### Debug Mode
- เปิด Debug Mode ใน sidebar
- ดูข้อมูล: RAG type, Context length, Sources, Images
- เหมาะกับการทดสอบและ fine-tune

### Document Management
- ดูรายการเอกสารทั้งหมด
- ลบทีละไฟล์
- ลบทั้งหมดได้

---

## 🤝 Contributing

Fork → Branch → Commit → Push → Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/personal-ai-assistant.git

# Create branch
git checkout -b feature/amazing-feature

# Make changes and test
streamlit run app.py

# Commit
git commit -m "Add amazing feature"

# Push
git push origin feature/amazing-feature

# Create Pull Request on GitHub
```

---

## 📝 License

MIT License - ใช้ฟรี ปรับแต่งได้ แชร์ต่อได้

---

## 🙏 Acknowledgments

Special thanks to:
- **Meta AI** for Llama 3.1/3.2
- **Alibaba** for Qwen 2.5
- **DeepSeek** for DeepSeek R1
- **Ollama team** for local LLM runtime
- **LangChain** for the amazing framework
- **Streamlit** for the beautiful UI
- **ChromaDB** for vector storage
- **Sentence Transformers** for fast embeddings

---

## 📧 Contact & Support

**6amdev** 
- 🐙 GitHub: [github.com/6amdev](https://github.com/6amdev)
- 🐛 Issues: [Report bugs](https://github.com/6amdev/personal-ai-assistant/issues)
- 💬 Discussions: [Ask questions](https://github.com/6amdev/personal-ai-assistant/discussions)

---

## 🌟 Star History

If you find this project useful, please give it a star! ⭐

---

**Made with ❤️ by [6amdev](https://github.com/6amdev)**

*Your data, your AI, your rules.* 🔒