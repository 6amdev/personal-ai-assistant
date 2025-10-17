#!/bin/bash

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  💬 Personal AI Assistant - Installation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check OS
OS="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="mac"
else
    echo -e "${RED}❌ Unsupported OS: $OSTYPE${NC}"
    exit 1
fi

echo "🖥️  Detected OS: $OS"
echo ""

# Check Python
echo "📦 Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo -e "${GREEN}✅ Python $PYTHON_VERSION found${NC}"
else
    echo -e "${RED}❌ Python 3 not found${NC}"
    echo "Install Python 3.8+: https://www.python.org/downloads/"
    exit 1
fi

# Check RAM
echo ""
echo "💾 Checking System RAM..."
if [[ "$OS" == "linux" ]]; then
    TOTAL_RAM=$(free -g | awk '/^Mem:/{print $2}')
elif [[ "$OS" == "mac" ]]; then
    TOTAL_RAM=$(sysctl -n hw.memsize | awk '{print int($1/1024/1024/1024)}')
fi

echo "Total RAM: ${TOTAL_RAM}GB"

if [ "$TOTAL_RAM" -lt 4 ]; then
    echo -e "${RED}⚠️  Warning: Low RAM (${TOTAL_RAM}GB). Recommend 4GB+${NC}"
    echo "Consider using lightweight models (llama3.2:1b)"
elif [ "$TOTAL_RAM" -lt 8 ]; then
    echo -e "${YELLOW}⚠️  RAM: ${TOTAL_RAM}GB. Can run small-medium models${NC}"
    echo "Recommended: llama3.2:3b, qwen2.5:3b"
else
    echo -e "${GREEN}✅ RAM: ${TOTAL_RAM}GB. Can run large models${NC}"
    echo "Recommended: qwen2.5:7b, llama3.1:8b"
fi

# Check Ollama
echo ""
echo "🤖 Checking Ollama..."
if command -v ollama &> /dev/null; then
    OLLAMA_VERSION=$(ollama --version 2>&1 | head -n 1)
    echo -e "${GREEN}✅ Ollama found: $OLLAMA_VERSION${NC}"
else
    echo -e "${YELLOW}⚠️  Ollama not found${NC}"
    echo ""
    echo "Installing Ollama..."
    
    if [[ "$OS" == "linux" ]] || [[ "$OS" == "mac" ]]; then
        curl -fsSL https://ollama.com/install.sh | sh
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Ollama installed successfully${NC}"
        else
            echo -e "${RED}❌ Failed to install Ollama${NC}"
            exit 1
        fi
    fi
fi

# Create virtual environment
echo ""
echo "🐍 Creating virtual environment..."
if [ -d "venv" ]; then
    echo -e "${YELLOW}⚠️  venv already exists. Skipping...${NC}"
else
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
fi

# Activate venv
echo ""
echo "📦 Installing Python packages..."
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip > /dev/null 2>&1

# Install requirements
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Python packages installed${NC}"
    else
        echo -e "${RED}❌ Failed to install packages${NC}"
        exit 1
    fi
else
    echo -e "${RED}❌ requirements.txt not found${NC}"
    exit 1
fi

# Create .env if not exists
if [ ! -f ".env" ]; then
    echo ""
    echo "⚙️  Creating .env file..."
    cat > .env << EOF
# LLM
LLM_MODEL=llama3.1:8b
LLM_TEMPERATURE=0.7

# Embeddings
EMBEDDING_PROVIDER=sentence-transformers
SENTENCE_TRANSFORMER_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_DEVICE=cuda

# Database
CHROMA_DB_DIR=./data/chroma_db
COLLECTION_NAME=personal_assistant

# Document Processing
CHUNK_SIZE=500
CHUNK_OVERLAP=50
EOF
    echo -e "${GREEN}✅ .env file created${NC}"
fi

# Suggest models
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  📥 Recommended Models"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ "$TOTAL_RAM" -lt 4 ]; then
    echo "For your system (${TOTAL_RAM}GB RAM):"
    echo "  ollama pull llama3.2:1b"
elif [ "$TOTAL_RAM" -lt 8 ]; then
    echo "For your system (${TOTAL_RAM}GB RAM):"
    echo "  ollama pull llama3.2:3b"
    echo "  ollama pull qwen2.5:3b"
else
    echo "For your system (${TOTAL_RAM}GB RAM):"
    echo "  ollama pull qwen2.5:7b"
    echo "  ollama pull llama3.1:8b"
    echo "  ollama pull deepseek-r1:8b"
fi

echo ""
read -p "Pull recommended model now? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ "$TOTAL_RAM" -lt 4 ]; then
        ollama pull llama3.2:1b
    elif [ "$TOTAL_RAM" -lt 8 ]; then
        ollama pull llama3.2:3b
    else
        ollama pull qwen2.5:7b
    fi
fi

# Done
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}  ✅ Installation Complete!${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🚀 To start the app:"
echo "  source venv/bin/activate"
echo "  streamlit run app.py"
echo ""
echo "🌐 Then open: http://localhost:8501"
echo ""