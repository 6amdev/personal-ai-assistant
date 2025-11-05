#!/bin/bash

# 🚀 API Security Setup Script
# Auto-setup API with security features

set -e  # Exit on error

echo "============================================================"
echo "🚀 API Security Setup"
echo "============================================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Step 1: Check Python
echo "Step 1: Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    print_success "Python found: $PYTHON_VERSION"
else
    print_error "Python3 not found! Please install Python 3.8+"
    exit 1
fi

# Step 2: Install python-dotenv
echo ""
echo "Step 2: Installing python-dotenv..."
if python3 -c "import dotenv" 2>/dev/null; then
    print_success "python-dotenv already installed"
else
    print_info "Installing python-dotenv..."
    pip install python-dotenv --break-system-packages
    print_success "python-dotenv installed"
fi

# Step 3: Generate API Key
echo ""
echo "Step 3: Generating API Key..."
API_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
print_success "API Key generated: ${API_KEY:0:8}..."

# Step 4: Setup .env file
echo ""
echo "Step 4: Setting up .env file..."

if [ -f ".env" ]; then
    print_warning ".env file already exists!"
    read -p "Do you want to backup and replace it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        BACKUP_FILE=".env.backup.$(date +%Y%m%d_%H%M%S)"
        cp .env "$BACKUP_FILE"
        print_success "Backup created: $BACKUP_FILE"
    else
        print_info "Skipping .env setup"
        exit 0
    fi
fi

# Create .env file
cat > .env << EOF
# LLM
LLM_MODEL=llama3.1:8b
LLM_TEMPERATURE=0.7

# Embeddings
EMBEDDING_PROVIDER=sentence-transformers
SENTENCE_TRANSFORMER_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_DEVICE=cuda

# OpenAI (optional)
OPENAI_API_KEY=
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Database
CHROMA_DB_DIR=./data/chroma_db
COLLECTION_NAME=personal_assistant

# Document Processing
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# API Security
API_KEY=$API_KEY
ENABLE_AUTH=true
EOF

print_success ".env file created"

# Step 5: Create .gitignore if not exists
echo ""
echo "Step 5: Setting up .gitignore..."

if [ ! -f ".gitignore" ]; then
    cat > .gitignore << 'EOF'
# Environment Variables
.env
.env.local
.env.*.local

# API Keys
*.key
*.pem
secrets/

# Python
__pycache__/
*.py[cod]
.Python
venv/
*.egg-info/

# Data
data/
*.db
chroma_db/
lightrag_db/
uploads/

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db
EOF
    print_success ".gitignore created"
else
    print_info ".gitignore already exists"
fi

# Step 6: Create data directories
echo ""
echo "Step 6: Creating data directories..."
mkdir -p data/chroma_db data/lightrag_db data/uploads
print_success "Data directories created"

# Step 7: Display summary
echo ""
echo "============================================================"
echo "✅ Setup Complete!"
echo "============================================================"
echo ""
echo "📝 Configuration Summary:"
echo "   API Key: ${API_KEY:0:12}...${API_KEY: -8}"
echo "   Auth Enabled: true"
echo "   .env: Created"
echo "   .gitignore: Ready"
echo ""
echo "🚀 Next Steps:"
echo ""
echo "1. Start the API:"
echo "   ${BLUE}python api.py${NC}"
echo ""
echo "2. Test the API:"
echo "   ${BLUE}python3 test_api_auth.py${NC}"
echo ""
echo "3. View API docs:"
echo "   ${BLUE}http://localhost:8000/docs${NC}"
echo ""
echo "4. Test with cURL:"
echo "   ${BLUE}curl -X POST http://localhost:8000/chat \\${NC}"
echo "     ${BLUE}-H \"X-API-Key: $API_KEY\" \\${NC}"
echo "     ${BLUE}-H \"Content-Type: application/json\" \\${NC}"
echo "     ${BLUE}-d '{\"query\": \"Hello\", \"rag_type\": \"hybrid\"}'${NC}"
echo ""
echo "⚠️  Security Reminder:"
echo "   - Keep your API key secret!"
echo "   - Never commit .env to git"
echo "   - Use HTTPS in production"
echo ""
echo "📖 Documentation:"
echo "   - QUICK_START.md - Quick start guide"
echo "   - API_SECURITY_GUIDE.md - Detailed security docs"
echo "   - README_SECURITY.md - Update summary"
echo ""
echo "============================================================"
echo "🎉 Happy Coding!"
echo "============================================================"