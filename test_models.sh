#!/bin/bash

echo "🧪 Testing Models for Raspberry Pi..."
echo ""

models=(
    "llama3.2:1b"
    "phi3.5:3.8b"
    "llama3.2:3b"
    "qwen2.5:3b"
    "qwen2.5:7b"
)

test_prompt="สวัสดีครับ ช่วยอธิบาย AI ให้เข้าใจง่ายๆ ใน 2-3 ประโยค"

for model in "${models[@]}"
do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🤖 Testing: $model"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # เช็คว่ามี model หรือยัง
    if ollama list | grep -q "$model"; then
        echo "✅ Model exists"
        
        # ทดสอบความเร็ว
        echo ""
        echo "⏱️  Speed test..."
        time ollama run "$model" "$test_prompt"
        
    else
        echo "⚠️  Model not found. Run: ollama pull $model"
    fi
    
    echo ""
    echo ""
done

echo "✅ Testing complete!"