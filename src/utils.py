"""Utilities - Helper functions (Complete Fix)"""
import re
import base64
import json
from pathlib import Path
from io import BytesIO
from typing import List, Dict
from PIL import Image


class ImageHandler:
    """จัดการรูปภาพทุกรูปแบบ"""
    
    @staticmethod
    def extract_images_from_context(context: str) -> List[Dict]:
        """
        ดึงข้อมูลรูปภาพจาก context - Complete Fix Version
        
        Args:
            context: ข้อความจาก RAG
            
        Returns:
            List of image info [{"type": "url/local/base64", "data": "...", "caption": "..."}]
        """
        images = []
        seen_urls = set()
        
        # ========================================
        # วิธีที่ 1: Parse JSON โดยตรง (แม่นที่สุด!)
        # ========================================
        try:
            # ตรวจสอบว่าเป็น JSON หรือไม่
            if '{' in context and '"image_url"' in context:
                # หา JSON objects ทั้งหมด
                json_matches = re.finditer(r'\{[^{}]*"image_url"[^{}]*\}', context, re.DOTALL)
                
                for match in json_matches:
                    try:
                        json_str = match.group(0)
                        obj = json.loads(json_str)
                        
                        # ดึง image_url
                        if 'image_url' in obj and obj['image_url']:
                            url = obj['image_url']
                            caption = obj.get('name', obj.get('title', ''))
                            
                            if url and url not in seen_urls:
                                seen_urls.add(url)
                                images.append({
                                    "type": "url",
                                    "data": url,
                                    "caption": str(caption) if caption else ""
                                })
                    except:
                        continue
        except:
            pass
        
        # ========================================
        # วิธีที่ 2: Regex หา "image_url": "..."
        # ========================================
        json_pattern = r'"image_url"\s*:\s*"([^"]+)"'
        json_urls = re.findall(json_pattern, context)
        
        for url in json_urls:
            url = url.strip()
            if url and url not in seen_urls and len(url) > 25:  # URL ต้องยาวพอ
                seen_urls.add(url)
                images.append({
                    "type": "url",
                    "data": url,
                    "caption": ImageHandler._extract_caption_near(context, url)
                })
        
        # ========================================
        # วิธีที่ 3: Regex Patterns หลายแบบ
        # ========================================
        
        # Pattern A: Apple CDN complete URLs
        pattern_apple = r'https?://[^\s"\'<>]+(?:storeimages|images)[^\s"\'<>]+\.(?:jpg|jpeg|png|gif|webp)(?:\?[^\s"\'<>]*)?'
        
        # Pattern B: URLs ที่มี fmt= parameter (Apple style)
        pattern_fmt = r'https?://[^\s"\'<>]+[?&]fmt=(?:jpg|jpeg|png|gif|webp)(?:&[^\s"\'<>]*)?'
        
        # Pattern C: Standard image URLs
        pattern_standard = r'https?://[^\s"\'<>]+\.(?:jpg|jpeg|png|gif|webp|svg)(?:\?[^\s"\'<>]*)?'
        
        # Pattern D: URLs ที่มีคำว่า image/photo/picture
        pattern_keyword = r'https?://[^\s"\'<>]*(?:image|photo|picture)[^\s"\'<>]*\.(?:jpg|jpeg|png|gif|webp)(?:\?[^\s"\'<>]*)?'
        
        all_patterns = [pattern_fmt, pattern_apple, pattern_keyword, pattern_standard]
        
        for pattern in all_patterns:
            urls = re.findall(pattern, context, re.IGNORECASE)
            
            for url in urls:
                # Clean URL
                url = url.rstrip('.,;:!?)\'"')
                
                # Filter: URL ต้องยาวพอ (ไม่ใช่ URL ที่ถูกตัด)
                if url and url not in seen_urls and len(url) > 25:
                    seen_urls.add(url)
                    images.append({
                        "type": "url",
                        "data": url,
                        "caption": ImageHandler._extract_caption_near(context, url)
                    })
        
        # ========================================
        # วิธีที่ 4: หาทุก URL แล้วกรองที่มี keyword
        # ========================================
        all_url_pattern = r'https?://[^\s"\'<>]+'
        all_urls = re.findall(all_url_pattern, context)
        
        image_keywords = ['image', 'img', 'photo', 'picture', 'storeimages', 'cdn', 'media']
        
        for url in all_urls:
            url = url.rstrip('.,;:!?)\'"')
            
            # เช็คว่ามี keyword หรือ extension ที่เกี่ยวข้อง
            if any(kw in url.lower() for kw in image_keywords) or any(ext in url.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', 'fmt=jpeg', 'fmt=png']):
                if url and url not in seen_urls and len(url) > 25:
                    seen_urls.add(url)
                    images.append({
                        "type": "url",
                        "data": url,
                        "caption": ImageHandler._extract_caption_near(context, url)
                    })
        
        # ========================================
        # วิธีที่ 5: Local file paths
        # ========================================
        local_pattern = r'(?:data/images/|\.?/data/images/|images/)[^\s\)"\'\]<>]+\.(?:jpg|jpeg|png|gif|webp)'
        paths = re.findall(local_pattern, context, re.IGNORECASE)
        
        for path in paths:
            path = path.strip()
            if Path(path).exists():
                images.append({
                    "type": "local",
                    "data": path,
                    "caption": ImageHandler._extract_caption_near(context, path)
                })
        
        # ========================================
        # วิธีที่ 6: Base64 images
        # ========================================
        base64_pattern = r'image_base64:\s*([A-Za-z0-9+/=]{100,})'
        base64_matches = re.findall(base64_pattern, context)
        
        for b64 in base64_matches:
            images.append({
                "type": "base64",
                "data": b64.strip(),
                "caption": "Base64 Image"
            })
        
        return images
    
    @staticmethod
    def _extract_caption_near(context: str, image_ref: str, radius: int = 300) -> str:
        """
        หา caption ใกล้ๆ รูปภาพ
        
        Args:
            context: ข้อความทั้งหมด
            image_ref: URL หรือ path ของรูป
            radius: รัศมีการค้นหา (ตัวอักษร)
        
        Returns:
            caption string
        """
        try:
            # หา index ของรูป
            idx = context.find(image_ref)
            if idx == -1:
                # ถ้าหาไม่เจอ อาจเป็นเพราะ URL ถูก escape
                idx = context.find(image_ref.replace('/', '\\/'))
            
            if idx == -1:
                return ""
            
            # ดูข้อความข้างหน้า
            start = max(0, idx - radius)
            before = context[start:idx]
            
            # หาบรรทัดที่มีข้อมูล
            lines = before.split('\n')
            
            # ลองหา pattern ต่างๆ
            for line in reversed(lines):
                line = line.strip()
                
                # Skip บรรทัดว่าง
                if not line:
                    continue
                
                # Pattern 1: "name": "..."
                name_match = re.search(r'"name"\s*:\s*"([^"]+)"', line)
                if name_match:
                    return name_match.group(1)
                
                # Pattern 2: name: ... (ไม่มี quotes)
                if 'name:' in line.lower():
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        caption = parts[1].strip().strip('"\'').strip(',')
                        if caption and len(caption) < 100:
                            return caption
                
                # Pattern 3: "description": "..."
                desc_match = re.search(r'"description"\s*:\s*"([^"]+)"', line)
                if desc_match:
                    return desc_match.group(1)
                
                # Pattern 4: หา text ใดๆ ที่อยู่ใน quotes
                text_match = re.search(r':\s*"([^"]{10,80})"', line)
                if text_match and 'http' not in text_match.group(1):
                    return text_match.group(1)
            
            return ""
            
        except Exception as e:
            print(f"Error extracting caption: {e}")
            return ""
    
    @staticmethod
    def image_to_base64(image_path: str) -> str:
        """
        แปลงรูปภาพเป็น base64 string
        
        Args:
            image_path: path to image file
            
        Returns:
            base64 encoded string
        """
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
            return base64.b64encode(image_data).decode()
        except Exception as e:
            print(f"Error encoding image: {e}")
            return ""
    
    @staticmethod
    def base64_to_image(base64_str: str):
        """
        แปลง base64 เป็น PIL Image
        
        Args:
            base64_str: base64 encoded string
            
        Returns:
            PIL Image object
        """
        try:
            image_data = base64.b64decode(base64_str)
            return Image.open(BytesIO(image_data))
        except Exception as e:
            print(f"Error decoding base64 image: {e}")
            return None
    
    @staticmethod
    def validate_image_url(url: str) -> bool:
        """
        เช็คว่า URL เป็นรูปภาพจริงไหม
        
        Args:
            url: image URL
            
        Returns:
            True if valid image URL
        """
        try:
            import requests
            response = requests.head(url, timeout=5, allow_redirects=True)
            content_type = response.headers.get('content-type', '')
            return 'image' in content_type.lower()
        except:
            # ถ้า request ล้มเหลว ให้ถือว่าเป็นรูป (optimistic)
            return True


# Helper function สำหรับใช้งานง่ายๆ
def extract_images(context: str) -> List[Dict]:
    """Shortcut function"""
    return ImageHandler.extract_images_from_context(context)