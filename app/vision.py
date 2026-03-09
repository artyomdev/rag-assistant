"""
VisionProcessor – screenshot analysis using an Ollama vision model.

Improvements:
- Increased timeout (300 seconds).
- Progressive resize / degradation mode: 1280 → 1024 → 960 → 768 px.
- Small delay between requests to keep GPU stable.
"""
import logging
import time
from io import BytesIO
from PIL import Image
from ollama import Client, ResponseError
from app.config import settings

logger = logging.getLogger(__name__)


class VisionProcessor:
    def __init__(self):
        self.model = settings.VISION_MODEL
        
        self.client = Client(timeout=300.0)
        
        self.request_delay = 1.0
        
        logger.info(f"👀 VisionProcessor initialised with model {self.model} (timeout=300s, degrade: 1280→1024→960→768px)")

    def describe_image(self, image_data: Image.Image | None) -> str:
        """
        Accepts a PIL.Image object, optimises it and sends it to Ollama.
        Uses progressive resize: 1280 → 1024 → 960 → 768 px if errors occur.
        """
        if image_data is None:
            logger.warning("⚠️ Vision: None image provided, skipping")
            return "[Image was not loaded]"
        
        prompt = """
        Analyse this screenshot of a business application UI.
        Briefly list the key elements that are useful for search and navigation:
        1. Window title.
        2. Names of visible tabs, buttons and input fields.
        3. If there is an error message or a table, describe their essence.
        Do not start with phrases like "The screenshot shows". Just list the facts.
        """
        
        size_attempts = [1280, 1024, 960, 768]
        last_error = None
        
        for max_size in size_attempts:
            img_bytes = self._prepare_image(image_data, max_size=max_size, quality=70)
            if img_bytes is None:
                last_error = "image_preparation_failed"
                continue
            
            try:
                time.sleep(self.request_delay)
                
                response = self.client.chat(
                    model=self.model,
                    messages=[{
                        'role': 'user',
                        'content': prompt,
                        'images': [img_bytes]
                    }],
                    options={'num_ctx': 2048}
                )
                
                description = response['message']['content'].replace("\n", " ").strip()
                return f"[SCREENSHOT: {description}]"
                
            except (ResponseError, Exception) as e:
                last_error = e
                logger.warning(f"⚠️ Vision error at size {max_size}px: {e}. Trying a smaller size...")
        
        logger.error(f"❌ Vision: all attempts failed. Last error: {last_error}")
        return "[Image analysis failed after several attempts]"

    def _prepare_image(self, image_data: Image.Image | None, max_size: int = 1280, quality: int = 70) -> bytes | None:
        """Prepare image bytes for sending to the model."""
        if image_data is None:
            return None
        try:
            target_size = (max_size, max_size)
            image_copy = image_data.copy()
            image_copy.thumbnail(target_size, Image.Resampling.LANCZOS)

            if image_copy.mode in ("RGBA", "P"):
                image_copy = image_copy.convert("RGB")
            
            buffered = BytesIO()
            image_copy.save(buffered, format="JPEG", quality=quality)
            return buffered.getvalue()
            
        except Exception as e:
            logger.error(f"⚠️ Error while preparing image: {e}")
            return None
