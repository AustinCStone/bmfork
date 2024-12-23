import aiohttp
import asyncio
from typing import Dict, List, Optional

class GenerationServiceProxyClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }

    async def generate_video(self, prompt: str, duration: int = 5) -> Dict:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/v1/generate-video",
                headers=self.headers,
                json={
                    "prompt": prompt,
                    "duration": duration,
                    "provider": "kling",
                    "additional_params": {
                        "mode": "std",
                        "aspect_ratio": "16:9"
                    }
                }
            ) as response:
                return await response.json()

    async def check_status(self, task_id: str) -> Optional[Dict]:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/api/v1/videos/{task_id}",
                headers=self.headers
            ) as response:
                if response.status != 200:
                    return None
                return await response.json()