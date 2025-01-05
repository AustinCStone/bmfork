import aiohttp
import asyncio
import os
from pathlib import Path
from typing import Dict, List, Optional

import bittensor as bt


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

      
async def download_video(url: str, output_path: Path) -> bool:
    """
    Download a video from a URL and save it to the specified path.
    Returns True if successful, False otherwise.
    """
    try:
        os.makedirs(output_path.parent, exist_ok=True)
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    bt.logging.error(f"Failed to download video: HTTP {response.status}")
                    return False

                total_size = int(response.headers.get('content-length', 0))
                bytes_downloaded = 0
                with open(output_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        if chunk:
                            f.write(chunk)
                            bytes_downloaded += len(chunk)
                            if total_size:
                                progress = (bytes_downloaded / total_size) * 100
                                bt.logging.debug(f"Download progress: {progress:.1f}%")
                bt.logging.info(f"Successfully downloaded video to {output_path}")
                return True
    except Exception as e:
        bt.logging.error(f"Error downloading video: {str(e)}")
        if output_path.exists():
            output_path.unlink()  # Clean up partial download
        return False