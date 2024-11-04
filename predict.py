import os
import asyncio
import aiohttp
import torch
import numpy as np
import av
import time
from typing import List
from cog import BasePredictor, Input, Path
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration

async def download_weights(url: str, dest: str):
    """Download model weights asynchronously."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                with open(dest, 'wb') as f:
                    f.write(await response.read())
            else:
                raise RuntimeError(f"Failed to download weights: {response.status}")

def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    for i, frame in enumerate(container.decode(video=0)):
        if i > indices[-1]:
            break
        if i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

class Predictor:
    def __init__(self):
        self.model = None
        self.processor = None

    async def setup(self):
        start_time = time.time()
        print("Starting model setup...")

        os.environ["TRANSFORMERS_CACHE"] = os.getenv("TRANSFORMERS_CACHE", "/src/model_cache")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        model_id = os.getenv("MODEL_ID", "LanguageBind/Video-LLaVA-7B-hf")
        weights_url = os.getenv("WEIGHTS_URL", "https://path/to/weights/model.tar")
        weights_dest = os.path.join(os.environ["TRANSFORMERS_CACHE"], "model.tar")

        # Download weights asynchronously
        await download_weights(weights_url, weights_dest)

        try:
            self.model = VideoLlavaForConditionalGeneration.from_pretrained(
                weights_dest,
                torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None,
                trust_remote_code=True,
                use_safetensors=True,
                load_in_8bit=True  # Enables quantization for smaller memory usage
            )

            if self.device.type != "cuda":
                self.model = self.model.to(self.device)

            # Load processor
            self.processor = VideoLlavaProcessor.from_pretrained(model_id)
            self.model.eval()
            print(f"Model loaded in {time.time() - start_time:.2f}s")

        except Exception as e:
            print(f"Error during setup: {str(e)}")
            raise RuntimeError(f"Model setup failed: {str(e)}") from e

    async def predict(self, videos: List[Path], prompts: List[str], num_frames: int = 10,
                      max_new_tokens: int = 500, temperature: float = 0.1, top_p: float = 0.9) -> List[str]:
        results = []
        for video, prompt in zip(videos, prompts):
            try:
                predict_start = time.time()
                print(f"Starting prediction for {video} at {time.strftime('%H:%M:%S')}")

                container = av.open(str(video))
                total_frames = container.streams.video[0].frames

                frames_to_use = min(total_frames, num_frames) if total_frames > 0 else num_frames
                print(f"Using {frames_to_use} frames")
                indices = np.linspace(0, total_frames - 1, frames_to_use, dtype=int)
                clip = read_video_pyav(container, indices)

                full_prompt = f"USER: <video>{prompt} ASSISTANT:"
                inputs = self.processor(text=full_prompt, videos=clip, return_tensors="pt")

                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.inference_mode():
                    generate_ids = self.model.generate(
                        **inputs,
                        max_length=max_new_tokens,
                        do_sample=temperature > 0,
                        temperature=temperature,
                        top_p=top_p
                    )

                result = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                results.append(result.split("ASSISTANT:")[-1].strip())
                print(f"Total prediction time for {video}: {time.time() - predict_start:.2f}s")

            except Exception as e:
                print(f"Error during prediction: {str(e)}")
                results.append(f"Error during prediction: {str(e)}")
            finally:
                if 'container' in locals():
                    container.close()

        return results

if __name__ == "__main__":
    video_paths = ["path_to_video1.mp4", "path_to_video2.mp4"]  # List of video paths
    prompts = ["What is happening in this video?", "Summarize this clip."]

    predictor = Predictor()

    # Run setup
    asyncio.run(predictor.setup())

    # Warm up run
    results = asyncio.run(predictor.predict(video_paths, prompts, max_new_tokens=150))
    for result in results:
        print("Result:", result)
