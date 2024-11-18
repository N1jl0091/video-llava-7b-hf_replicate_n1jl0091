import os
import asyncio
import aiohttp
import torch
import numpy as np
import av
import time
from typing import List
from typing import Union
from cog import BasePredictor, Input, Path
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration, BitsAndBytesConfig
from PIL import Image

def print_log(*args, **kwargs):
    print(*args, **kwargs, flush=True)


WEIGHTS_CACHE = "/src/weights"


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

    def setup(self):
        start_time = time.time()
        print("Starting model setup...")

        os.makedirs(WEIGHTS_CACHE, exist_ok=True)
        os.environ["TRANSFORMERS_CACHE"] = WEIGHTS_CACHE

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

             # Print configuration details
        print_log("\nModel Configuration:")
        print_log(f"• Device: {self.device.type}")
      # print_log(f"• Dtype: {torch.bfloat16 if self.device.type == 'cuda' else torch.float32}")
      # print_log(f"• Device Map: {'auto' if self.device.type == 'cuda' else 'None'}")
      # print_log(f"• Quantization: {'8-bit' if self.device.type == 'cuda' else 'None'}")
        print_log(f"• Cache Dir: {os.environ['TRANSFORMERS_CACHE']}")
      # print_log(f"• Offload Folder: /tmp/offload")
        print_log("--------------------\n")

        model_id = os.getenv("MODEL_ID", "LanguageBind/Video-LLaVA-7B-hf")


        try:
                print(f"Loading model from local cache at {WEIGHTS_CACHE}...")
                # bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                self.model = VideoLlavaForConditionalGeneration.from_pretrained(
                    WEIGHTS_CACHE,
                    cache_dir=WEIGHTS_CACHE,
                    torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
                    device_map="auto" if self.device.type == "cuda" else None,
                    trust_remote_code=True,
                    use_safetensors=True,
                   # offload_folder="/tmp/offload",  # Enable disk offloading for faster boot?
                    # quantization_config=bnb_config
                )
                self.processor = VideoLlavaProcessor.from_pretrained(
                    WEIGHTS_CACHE,
                    trust_remote_code=True
                )
                if self.device.type != "cuda":
                    self.model = self.model.to(self.device)

                if self.device.type == "cuda":
                    print('setting memory fraction to 0.95')
                    # torch.cuda.empty_cache()
                    torch.cuda.set_per_process_memory_fraction(0.95)


                    # Load processor
                self.model.eval()
                print(f"Model loaded in {time.time() - start_time:.2f}s")


        except Exception as e:
            print(f"Error during setup: {str(e)}")
            raise RuntimeError(f"Model setup failed: {str(e)}") from e

    def predict(self, videos: List[Path], prompts: List[str], num_frames: int = 10,
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
