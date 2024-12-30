from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import threading

import numpy as np
import queue
import time
import cv2
from PIL import Image
from pathlib import Path
from optimum.intel.openvino import OVModelForVisualCausalLM
from transformers import AutoProcessor
import json
import re
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect


# UnifiedProcessor class definition
class UnifiedProcessor:
    MODEL_REPO = "mistral-community/pixtral-12b"
    MODEL_BASE_DIR = Path("models")
    PRECISION = "INT4"
    DEVICE = "AUTO"

    def __init__(self, video_path=None):
        self.video_path = video_path
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.threads = []
        self.question = (
            "Provide a direct and concise answer in JSON format. "
            "The JSON should include key: 'wildfire'. "
            "The 'wildfire' key must have a boolean value indicating whether a wildfire is detected. "
        )
        self.display_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.model_dir = self.MODEL_BASE_DIR / self.PRECISION
        self.processor = AutoProcessor.from_pretrained(self.model_dir)
        self.ov_model = OVModelForVisualCausalLM.from_pretrained(
            self.model_dir, device=self.DEVICE.lower()
        )

    def update_video_path(self, new_video_path):
        self.video_path = new_video_path

    def set_question(self, question):
        self.question = question

    def process_image_with_question(self, image, question):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "content": question},
                    {"type": "image"},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = self.processor(text=text, images=[image], return_tensors="pt")
        generate_ids = self.ov_model.generate(
            **inputs, do_sample=False, max_new_tokens=50
        )
        output = self.processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
        json_match = re.search(r'```json\s*({.*?})\s*```', output, re.DOTALL)
        return json.loads(json_match.group(1)) if json_match else {}

    def video_loop(self):
        import os

        # Wait for a valid video path
        while self.video_path is None or not os.path.exists(self.video_path):
            if self.video_path is None:
                print("Waiting for video path to be updated...")
            else:
                print(f"Video path '{self.video_path}' does not exist. Waiting for updates...")
            time.sleep(5)

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Unable to open video file '{self.video_path}'.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_time = 1 / fps

        while not self.stop_event.is_set():
            start_time = time.time()
            ret, frame = cap.read()

            if not ret or frame is None:
                # Check if video path has changed or is invalid
                if self.video_path is None or not os.path.exists(self.video_path):
                    print("Video path updated or no longer exists. Re-checking...")
                    cap.release()
                    while self.video_path is None or not os.path.exists(self.video_path):
                        if self.video_path is None:
                            print("Waiting for video path to be updated...")
                        else:
                            print(f"Video path '{self.video_path}' does not exist. Waiting for updates...")
                        time.sleep(5)
                    cap = cv2.VideoCapture(self.video_path)
                    if not cap.isOpened():
                        print(f"Error: Unable to open updated video file '{self.video_path}'.")
                        return
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

            # Check if the frame is valid before resizing
            if frame.size == 0:
                print("Warning: Captured an empty frame. Skipping...")
                continue

            resized_frame = cv2.resize(frame, (224, 224))
            resized_frame = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))

            if not self.frame_queue.full():
                self.frame_queue.put((resized_frame, frame))

            time.sleep(max(0, frame_time - (time.time() - start_time)))

        cap.release()

    def inference_loop(self):
        while not self.stop_event.is_set():
            try:
                resized_frame, _ = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            start_time = time.time()
            result = self.process_image_with_question(resized_frame, self.question)
            self.result_queue.put(result)
            elapsed_time = time.time() - start_time
            print(f"Elapsed time: {elapsed_time:.2f}s")

    def display_loop(self):
        current_result = None

        while not self.stop_event.is_set():
            try:
                _, original_frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            if not self.result_queue.empty():
                current_result = self.result_queue.get()

            wildfire = current_result.get("wildfire", False) if current_result else False
            text = (
                f"Inference: {current_result}"
                if current_result
                else "Inference: {'wildfire': False}"
            )
            color = (0, 0, 255) if wildfire else (0, 255, 0)

            cv2.putText(
                original_frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
            )
            self.display_frame = original_frame

    def start(self):
        self.stop_event.clear()
        self.threads = [
            threading.Thread(target=self.video_loop, daemon=True),
            threading.Thread(target=self.inference_loop, daemon=True),
            threading.Thread(target=self.display_loop, daemon=True),
        ]
        for thread in self.threads:
            thread.start()

    def stop(self):
        self.stop_event.set()
        for thread in self.threads:
            thread.join()

# FastAPI setup
# Initialize FastAPI app
app = FastAPI()

# Initialize UnifiedProcessor
processor = UnifiedProcessor()
# processor = UnifiedProcessor("wildfire.mp4")

@app.on_event("startup")
def on_startup():
    """Start the UnifiedProcessor when the API starts."""
    threading.Thread(target=processor.start, daemon=True).start()

@app.on_event("shutdown")
def on_shutdown():
    """Stop the UnifiedProcessor when the API stops."""
    processor.stop()

class UpdateVideoPathRequest(BaseModel):
    video_path: str

class UpdatePromptRequest(BaseModel):
    prompt: str

@app.post("/start")
def start_processor():
    if any(thread.is_alive() for thread in processor.threads):
        raise HTTPException(status_code=400, detail="Processor is already running.")
    processor.start()
    return {"status": "success", "message": "Processor started"}

@app.post("/stop")
def stop_processor():
    if not any(thread.is_alive() for thread in processor.threads):
        raise HTTPException(status_code=400, detail="Processor is not running.")
    processor.stop()
    return {"status": "success", "message": "Processor stopped"}

@app.post("/update-video-path")
def update_video_path(request: UpdateVideoPathRequest):
    processor.update_video_path(request.video_path)
    return {"status": "success", "message": f"Video path updated to {request.video_path}"}

@app.post("/update-prompt")
def update_prompt(request: UpdatePromptRequest):
    processor.set_question(request.prompt)
    return {"status": "success", "message": f"Prompt updated to: {request.prompt}"}

@app.websocket("/ws/frames")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint to stream frames in real-time.
    Encodes frames as JPEG and sends them over the WebSocket.
    """
    await websocket.accept()
    try:
        while True:
            if processor.display_frame is not None:
                # Encode the frame as JPEG
                _, encoded_frame = cv2.imencode('.jpg', processor.display_frame)
                # Send the frame as bytes over the WebSocket
                await websocket.send_bytes(encoded_frame.tobytes())
    except WebSocketDisconnect:
        print("Client disconnected.")
    finally:
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
