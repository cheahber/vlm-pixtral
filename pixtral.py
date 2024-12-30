import cv2
import threading
import queue
import time
import numpy as np
import json
import re
from PIL import Image
from pathlib import Path
from optimum.intel.openvino import OVModelForVisualCausalLM
from transformers import AutoProcessor
from gradio_helper import chat_template

class UnifiedProcessor:
    MODEL_REPO = "mistral-community/pixtral-12b"
    MODEL_BASE_DIR = Path("pixtral-12b")
    PRECISION = "INT4"
    DEVICE = "AUTO"

    def __init__(self, video_path):
        # Video-related attributes
        self.video_path = video_path
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.threads = []

        # Default question for inference
        self.question = (
            "Provide a direct and concise answer in JSON format. "
            "The JSON should include key: 'wildfire'. "
            "The 'wildfire' key must have a boolean value indicating whether a wildfire is detected. "
        )

        # Display frame placeholder
        self.display_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Model-related attributes
        self.model_dir = self.MODEL_BASE_DIR / self.PRECISION
        self.processor = AutoProcessor.from_pretrained(self.model_dir)
        self.ov_model = OVModelForVisualCausalLM.from_pretrained(
            self.model_dir, device=self.DEVICE.lower()
        )
        # If no chat template is configured, attach one
        if not self.processor.chat_template:
            self.processor.set_chat_template(chat_template)

    def update_video_path(self, new_video_path):
        self.stop()
        self.video_path = new_video_path
        self.start()

    def set_question(self, question):
        self.stop()
        self.question = question
        self.start()

    def process_image_with_question(self, image, question):
        """Processes an image by sending it to the model along with a question."""
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
            **inputs, do_sample=False, max_new_tokens=100
        )
        output = self.processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
        json_match = re.search(r'```json\s*({.*?})\s*```', output, re.DOTALL)
        return json.loads(json_match.group(1)) if json_match else {}

    def video_loop(self):
        """Continuously read frames from the video and enqueue them for inference."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error: Unable to open video file.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_time = 1 / fps

        while not self.stop_event.is_set():
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                # If video ends or cannot be read, restart from beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            resized_frame = cv2.resize(frame, (224, 224))
            resized_frame = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))

            if not self.frame_queue.full():
                self.frame_queue.put((resized_frame, frame))

            # Sleep for frame_time adjusted by how much processing time has passed
            time.sleep(max(0, frame_time - (time.time() - start_time)))

        cap.release()

    def inference_loop(self):
        """Continuously pop frames from the queue, run inference, and put the result in another queue."""
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
        """Continuously pop frames (for display) and overlay inference results."""
        current_result = None

        while not self.stop_event.is_set():
            try:
                # Retrieve a frame from the queue
                _, original_frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            # Check for new inference results
            if not self.result_queue.empty():
                current_result = self.result_queue.get()

            # Default inference result if no result available
            wildfire = current_result.get("wildfire", False) if current_result else False
            text = (
                f"Inference: {current_result}"
                if current_result
                else "Inference: {'wildfire': False}"
            )
            color = (0, 0, 255) if wildfire else (0, 255, 0)

            # Overlay text on the frame
            cv2.putText(
                original_frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA
            )

            self.display_frame = original_frame
            print(self.display_frame)
            # cv2.imshow("Video", original_frame)
            #
            # # Handle keypress to stop display loop
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     self.stop_event.set()
            #     break  # Exit loop if 'q' is pressed

        # # Release OpenCV resources after exiting the loop
        # cv2.destroyAllWindows()
        # cv2.waitKey(1)  # Ensure all OpenCV windows are closed

    def start(self):
        """Launches the threads for video reading, inference, and display."""
        self.stop_event.clear()
        self.threads = [
            threading.Thread(target=self.video_loop, daemon=True),
            threading.Thread(target=self.inference_loop, daemon=True),
            threading.Thread(target=self.display_loop, daemon=True),
        ]
        for thread in self.threads:
            thread.start()

    def stop(self):
        """Signals all threads to stop and waits for them to terminate."""
        self.stop_event.set()
        for thread in self.threads:
            thread.join()

if __name__ == "__main__":
    processor = UnifiedProcessor("wildfire.mp4")
    try:
        processor.start()
        time.sleep(10)  # Let it run for 10 seconds
        processor.stop()
        time.sleep(5)  # Pause before restarting
        processor.start()
        time.sleep(10)  # Let it run for another 10 seconds
    finally:
        processor.stop()
