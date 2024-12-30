import cv2
import asyncio
import websockets
import numpy as np

# WebSocket URL of the FastAPI server
WEBSOCKET_URL = "ws://localhost:8000/ws/frames"


async def display_frames():
    """
    Connects to the WebSocket server and displays frames received in real-time.
    """
    async with websockets.connect(WEBSOCKET_URL) as websocket:
        while True:
            # Receive the frame bytes from the WebSocket
            frame_bytes = await websocket.recv()

            # Decode the JPEG bytes into an OpenCV image
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Display the frame using OpenCV
            cv2.imshow("Real-Time Frame", frame)

            # Break the loop if the user presses 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release resources
    cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(display_frames())
