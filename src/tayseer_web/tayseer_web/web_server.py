#!/usr/bin/env python3
import asyncio
import json
from typing import Set
import threading
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles  # <-- for serving static files
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory  # <-- ROS2 utility
import os

app = FastAPI()
ws_clients: Set[WebSocket] = set()
ros_node = None
executor = None
ros_thread = None

def load_html() -> str:
    html_path = os.path.join(
        get_package_share_directory('tayseer_web'),
        'web',
        'index.html'
    )
    with open(html_path, 'r', encoding='utf-8') as f:
        return f.read()

HTML_CONTENT = load_html()


class WebBridge(Node):
    def __init__(self, loop):
        super().__init__('web_bridge')
        self.loop = loop
        
        self.prompt_pub = self.create_publisher(String, '/user_prompt', 10)
        self.create_subscription(String, '/commander_status', self.status_cb, 10)
        self.create_subscription(String, '/world_state', self.world_state_cb, 10)
        self.create_subscription(String, '/commander_plan', self.plan_cb, 10)
        self.create_subscription(String, '/chat_message', self.chat_cb, 10)
        
        self.get_logger().info("Web Bridge ready")

    def publish_message(self, text: str):
        msg = String()
        msg.data = text
        self.prompt_pub.publish(msg)

    def status_cb(self, msg):
        data = json.dumps({"type": "status", "data": json.loads(msg.data)})
        asyncio.run_coroutine_threadsafe(broadcast(data), self.loop)

    def world_state_cb(self, msg):
        data = json.dumps({"type": "world_state", "data": json.loads(msg.data)})
        asyncio.run_coroutine_threadsafe(broadcast(data), self.loop)

    def plan_cb(self, msg):
        data = json.dumps({"type": "plan", "data": json.loads(msg.data)})
        asyncio.run_coroutine_threadsafe(broadcast(data), self.loop)

    def chat_cb(self, msg):
        data = json.dumps({"type": "chat", "data": json.loads(msg.data)})
        asyncio.run_coroutine_threadsafe(broadcast(data), self.loop)


async def broadcast(message: str):
    disconnected = set()
    for ws in ws_clients:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.add(ws)
    for ws in disconnected:
        ws_clients.discard(ws)


@app.on_event("startup")
async def startup():
    global loop
    loop = asyncio.get_running_loop()
    rclpy.init()
    global ros_node
    ros_node = WebBridge(loop)
    global executor
    executor = MultiThreadedExecutor()
    executor.add_node(ros_node)
    global ros_thread
    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()


@app.on_event("shutdown")
async def shutdown():
    executor.shutdown()
    ros_node.destroy_node()
    rclpy.shutdown()


@app.get("/")
async def root():
    return HTMLResponse(HTML_CONTENT)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    ws_clients.add(websocket)
    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            if msg.get("type") == "message":
                ros_node.publish_message(msg["data"])
                await websocket.send_text(json.dumps({"type": "ack", "data": "Sent"}))
    except Exception:
        pass
    finally:
        ws_clients.discard(websocket)


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == '__main__':
    main()