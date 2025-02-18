import json
import asyncio
import websockets
import json

async def send_result(result):
    uri = "ws://localhost:8288"
    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps(result, ensure_ascii=False))
        # print(await websocket.recv())
    