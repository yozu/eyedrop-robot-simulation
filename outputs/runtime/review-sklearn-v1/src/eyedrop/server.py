"""明示的な通信試験専用。学習済み認識器ではない。"""
import io
import time
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError
from .protocol import IMAGE_SIZE, PROTOCOL_VERSION, Observation, Reply

MAX_IMAGE_BYTES = 512 * 1024
STUB_MODEL = "transport-stub-v1"
app = FastAPI()


@app.get("/health")
def health():
    return {"version": PROTOCOL_VERSION, "mode": "stub", "model": STUB_MODEL,
            "device": "none", "image_size": IMAGE_SIZE}


@app.post("/infer", response_model=Reply)
async def infer(metadata: str = Form(), image: UploadFile = File()):
    started = time.perf_counter()
    try:
        observation = Observation.model_validate_json(metadata)
        content = await image.read(MAX_IMAGE_BYTES + 1)
        if len(content) > MAX_IMAGE_BYTES:
            raise ValueError("画像サイズ上限超過")
        with Image.open(io.BytesIO(content)) as decoded:
            if decoded.size != IMAGE_SIZE or decoded.mode != "RGB" or decoded.format != "PNG":
                raise ValueError("256×256 RGB PNGが必要です")
            decoded.load()
    except (ValidationError, ValueError, UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return Reply(**observation.model_dump(), model=STUB_MODEL, mode="stub",
                 position=(0.5, 0.5), eyelid=(0.0, 0.0, 1.0), visible=0.0,
                 service_ms=(time.perf_counter() - started) * 1000)
