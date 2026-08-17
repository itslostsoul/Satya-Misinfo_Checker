import asyncio
from pipelines.image_pipeline import run_image_pipeline
from pipelines.text_pipeline import run_text_pipeline

async def classify_and_route(text: str, image_bytes: bytes) -> dict:
    tasks = []
    keys = []

    if image_bytes:
        tasks.append(run_image_pipeline(image_bytes))
        keys.append("image_result")
    if text:
        tasks.append(run_text_pipeline(text))
        keys.append("text_result")

    results = await asyncio.gather(*tasks)

    return {keys[i]: results[i] for i in range(len(keys))}
