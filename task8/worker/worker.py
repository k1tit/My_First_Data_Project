import asyncio
import json
import logging

import aio_pika

from shared.config import RABBITMQ_URL, SCORING_QUEUE, MODEL_VERSION
from shared.db import SessionLocal, init_db
from shared.models import ScoringRequest, ScoringResult
from shared.scoring_engine import ScoringEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scoring-worker")

engine = ScoringEngine()


def process_job(request_id: int) -> None:
    with SessionLocal() as session:
        req = session.get(ScoringRequest, request_id)
        if not req:
            logger.warning("request %s not found", request_id)
            return
        req.status = "processing"
        session.commit()

        try:
            payload = json.loads(req.payload_json)
            raw, calibrated, level = engine.predict(payload)
            req.status = "completed"
            req.result = ScoringResult(
                request_id=req.id,
                probability=raw,
                probability_calibrated=calibrated,
                risk_level=level,
            )
            req.model_version = MODEL_VERSION
            session.commit()
            logger.info("request %s scored: %.4f (%s)", request_id, calibrated, level)
        except Exception as exc:
            req.status = "failed"
            req.error_message = str(exc)
            session.commit()
            logger.exception("request %s failed", request_id)


async def on_message(message: aio_pika.IncomingMessage) -> None:
    async with message.process():
        data = json.loads(message.body.decode())
        request_id = int(data["request_id"])
        await asyncio.to_thread(process_job, request_id)


async def main() -> None:
    logger.info("worker starting, queue=%s", SCORING_QUEUE)
    init_db()
    engine.load()
    connection = await aio_pika.connect_robust(RABBITMQ_URL)
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=1)
    queue = await channel.declare_queue(SCORING_QUEUE, durable=True)
    await queue.consume(on_message)
    logger.info("worker ready")
    await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
