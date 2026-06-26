import json

from shared.config import RABBITMQ_URL, SCORING_QUEUE


async def publish_scoring_job(request_id: int) -> None:
    import aio_pika

    connection = await aio_pika.connect_robust(RABBITMQ_URL)
    async with connection:
        channel = await connection.channel()
        await channel.declare_queue(SCORING_QUEUE, durable=True)
        body = json.dumps({"request_id": request_id}).encode()
        await channel.default_exchange.publish(
            aio_pika.Message(body=body, delivery_mode=aio_pika.DeliveryMode.PERSISTENT),
            routing_key=SCORING_QUEUE,
        )
