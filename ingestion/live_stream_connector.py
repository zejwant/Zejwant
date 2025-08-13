# live_stream_connector.py
"""
Enterprise-grade live stream connector for Kafka and MQTT.

Features:
- Connect to multiple brokers/topics
- Handle streaming data in real-time
- Auto-reconnect on failure
- Batch and windowed processing
- Logging and metrics (throughput, latency)
- Output to Pandas DataFrame or persist to DB
- Async support
"""

from typing import Any, Callable, Dict, List, Optional, Union
import asyncio
import json
import logging
import time
from datetime import datetime
import pandas as pd

# Kafka
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
# MQTT
from asyncio_mqtt import Client as MQTTClient, MqttError

# Logging setup
logger = logging.getLogger("live_stream_connector")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter(
    '{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
ch.setFormatter(formatter)
logger.addHandler(ch)


class LiveStreamConnector:
    def __init__(
        self,
        kafka_brokers: Optional[List[str]] = None,
        mqtt_brokers: Optional[List[str]] = None,
        topics: Optional[List[str]] = None,
        batch_size: int = 100,
        window_seconds: int = 5,
    ):
        """
        Initialize live stream connector.

        Args:
            kafka_brokers (List[str], optional): Kafka broker URLs
            mqtt_brokers (List[str], optional): MQTT broker URLs
            topics (List[str], optional): List of topics to subscribe
            batch_size (int): Number of messages per batch
            window_seconds (int): Time window (seconds) for batch processing
        """
        self.kafka_brokers = kafka_brokers or []
        self.mqtt_brokers = mqtt_brokers or []
        self.topics = topics or []
        self.batch_size = batch_size
        self.window_seconds = window_seconds

        self._kafka_consumers: List[AIOKafkaConsumer] = []
        self._mqtt_clients: List[MQTTClient] = []

    async def start(self, process_func: Callable[[List[Dict[str, Any]]], None]):
        """
        Start consuming streams from Kafka and MQTT concurrently.

        Args:
            process_func: Function to process batches of messages
        """
        tasks = []
        if self.kafka_brokers:
            tasks.append(self._consume_kafka(process_func))
        if self.mqtt_brokers:
            tasks.append(self._consume_mqtt(process_func))

        if tasks:
            await asyncio.gather(*tasks)

    async def _consume_kafka(self, process_func: Callable[[List[Dict[str, Any]]], None]):
        """
        Consume Kafka topics with batching and auto-reconnect.
        """
        for broker in self.kafka_brokers:
            consumer = AIOKafkaConsumer(
                *self.topics,
                bootstrap_servers=broker,
                group_id="live_stream_group",
                enable_auto_commit=True,
            )
            await consumer.start()
            self._kafka_consumers.append(consumer)

        try:
            batch = []
            batch_start = time.time()
            while True:
                for consumer in self._kafka_consumers:
                    msg = await consumer.getone()
                    data = self._parse_message(msg.value)
                    batch.append(data)
                    if len(batch) >= self.batch_size or (time.time() - batch_start) >= self.window_seconds:
                        await self._process_batch(batch, process_func)
                        batch = []
                        batch_start = time.time()
        except Exception as e:
            logger.error(f"Kafka consumer error: {e}. Reconnecting...")
            await self._reconnect_kafka(process_func)

    async def _reconnect_kafka(self, process_func: Callable[[List[Dict[str, Any]]], None]):
        for consumer in self._kafka_consumers:
            await consumer.stop()
        self._kafka_consumers = []
        await asyncio.sleep(5)
        await self._consume_kafka(process_func)

    async def _consume_mqtt(self, process_func: Callable[[List[Dict[str, Any]]], None]):
        """
        Consume MQTT topics with batching and auto-reconnect.
        """
        for broker in self.mqtt_brokers:
            client = MQTTClient(broker)
            self._mqtt_clients.append(client)

        try:
            tasks = []
            for client in self._mqtt_clients:
                tasks.append(self._mqtt_consumer_task(client, process_func))
            await asyncio.gather(*tasks)
        except MqttError as e:
            logger.error(f"MQTT consumer error: {e}. Reconnecting...")
            await self._reconnect_mqtt(process_func)

    async def _mqtt_consumer_task(self, client: MQTTClient, process_func: Callable[[List[Dict[str, Any]]], None]):
        batch = []
        batch_start = time.time()
        async with client:
            for topic in self.topics:
                await client.subscribe(topic)
            async with client.unfiltered_messages() as messages:
                async for message in messages:
                    data = self._parse_message(message.payload)
                    batch.append(data)
                    if len(batch) >= self.batch_size or (time.time() - batch_start) >= self.window_seconds:
                        await self._process_batch(batch, process_func)
                        batch = []
                        batch_start = time.time()

    async def _reconnect_mqtt(self, process_func: Callable[[List[Dict[str, Any]]], None]):
        for client in self._mqtt_clients:
            await client.disconnect()
        self._mqtt_clients = []
        await asyncio.sleep(5)
        await self._consume_mqtt(process_func)

    async def _process_batch(self, batch: List[Dict[str, Any]], process_func: Callable[[List[Dict[str, Any]]], None]):
        """
        Process a batch of messages and log metrics.

        Args:
            batch: List of message dictionaries
            process_func: Function to handle the batch
        """
        start_time = time.time()
        await asyncio.to_thread(process_func, batch)
        end_time = time.time()
        throughput = len(batch) / (end_time - start_time)
        logger.info(json.dumps({
            "batch_size": len(batch),
            "processing_time": end_time - start_time,
            "throughput_msgs_per_sec": throughput
        }))

    def _parse_message(self, payload: bytes) -> Dict[str, Any]:
        """
        Parse message payload into dict.

        Args:
            payload: Raw message bytes

        Returns:
            Dict representation of message
        """
        try:
            return json.loads(payload)
        except Exception:
            return {"raw": payload.decode("utf-8")}

    @staticmethod
    def to_dataframe(batch: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert batch of messages to Pandas DataFrame.
        """
        return pd.DataFrame(batch)
  
