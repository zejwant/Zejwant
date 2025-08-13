# kafka_connector.py
"""
Kafka Connector for enterprise-grade ingestion pipelines.

Features:
- Connect to Kafka clusters
- Consume messages from topics
- Handle retries and offset management
- Logging and metrics
- Async support
- Returns structured Pandas DataFrame
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import logging
import asyncio
from aiokafka import AIOKafkaConsumer
from aiokafka.helpers import create_ssl_context

# Logging setup
logger = logging.getLogger("kafka_connector")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter(
    '{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
ch.setFormatter(formatter)
logger.addHandler(ch)


class KafkaConnector:
    def __init__(
        self,
        bootstrap_servers: List[str],
        topic: str,
        group_id: str,
        auto_offset_reset: str = "earliest",
        security_protocol: str = "PLAINTEXT",
        ssl_context: Optional[Any] = None,
        retries: int = 5,
    ):
        """
        Initialize Kafka Connector.

        Args:
            bootstrap_servers (List[str]): Kafka broker addresses
            topic (str): Topic to consume
            group_id (str): Consumer group ID
            auto_offset_reset (str): Where to start if no offset ('earliest' or 'latest')
            security_protocol (str): Security protocol (PLAINTEXT, SSL, SASL_PLAINTEXT, etc.)
            ssl_context (Any, optional): SSL context if using SSL
            retries (int): Number of retries on failure
        """
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.auto_offset_reset = auto_offset_reset
        self.security_protocol = security_protocol
        self.ssl_context = ssl_context
        self.retries = retries
        self.consumer: Optional[AIOKafkaConsumer] = None

    async def connect(self):
        """
        Connect to Kafka and start consumer.
        """
        self.consumer = AIOKafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            auto_offset_reset=self.auto_offset_reset,
            security_protocol=self.security_protocol,
            ssl_context=self.ssl_context,
        )
        for attempt in range(1, self.retries + 1):
            try:
                await self.consumer.start()
                logger.info(f"Kafka consumer started for topic: {self.topic}")
                return
            except Exception as e:
                logger.warning(f"Attempt {attempt} failed to connect to Kafka: {e}")
                await asyncio.sleep(2 ** attempt)
        raise ConnectionError("Failed to connect to Kafka after multiple retries")

    async def consume(
        self,
        max_messages: Optional[int] = None,
        timeout_ms: int = 1000
    ) -> pd.DataFrame:
        """
        Consume messages from Kafka and return as Pandas DataFrame.

        Args:
            max_messages (int, optional): Max number of messages to consume
            timeout_ms (int): Poll timeout in milliseconds

        Returns:
            pd.DataFrame: Consumed messages as DataFrame
        """
        if not self.consumer:
            raise RuntimeError("Kafka consumer is not connected. Call `connect()` first.")

        messages: List[Dict[str, Any]] = []
        consumed = 0

        try:
            while True:
                batch = await self.consumer.getmany(timeout_ms=timeout_ms)
                for tp, msgs in batch.items():
                    for msg in msgs:
                        messages.append({
                            "topic": msg.topic,
                            "partition": msg.partition,
                            "offset": msg.offset,
                            "key": msg.key.decode() if msg.key else None,
                            "value": msg.value.decode() if msg.value else None,
                            "timestamp": pd.to_datetime(msg.timestamp, unit='ms')
                        })
                        consumed += 1
                        if max_messages and consumed >= max_messages:
                            break
                    if max_messages and consumed >= max_messages:
                        break
                if max_messages and consumed >= max_messages:
                    break

            df = pd.DataFrame(messages)
            logger.info(f"Kafka messages consumed: {len(df)} rows")
            return df

        except Exception as e:
            logger.error(f"Error consuming Kafka messages: {e}")
            raise

    async def close(self):
        """
        Close the Kafka consumer.
        """
        if self.consumer:
            await self.consumer.stop()
            logger.info(f"Kafka consumer for topic {self.topic} stopped")
          
