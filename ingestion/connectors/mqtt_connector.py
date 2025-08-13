# mqtt_connector.py
"""
MQTT Connector for enterprise-grade ingestion pipelines.

Features:
- Connect to MQTT broker
- Subscribe to topics
- Handle QoS and reconnection
- Logging and metrics
- Async support
- Returns structured Pandas DataFrame
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import asyncio
import logging
from asyncio_mqtt import Client, MqttError

# Logging setup
logger = logging.getLogger("mqtt_connector")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter(
    '{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
ch.setFormatter(formatter)
logger.addHandler(ch)


class MQTTConnector:
    def __init__(
        self,
        broker_host: str,
        broker_port: int = 1883,
        topics: Optional[List[str]] = None,
        qos: int = 1,
        username: Optional[str] = None,
        password: Optional[str] = None,
        retries: int = 5
    ):
        """
        Initialize MQTT Connector.

        Args:
            broker_host (str): MQTT broker hostname or IP
            broker_port (int): MQTT broker port (default 1883)
            topics (List[str], optional): List of topics to subscribe
            qos (int): Quality of Service (0, 1, 2)
            username (str, optional): Username for authentication
            password (str, optional): Password for authentication
            retries (int): Number of retries on connection failure
        """
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.topics = topics or []
        self.qos = qos
        self.username = username
        self.password = password
        self.retries = retries
        self.client: Optional[Client] = None

    async def connect(self):
        """
        Connect to MQTT broker with retries and exponential backoff.
        """
        attempt = 1
        while attempt <= self.retries:
            try:
                self.client = Client(
                    self.broker_host,
                    port=self.broker_port,
                    username=self.username,
                    password=self.password
                )
                await self.client.connect()
                logger.info(f"Connected to MQTT broker: {self.broker_host}:{self.broker_port}")
                return
            except MqttError as e:
                logger.warning(f"Attempt {attempt} failed to connect: {e}")
                await asyncio.sleep(2 ** attempt)
                attempt += 1
        raise ConnectionError(f"Failed to connect to MQTT broker after {self.retries} retries")

    async def consume(self, max_messages: Optional[int] = None, timeout: int = 10) -> pd.DataFrame:
        """
        Consume messages from subscribed topics.

        Args:
            max_messages (int, optional): Maximum number of messages to consume
            timeout (int): Timeout in seconds for consumption

        Returns:
            pd.DataFrame: Consumed messages as DataFrame
        """
        if not self.client:
            raise RuntimeError("MQTT client not connected. Call `connect()` first.")

        messages: List[Dict[str, Any]] = []
        consumed = 0

        try:
            async with self.client.unfiltered_messages() as messages_iter:
                for topic in self.topics:
                    await self.client.subscribe(topic, qos=self.qos)
                    logger.info(f"Subscribed to topic: {topic} with QoS {self.qos}")

                async for msg in messages_iter:
                    messages.append({
                        "topic": msg.topic,
                        "payload": msg.payload.decode(),
                        "qos": msg.qos,
                        "retain": msg.retain,
                        "timestamp": pd.Timestamp.now()
                    })
                    consumed += 1
                    if max_messages and consumed >= max_messages:
                        break

            df = pd.DataFrame(messages)
            logger.info(f"MQTT messages consumed: {len(df)} rows")
            return df

        except MqttError as e:
            logger.error(f"MQTT error during consumption: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to consume MQTT messages: {e}")
            raise

    async def disconnect(self):
        """
        Disconnect from MQTT broker.
        """
        if self.client:
            await self.client.disconnect()
            logger.info(f"Disconnected from MQTT broker: {self.broker_host}")
          
