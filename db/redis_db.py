import json
import os
from typing import Any, Callable, Optional
import redis
from dotenv import load_dotenv

from logger import get_logger

logger = get_logger(__name__)
load_dotenv()


class RedisDB:
    def __init__(
        self,
        host,
        port,
        channel: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """Initializes the Redis client.

        Args:
            host: Redis server host (defaults to REDIS_HOST from .env)
            port: Redis server port (defaults to REDIS_PORT from .env)
            channel: Default channel name (defaults to REDIS_CHANNEL from .env)
            username: Redis username (defaults to REDIS_USERNAME from .env)
            password: Redis password (defaults to REDIS_PASSWORD from .env)
        """

        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                decode_responses=True,
                username=username,
                password=password,
            )
            self.channel = channel
            # Test connection
            self.client.ping()
            logger.info(f"Connected to Redis at {host}:{port}")
        except redis.ConnectionError as e:
            logger.exception(f"Failed to connect to Redis: {e}")
            raise

    def publish(self, message: Any, channel: Optional[str] = None) -> int:
        """Publishes a message to a Redis channel.

        Args:
            message: Message to publish (will be JSON serialized if dict/list)
            channel: Channel name (defaults to self.channel)

        Returns:
            Number of subscribers that received the message
        """
        target_channel = channel or self.channel

        # Serialize message if it's a dict or list
        if isinstance(message, (dict, list)):
            message = json.dumps(message)

        try:
            result = self.client.publish(target_channel, message)
            logger.info(f"Published message to '{target_channel}': {message[:100]}...")
            return result
        except Exception as e:
            logger.exception(f"Error publishing to '{target_channel}': {e}")
            raise

    def subscribe(self, callback: Callable[[str], None], channel: Optional[str] = None):
        """Subscribes to a Redis channel and processes incoming messages. This
        is a blocking operation.

        Args:
            callback: Function to call with each received message
            channel: Channel name (defaults to self.channel)
        """
        target_channel = channel or self.channel

        try:
            pubsub = self.client.pubsub()
            pubsub.subscribe(target_channel)
            logger.info(f"Subscribed to channel '{target_channel}'")

            for message in pubsub.listen():
                if message["type"] == "message":
                    data = message["data"]
                    logger.debug(f"Received message from '{target_channel}': {data}")

                    # Try to parse JSON if possible
                    try:
                        data = json.loads(data)
                    except (json.JSONDecodeError, TypeError):
                        pass  # Keep as string if not JSON

                    callback(data)
        except Exception as e:
            logger.exception(f"Error subscribing to '{target_channel}': {e}")
            raise

    def subscribe_multiple(self, callback: Callable[[str, str], None], channels: list):
        """Subscribes to multiple Redis channels. This is a blocking operation.

        Args:
            callback: Function to call with (channel, message) for each received message
            channels: List of channel names to subscribe to
        """
        try:
            pubsub = self.client.pubsub()
            pubsub.subscribe(*channels)
            logger.info(f"Subscribed to channels: {', '.join(channels)}")

            for message in pubsub.listen():
                if message["type"] == "message":
                    channel = message["channel"]
                    data = message["data"]
                    logger.debug(f"Received message from '{channel}': {data}")

                    # Try to parse JSON if possible
                    try:
                        data = json.loads(data)
                    except (json.JSONDecodeError, TypeError):
                        pass  # Keep as string if not JSON

                    callback(channel, data)
        except Exception as e:
            logger.exception(f"Error subscribing to channels: {e}")
            raise

    def set(self, key: str, value: Any, expiry: Optional[int] = None):
        """Sets a key-value pair in Redis.

        Args:
            key: Key name
            value: Value to store (will be JSON serialized if dict/list)
            expiry: Optional expiry time in seconds
        """
        if isinstance(value, (dict, list)):
            value = json.dumps(value)

        try:
            if expiry:
                self.client.setex(key, expiry, value)
            else:
                self.client.set(key, value)
            logger.debug(f"Set key '{key}' with value: {str(value)[:100]}...")
        except Exception as e:
            logger.exception(f"Error setting key '{key}': {e}")
            raise

    def get(self, key: str) -> Optional[Any]:
        """Gets a value from Redis by key.

        Args:
            key: Key name

        Returns:
            Value (parsed from JSON if possible), or None if key doesn't exist
        """
        try:
            value = self.client.get(key)
            if value is None:
                return None

            # Try to parse JSON
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        except Exception as e:
            logger.exception(f"Error getting key '{key}': {e}")
            raise

    def close(self):
        """Closes the Redis connection."""
        if self.client:
            self.client.close()
            logger.info("Redis connection closed.")


if __name__ == "__main__":
    # Example Usage
    try:
        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", 6379))
        channel = os.getenv("REDIS_CHANNEL", "ai-service")
        user = os.getenv("REDIS_USER", "default")
        password = os.getenv("REDIS_PASSWORD", "")

        redis_db = RedisDB(host, port, channel, user, password)

        # Test publish
        redis_db.publish({"message": "Hello from Redis!", "type": "test"})

        # Test key-value operations
        redis_db.set("test_key", {"data": "test_value"})
        result = redis_db.get("test_key")
        logger.info(f"Retrieved value: {result}")

        # Test subscribe (this will block, so comment out for normal testing)
        # def message_handler(message):
        #     logger.info(f"Received: {message}")
        #
        # redis_db.subscribe(message_handler)

    except Exception as e:
        logger.exception(f"Redis test failed: {e}")
