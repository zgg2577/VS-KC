import os
import time
import base64
import requests
from typing import List, Dict, Optional, Any
from pathlib import Path
from .config import Config


class APIClient:
    """API client for making requests to vision language models"""

    def __init__(self):
        self.endpoint = Config.API_ENDPOINT
        self.api_key = Config.API_KEY
        self.model = Config.API_MODEL
        self.timeout = Config.API_TIMEOUT
        self.max_retries = Config.MAX_RETRIES
        self.retry_delay = Config.RETRY_DELAY

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _prepare_headers(self) -> Dict[str, str]:
        """Prepare request headers with authentication"""
        # 准备带有身份验证的请求头
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _prepare_payload(self, messages: List[Dict]) -> Dict:
        """Prepare request payload with messages and generation config"""
        # Process messages to handle image content
        processed_messages = []

        for message in messages:
            if message["role"] == "assistant" and message["content"] is None:
                # Skip unprocessed assistant messages
                continue

            processed_content = []

            if isinstance(message["content"], list):
                for content_item in message["content"]:
                    if content_item["type"] == "image":
                        # Convert image path to base64
                        processed_content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{self._encode_image(content_item['image'])}"
                                },
                            }
                        )
                    else:
                        processed_content.append(content_item)

                processed_message = {
                    "role": message["role"],
                    "content": processed_content,
                }
            else:
                # Handle string content or None
                processed_message = {
                    "role": message["role"],
                    "content": message["content"],
                }

            processed_messages.append(processed_message)

        # Prepare complete payload
        payload = {
            "model": self.model,
            "messages": processed_messages,
            **Config.API_GENERATION_CONFIG,
        }

        return payload

    def _make_api_request(self, payload: Dict) -> Dict:
        """Make API request with retry logic"""
        headers = self._prepare_headers()

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.endpoint}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )

                response.raise_for_status()
                return response.json()

            except requests.RequestException as e:
                print(
                    f"API request failed (attempt {attempt+1}/{self.max_retries}): {str(e)}"
                )

                if attempt < self.max_retries - 1:
                    sleep_time = self.retry_delay * (2**attempt)  # Exponential backoff
                    print(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    print("Max retries reached. Giving up.")
                    raise

    def multi_turn_chat(self, messages: List[Dict]) -> List[Dict]:
        """
        Execute a multi-turn chat conversation with the LLM

        Parameters:
            messages: List of message objects in the format:
                [{"role": "user", "content": [{"type": "image", "image": "path"}, {"type": "text", "text": "prompt"}]}, ...]

        Returns:
            List of messages including model responses
        """
        result_conversation = []

        # First round: Get response to the initial image query
        first_user_message = messages[0]
        result_conversation.append(first_user_message)

        try:
            # Make API request for first round
            first_payload = self._prepare_payload([first_user_message])
            first_response = self._make_api_request(first_payload)

            # Extract the assistant's response
            first_assistant_content = (
                first_response.get("choices", [{}])[0]
                .get("message", {})
                .get("content", Config.ERROR_PLACEHOLDER)
            )
            first_assistant_message = {
                "role": "assistant",
                "content": first_assistant_content,
            }

            result_conversation.append(first_assistant_message)

            # Second round: Send the follow-up question
            second_user_message = messages[2]  # This is the follow-up question
            result_conversation.append(second_user_message)

            # Prepare full conversation history for second round
            second_payload = self._prepare_payload(
                [first_user_message, first_assistant_message, second_user_message]
            )

            # Make API request for second round
            second_response = self._make_api_request(second_payload)

            # Extract the assistant's second response
            second_assistant_content = (
                second_response.get("choices", [{}])[0]
                .get("message", {})
                .get("content", Config.ERROR_PLACEHOLDER)
            )
            second_assistant_message = {
                "role": "assistant",
                "content": second_assistant_content,
            }

            result_conversation.append(second_assistant_message)

            return result_conversation

        except Exception as e:
            print(f"Error in multi-turn chat: {str(e)}")
            # Return partial conversation with error placeholder if needed
            if len(result_conversation) == 1:
                result_conversation.append(
                    {"role": "assistant", "content": Config.ERROR_PLACEHOLDER}
                )
                result_conversation.append(messages[2])
                result_conversation.append(
                    {"role": "assistant", "content": Config.ERROR_PLACEHOLDER}
                )
            elif len(result_conversation) == 3:
                result_conversation.append(
                    {"role": "assistant", "content": Config.ERROR_PLACEHOLDER}
                )

            return result_conversation

    # 在 APIClient 类中新增方法

    def get_text_response(self, prompt: str, context: str) -> str:

        try:
            # 构造与LocalModel一致的输入格式
            formatted_input = f"{context}\n\n{prompt}"  # 保持与LocalModel相同的拼接方式

            # 构建纯文本消息
            messages = [{"role": "user", "content": formatted_input}]

            payload = self._prepare_payload(messages)
            response = self._make_api_request(payload)

            # 提取模型回复
            return (
                response.get("choices", [{}])[0]
                .get("message", {})
                .get("content", Config.ERROR_PLACEHOLDER)
            )

        except Exception as e:
            print(f"Error in get_text_response: {str(e)}")
            return Config.ERROR_PLACEHOLDER

            # 在 APIClient 类中新增方法
