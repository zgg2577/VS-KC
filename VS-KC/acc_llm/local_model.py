import torch
from PIL import Image
from typing import Dict, List
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from .config import Config


class LocalModel:
    def __init__(self):
        model_args = {
            "torch_dtype": Config.TORCH_DTYPE,
            "device_map": "auto",  # Or specify a device like f"cuda:{Config.GPU_ID}"
            "trust_remote_code": True,
        }
        # Ensure Config.DEVICE is set correctly (e.g., "cuda" or "cpu")
        if Config.USE_FLASH_ATTN and "cuda" in str(Config.DEVICE).lower():
            model_args["attn_implementation"] = "flash_attention_2"
            print("Using Flash Attention 2")
        else:
            print("Flash Attention 2 not used (either not enabled or not on CUDA)")

        print(f"Loading model {Config.MODEL_NAME}...")
        # 加载基础模型
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            Config.MODEL_NAME, **model_args
        )

        # 加载LoRA适配器
        if Config.LORA_ADAPTER_PATH:
            print(f"Loading LoRA adapter from {Config.LORA_ADAPTER_PATH}...")
            self.model = PeftModel.from_pretrained(
                base_model,
                Config.LORA_ADAPTER_PATH,
                torch_dtype=Config.TORCH_DTYPE,  # Ensure dtype matches base model if needed
            )
            print("LoRA adapter loaded.")
        else:
            self.model = base_model
            print("No LoRA adapter path provided, using base model.")

        # Ensure model is on the correct device after potential PeftModel wrapping
        self.model.to(Config.DEVICE)
        self.model.eval()
        print("Model loaded and set to evaluation mode.")

        print(f"Loading processor for {Config.MODEL_NAME}...")
        self.processor = AutoProcessor.from_pretrained(
            Config.MODEL_NAME,
            min_pixels=Config.MIN_PIXELS,  # Ensure these are in Config
            max_pixels=Config.MAX_PIXELS,  # Ensure these are in Config
        )
        print("Processor loaded.")

    def multi_turn_chat(self, messages: List[Dict]) -> List[Dict]:
        """执行真正的上下文关联多轮对话 (Stage 1 & 2)"""
        # This method remains largely the same as you provided
        # We need to handle potential errors and ensure the returned messages list has the expected structure

        if (
            not messages
            or len(messages) < 1
            or messages[0]["role"] != "user"
            or "content" not in messages[0]
            or len(messages[0]["content"]) < 2
            or messages[0]["content"][0]["type"] != "image"
            or messages[0]["content"][1]["type"] != "text"
        ):
            print("Invalid message structure for multi_turn_chat Stage 1")
            return []  # Return empty list or handle error

        img_path = messages[0]["content"][0]["image"]
        # Make a copy to avoid modifying the original list if errors occur later
        current_messages = messages[:]

        try:
            # --- Process Stage 1 ---
            # Use messages up to the first user message for the template
            input_messages_stage1 = current_messages[
                :1
            ]  # Should only contain the first user message with image and text

            inputs = self.processor(
                text=self.processor.apply_chat_template(
                    input_messages_stage1, tokenize=False, add_generation_prompt=True
                ),  # Add generation prompt
                images=[Image.open(img_path)],
                return_tensors="pt",
            ).to(Config.DEVICE)

            outputs = self.model.generate(
                **inputs,
                **Config.GENERATION_CONFIG,  # Use generation config from Config
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
            reply1 = self._extract_last_response(outputs[0])
            # Add Assistant's Stage 1 response to the message list
            current_messages[1][
                "content"
            ] = reply1  # messages[1] is the assistant placeholder

            # --- Process Stage 2 ---
            # Use messages up to the second user message for the template
            input_messages_stage2 = current_messages[:3]  # User1, Assistant1, User2

            inputs = self.processor(
                text=self.processor.apply_chat_template(
                    input_messages_stage2, tokenize=False, add_generation_prompt=True
                ),  # Add generation prompt
                images=[Image.open(img_path)],  # Reuse the same image
                return_tensors="pt",
            ).to(Config.DEVICE)

            outputs = self.model.generate(
                **inputs,
                **Config.GENERATION_CONFIG,  # Use generation config from Config
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
            reply2 = self._extract_last_response(outputs[0])
            # Add Assistant's Stage 2 response to the message list
            # We need to add a new message for the assistant's second reply
            current_messages.append({"role": "assistant", "content": reply2})

            return current_messages  # Return the updated list with all turns

        except Exception as e:
            print(f"Multi-turn chat failed: {str(e)}")
            # Return the messages list as it was before the error, or an indication of failure
            # Returning the potentially incomplete list might help debugging
            return current_messages  # Or return [] depending on desired error handling

    def get_text_response(self, prompt: str, context: str) -> str:
        """
        Generates a text response based on a prompt and a context text.
        This is a single-turn, text-only interaction, used for the semantic check (Stage 3).
        The prompt is designed to instruct the model to use *only* the provided context.
        """
        try:
            # Construct the input message for the model.
            # Format the prompt to explicitly include the context and the instruction
            # to base the answer ONLY on that context.
            formatted_input_text = f"Context:\n{context.strip()}\n\nBased ONLY on the context provided above, answer the following question: {prompt.strip()}"

            # Create a simple messages list for the chat template (single user turn)
            messages = [{"role": "user", "content": formatted_input_text}]

            # Apply the chat template. No image needed.
            # add_generation_prompt=True tells the processor to add the token(s)
            # that signal the model to start generating the assistant's response.
            input_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Tokenize the input text
            inputs = self.processor(text=input_text, return_tensors="pt").to(
                Config.DEVICE
            )

            # Generate the response
            # We can use a limited max_new_tokens since the expected answer is short ('Yes', 'No', 'Not mentioned').
            # Using a small number here helps prevent the model from generating too much extra text.
            # Ensure you have a suitable Config.GENERATION_CONFIG or pass relevant args directly.
            outputs = self.model.generate(
                **inputs,
                **Config.GENERATION_CONFIG,  # Use generation config from Config
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

            # Decode the output and extract the response
            # The output tensor contains the input text + the generated response.
            # We need to decode the full output and then extract only the part generated by the model.
            full_output = self.processor.decode(outputs[0], skip_special_tokens=True)

            # The apply_chat_template with add_generation_prompt=True usually adds a specific sequence
            # before the assistant's turn. A common pattern for Qwen is "<|im_start|>user\n...\n<|im_end|>\n<|im_start|>assistant\n".
            # Splitting by "assistant" and taking the last part (as in _extract_last_response) should work
            # if the template uses "assistant" as a marker.
            reply = self._extract_last_response(outputs[0])

            return reply.strip()  # Return the cleaned response

        except Exception as e:
            print(f"Text-only response generation failed: {str(e)}")
            # Return the error placeholder defined in Config
            return Config.ERROR_PLACEHOLDER

    def _extract_last_response(self, output_tensor: torch.Tensor) -> str:
        """从完整输出中提取最新回复"""
        # This function extracts the part of the decoded output that comes *after* the last 'assistant' marker.
        # This assumes the chat template used by the processor adds the 'assistant' role label before the model's response.
        # If your chat template uses different markers (like <|im_start|>assistant), you might need to adjust this logic
        # to find the specific start token sequence of the assistant's turn.

        # Decode the entire generated tensor
        full_text = self.processor.decode(output_tensor, skip_special_tokens=True)

        # Split the text by the 'assistant' marker and take the last part
        # This is robust for both single and multi-turn outputs if the template is consistent.
        parts = full_text.split(
            "assistant"
        )  # Or split by '<|im_start|>assistant\n' if that's the exact marker

        # If 'assistant' marker was found, the last part is the final response
        if len(parts) > 1:
            return parts[
                -1
            ].strip()  # Return the part after the last 'assistant' marker

        print(
            f"Warning: 'assistant' marker not found in output: {full_text[:100]}..."
        )  # Log warning
        return ""  # Or return Config.ERROR_PLACEHOLDER
