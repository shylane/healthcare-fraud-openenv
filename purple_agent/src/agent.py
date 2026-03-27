"""
HealthcareFraudAgent — runs the GSPO-trained Qwen3-1.7B model.

Receives a claim prompt from the green agent and returns a structured
fraud detection decision using the trained LoRA adapter.
"""

import logging
from typing import Optional

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TextPart
from a2a.utils import new_agent_text_message

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert healthcare insurance fraud investigator.
Analyze the claim and provide your assessment in this exact format:

<think>
[Your step-by-step reasoning about fraud indicators]
</think>

Decision: [APPROVE|FLAG_REVIEW|INVESTIGATE|DENY|REQUEST_INFO]
Rationale: [2-3 sentence explanation]
Evidence: [Specific data points that influenced your decision]
Recommendation: [Optional follow-up action]"""


def _extract_text(message: Message) -> str:
    for part in message.parts or []:
        if isinstance(part.root, TextPart):
            return part.root.text
    return ""


class HealthcareFraudAgent:
    """
    Wraps the GSPO-trained Qwen3-1.7B + LoRA adapter for inference.
    Model is loaded lazily on first call to avoid OOM on startup.
    """

    def __init__(self, model_path: str = "/app/model"):
        self.model_path = model_path
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy load: only runs on first inference call."""
        if self._model is not None:
            return

        logger.info(f"Loading model from {self.model_path}...")
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_path,
                max_seq_length=2048,
                load_in_4bit=True,
                dtype=None,
            )
            FastLanguageModel.for_inference(model)
            self._model = model
            self._tokenizer = tokenizer
            logger.info("Model loaded successfully")
        except ImportError:
            # Fallback: load with transformers directly (no unsloth in env)
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            logger.info("Model loaded via transformers (no unsloth)")

    def _generate(self, prompt: str) -> str:
        """Run inference and return the model's response text."""
        import torch

        self._load_model()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        # Apply chat template
        input_text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,  # Qwen3 thinking mode
        )

        inputs = self._tokenizer(input_text, return_tensors="pt").to(
            next(self._model.parameters()).device
        )

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.6,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Strip <think>...</think> from the output for the environment
        if "</think>" in response:
            response = response.split("</think>", 1)[1].strip()

        return response

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        claim_prompt = _extract_text(message)

        if not claim_prompt:
            response = "Decision: APPROVE\nRationale: Empty prompt received.\nEvidence: N/A"
        else:
            # Run inference (synchronous — model is not async)
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self._generate, claim_prompt)

        await updater.add_artifact(
            [TextPart(text=response)],
            name="fraud_decision",
        )
