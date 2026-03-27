"""
Purple Agent Server — Healthcare Claims Fraud RL Model
Serves the GSPO-trained Qwen3-1.7B LoRA model as an A2A endpoint.

The purple agent receives a claim prompt (text) and returns a fraud
detection decision in the format expected by the environment:
  Decision: APPROVE|FLAG_REVIEW|INVESTIGATE|DENY|REQUEST_INFO
  Rationale: ...
  Evidence: ...
"""

import argparse
import logging

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from executor import Executor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Healthcare Fraud Purple Agent")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9009)
    parser.add_argument("--model-path", default="/app/model",
                        help="Path to merged model weights OR HF model ID")
    parser.add_argument("--card-url", default=None)
    args = parser.parse_args()

    base_url = args.card_url or f"http://{args.host}:{args.port}/"

    skill = AgentSkill(
        id="fraud_detection",
        name="Healthcare Fraud Detection",
        description=(
            "Given a healthcare insurance claim prompt, decides whether to "
            "APPROVE, FLAG_REVIEW, INVESTIGATE, DENY, or REQUEST_INFO. "
            "Trained via GSPO reinforcement learning on synthetic claim data."
        ),
        tags=["healthcare", "fraud-detection", "rl-agent"],
        examples=["Analyze claim #CLM-12345 for fraud indicators"],
        input_modes=["text"],
        output_modes=["text"],
    )

    agent_card = AgentCard(
        name="Healthcare Fraud Detection RL Agent",
        description=(
            "GSPO-trained Qwen3-1.7B agent for healthcare insurance fraud detection. "
            "Trained over 5 RL cycles using the OpenEnv Healthcare Claims environment."
        ),
        url=base_url,
        version="0.1.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(model_path=args.model_path),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    logger.info(f"Purple agent loading model from: {args.model_path}")
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
