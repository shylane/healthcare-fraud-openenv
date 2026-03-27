"""
PurpleAgentClient — A2A client that calls the purple agent for claim decisions.

The purple agent (our trained RL model) exposes an A2A endpoint. We send it
the claim prompt and receive back a text decision response.
"""

import logging
from uuid import uuid4

import httpx
from a2a.client import A2AClient, A2ACardResolver
from a2a.types import MessageSendParams, SendMessageRequest
from a2a.utils import new_agent_text_message

logger = logging.getLogger(__name__)

# Timeout for purple agent responses (model inference can be slow)
PURPLE_AGENT_TIMEOUT = 60.0


class PurpleAgentClient:
    """
    Async A2A client for calling the purple agent.

    Wraps a2a.client.A2AClient with healthcare-specific helpers.
    """

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._http_client = httpx.AsyncClient(timeout=PURPLE_AGENT_TIMEOUT)
        self._a2a_client: A2AClient | None = None

    async def _get_client(self) -> A2AClient:
        if self._a2a_client is None:
            resolver = A2ACardResolver(
                httpx_client=self._http_client,
                base_url=self.base_url,
            )
            agent_card = await resolver.get_agent_card()
            self._a2a_client = A2AClient(
                httpx_client=self._http_client,
                agent_card=agent_card,
            )
        return self._a2a_client

    async def get_decision(self, claim_prompt: str) -> str:
        """
        Send a claim prompt to the purple agent and return its text response.

        Args:
            claim_prompt: Full LLM prompt with claim details

        Returns:
            Agent's response text (should contain Decision:/Rationale:/Evidence:)
        """
        client = await self._get_client()

        request = SendMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(
                message=new_agent_text_message(claim_prompt),
            ),
        )

        response = await client.send_message(request)

        # Extract text from response — result may be a Message or Task
        result = response.root.result

        # Task path: artifacts added via updater.add_artifact()
        artifacts = getattr(result, "artifacts", None)
        if artifacts:
            for artifact in artifacts:
                for part in (artifact.parts or []):
                    if hasattr(part.root, "text"):
                        return part.root.text

        # Message path: direct parts on the message
        parts = getattr(result, "parts", None)
        if parts:
            for part in parts:
                if hasattr(part.root, "text"):
                    return part.root.text

        logger.warning("Purple agent returned no text parts")
        return "Decision: APPROVE\nRationale: No response received.\nEvidence: N/A"

    async def close(self):
        await self._http_client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
