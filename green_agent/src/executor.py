"""
AgentExecutor — thin A2A glue layer.
Follows the canonical AgentBeats pattern (osworld-green, fieldworkarena-green).
"""

import logging
from collections.abc import AsyncGenerator

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InternalError,
    InvalidParamsError,
    Message,
    Part,
    Task,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import new_agent_text_message, new_task

from agent import HealthcareFraudGreenAgent

logger = logging.getLogger(__name__)


class Executor(AgentExecutor):
    """
    AgentExecutor for the Healthcare Fraud Green Agent.

    Maintains per-context agent instances so concurrent assessments
    don't share episode state.
    """

    def __init__(self):
        self._agents: dict[str, HealthcareFraudGreenAgent] = {}

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        task = context.current_task
        if task is None:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)

        context_id = task.context_id
        agent = self._agents.setdefault(
            context_id, HealthcareFraudGreenAgent()
        )

        updater = TaskUpdater(event_queue, task.id, context_id)
        await updater.start_work()

        try:
            await agent.run(context.message, updater)
            await updater.complete()
        except Exception as e:
            logger.exception(f"Assessment failed for context {context_id}: {e}")
            await updater.failed(message=str(e))
        finally:
            # Clean up completed sessions (memory management)
            self._agents.pop(context_id, None)

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        context_id = context.current_task.context_id if context.current_task else None
        if context_id:
            self._agents.pop(context_id, None)
        raise UnsupportedOperationError(message="Cancel not supported")
