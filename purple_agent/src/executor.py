"""Purple Agent Executor — A2A glue layer for the RL model."""

import logging

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import UnsupportedOperationError
from a2a.utils import new_task

from agent import HealthcareFraudAgent

logger = logging.getLogger(__name__)


class Executor(AgentExecutor):
    def __init__(self, model_path: str = "/app/model"):
        # Model loads once at startup (lazy on first call)
        self._agent = HealthcareFraudAgent(model_path=model_path)

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task
        if task is None:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work()

        try:
            await self._agent.run(context.message, updater)
            await updater.complete()
        except Exception as e:
            logger.exception(f"Model inference failed: {e}")
            await updater.failed(message=str(e))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise UnsupportedOperationError(message="Cancel not supported")
