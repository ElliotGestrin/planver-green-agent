"""
Mock PDDL Planner Agent for Testing

This agent returns a pre-written valid plan for blocksworld p01.
Used for deterministic testing without relying on LLM API calls.
"""

import os
from pathlib import Path

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart
from a2a.utils import get_message_text, new_agent_text_message


class Agent:
    def __init__(self):
        # Load the sample plan from file
        plan_file = Path(__file__).parent / "sample_plan.txt"
        with open(plan_file, 'r') as f:
            self.sample_plan = f.read().strip()

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Return the pre-written plan.
        
        Args:
            message: Contains the planning task (ignored)
            updater: For reporting progress and results
        """
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Generating plan...")
        )

        # Simply return the pre-written plan
        #await updater.add_artifact(
        #    parts=[Part(root=TextPart(text=self.sample_plan))],
        #    name="Plan",
        #)

        # Send the plan as a text message
        await updater.complete(
            new_agent_text_message(f"{self.sample_plan}")
        )
        