import argparse
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from executor import Executor


def main():
    parser = argparse.ArgumentParser(description="Run the A2A agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    args = parser.parse_args()

    skill = AgentSkill(
        id="pddl-planner-evaluation",
        name="PDDL Planner Evaluation",
        description="Evaluates PDDL planning agents by testing them on various domains and difficulty levels. "
                    "Generates natural language planning tasks, collects plans from the agent, and validates them.",
        tags=["planning", "pddl", "evaluation", "validation"],
        examples=[
            "Test a planner on blocksworld domain with 5 easy, 3 medium, and 2 hard problems",
            "Evaluate a planner across all domains with 2 problems per difficulty level",
            "Test planner on gripper and logistics domains with 10 easy problems each"
        ]
    )

    agent_card = AgentCard(
        name="PDDL Planner Evaluator",
        description="A green agent that evaluates PDDL planning capabilities. "
                    "It generates natural language planning tasks from PDDL domains, "
                    "sends them to a planner agent, and validates the returned plans. "
                    "Supports testing across 30+ PDDL domains with configurable difficulty levels.",
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill]
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
