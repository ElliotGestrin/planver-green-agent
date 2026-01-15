"""
Mock PDDL Planner Server for Testing

Runs a simple planner agent that can be used to test the PDDL evaluator.
"""

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
    parser = argparse.ArgumentParser(description="Run the mock PDDL planner agent")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9010, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    args = parser.parse_args()

    skill = AgentSkill(
        id="pddl-planning",
        name="PDDL Planning",
        description="Generates plans for PDDL planning tasks using LLM reasoning",
        tags=["planning", "pddl"],
        examples=[],
    )

    agent_card = AgentCard(
        name="mock_pddl_planner",
        description="Mock PDDL planner for testing the evaluator agent",
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    print(f"Starting mock PDDL planner at http://{args.host}:{args.port}")
    print(f"Use this URL in your evaluator config: http://{args.host}:{args.port}")
    
    uvicorn.run(
        app.build(),
        host=args.host,
        port=args.port,
        timeout_keep_alive=300,
    )


if __name__ == "__main__":
    main()
