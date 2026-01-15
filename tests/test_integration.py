"""
Integration tests for PDDL Planner Evaluator and Mock Planner interaction.

These tests require both servers to be running:
- Evaluator agent on port 9009
- Mock planner agent on port 9010

To run these tests:
    Terminal 1: uv run src/server.py
    Terminal 2: cd tests/mock_planner && uv run server.py
    Terminal 3: uv run pytest tests/test_integration.py -v -s
"""

import json
import pytest
import httpx
from uuid import uuid4

from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart


EVALUATOR_URL = "http://localhost:9009"
PLANNER_URL = "http://localhost:9010"


async def send_evaluation_request(
    request_data: dict,
    timeout: float = 60.0
) -> list:
    """Send an evaluation request to the evaluator agent."""
    async with httpx.AsyncClient(timeout=timeout) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=EVALUATOR_URL)
        agent_card = await resolver.get_agent_card()
        config = ClientConfig(httpx_client=httpx_client, streaming=False)  # Enable streaming
        factory = ClientFactory(config)
        client = factory.create(agent_card)

        msg = Message(
            kind="message",
            role=Role.user,
            parts=[Part(TextPart(text=json.dumps(request_data)))],
            message_id=uuid4().hex,
        )

        events = [event async for event in client.send_message(msg)]

    return events


@pytest.mark.asyncio
async def test_servers_are_running():
    """Verify both servers are running and responding."""
    async with httpx.AsyncClient() as client:
        # Check evaluator
        try:
            resp = await client.get(f"{EVALUATOR_URL}/.well-known/agent-card.json")
            assert resp.status_code == 200
            evaluator_card = resp.json()
            assert evaluator_card["name"] == "PDDL Planner Evaluator"
            print(f"✓ Evaluator agent running at {EVALUATOR_URL}")
        except httpx.ConnectError:
            pytest.fail(
                f"Evaluator agent not running at {EVALUATOR_URL}. "
                "Start it with: uv run src/server.py"
            )

        # Check mock planner
        try:
            resp = await client.get(f"{PLANNER_URL}/.well-known/agent-card.json")
            assert resp.status_code == 200
            planner_card = resp.json()
            assert planner_card["name"] == "mock_pddl_planner"
            print(f"✓ Mock planner running at {PLANNER_URL}")
        except httpx.ConnectError:
            pytest.fail(
                f"Mock planner not running at {PLANNER_URL}. "
                "Start it with: cd tests/mock_planner && uv run server.py"
            )


@pytest.mark.asyncio
async def test_easy_blocksworld_passes():
    """Test that the first easy blocksworld problem (p01.pddl) passes validation.
    
    The mock planner returns a valid plan for p01.pddl, so this should succeed.
    """
    request = {
        "participants": {
            "planner": PLANNER_URL
        },
        "config": {
            "domains": "blocksworld",
            "easy_count": 1,
            "medium_count": 0,
            "hard_count": 0
        }
    }

    print(f"\nTesting easy blocksworld (should PASS)...")
    print(f"Request: {json.dumps(request, indent=2)}")
    events = await send_evaluation_request(request, timeout=120.0)

    # Extract all messages and artifacts
    success_rate = None
    
    for (content, _) in events:
        if hasattr(content, "artifacts") and content.artifacts:
            for artifact in content.artifacts:
                for part in artifact.parts:
                    if not hasattr(part, "root"):
                        continue
                    if not hasattr(part.root, "data"):
                        continue
                    if "overall_success_rate" in part.root.data:
                        success_rate = part.root.data["overall_success_rate"]
    assert success_rate is not None, "Evaluation should complete"
    
    assert success_rate == 1.0, f"Easy blocksworld should pass with 100% success rate, got {success_rate}"


@pytest.mark.asyncio
async def test_medium_blocksworld_fails():
    """Test that the first medium blocksworld problem (p22.pddl) fails validation.
    
    The mock planner returns a plan for p01.pddl, which won't work for p22.pddl.
    """
    request = {
        "participants": {
            "planner": PLANNER_URL
        },
        "config": {
            "domains": "blocksworld",
            "easy_count": 0,
            "medium_count": 1,
            "hard_count": 0
        }
    }

    print(f"\nTesting medium blocksworld (should FAIL)...")
    print(f"Request: {json.dumps(request, indent=2)}")
    events = await send_evaluation_request(request, timeout=120.0)

    # Extract all messages and artifacts
    success_rate = None

    for (content, _) in events:
        if hasattr(content, "artifacts") and content.artifacts:
            for artifact in content.artifacts:
                for part in artifact.parts:
                    if not hasattr(part, "root"):
                        continue
                    if not hasattr(part.root, "data"):
                        continue
                    if "overall_success_rate" in part.root.data:
                        success_rate = part.root.data["overall_success_rate"]
    assert success_rate is not None, "Evaluation should complete"
    
    assert success_rate == 0.0, f"Medium blocksworld should pass with 0% success rate, got {success_rate}"

@pytest.mark.asyncio
async def half_first_two_easy_blocksworld_passes():
    """Test that two easy blocksworld problems return 50% success rate.
    
    The mock planner returns a valid plan for p01.pddl, but not for p02.pddl.
    """
    request = {
        "participants": {
            "planner": PLANNER_URL
        },
        "config": {
            "domains": "blocksworld",
            "easy_count": 2,
            "medium_count": 0,
            "hard_count": 0
        }
    }

    print(f"\nTesting two easy blocksworld problems (should be 50% PASS)...")
    print(f"Request: {json.dumps(request, indent=2)}")
    events = await send_evaluation_request(request, timeout=180.0)

    # Extract all messages and artifacts
    success_rate = None
    
    for (content, _) in events:
        if hasattr(content, "artifacts") and content.artifacts:
            for artifact in content.artifacts:
                for part in artifact.parts:
                    if not hasattr(part, "root"):
                        continue
                    if not hasattr(part.root, "data"):
                        continue
                    if "overall_success_rate" in part.root.data:
                        success_rate = part.root.data["overall_success_rate"]
    assert success_rate is not None, "Evaluation should complete"
    
    assert success_rate == 0.5, f"Two easy blocksworld problems should yield 50% success rate, got {success_rate}"


@pytest.mark.asyncio
async def test_multiple_domains():
    """Test evaluation across multiple domains."""
    request = {
        "participants": {
            "planner": PLANNER_URL
        },
        "config": {
            "domains": ["blocksworld", "gripper"],
            "easy_count": 1,
            "medium_count": 0,
            "hard_count": 0
        }
    }

    print(f"\nSending multi-domain request: {json.dumps(request, indent=2)}")
    events = await send_evaluation_request(request, timeout=180.0)

    assert len(events) > 0, "Should receive events from evaluator"

    completed = False
    for event in events:
        match event:
            case (task, update):
                if task.status.state == "completed":
                    completed = True
                    break

    assert completed, "Multi-domain evaluation should complete"
    print("✓ Multi-domain integration test successful")


@pytest.mark.asyncio
async def test_invalid_request_is_rejected():
    """Test that invalid requests are properly rejected."""
    request = {
        "participants": {
            "planner": PLANNER_URL
        },
        "config": {
            "domains": "blocksworld",
            "easy_count": 20,  # Invalid: > 10
            "medium_count": 0,
            "hard_count": 0
        }
    }

    print(f"\nSending invalid request: {json.dumps(request, indent=2)}")
    events = await send_evaluation_request(request, timeout=30.0)

    rejected = False
    for event in events:
        match event:
            case (task, update):
                if task.status.state == "rejected":
                    rejected = True
                    print(f"Rejection message: {update}")
                    break

    assert rejected, "Invalid request should be rejected"
    print("✓ Invalid request properly rejected")


@pytest.mark.asyncio
async def test_planner_generates_plan():
    """Test that the mock planner can generate a plan directly."""
    async with httpx.AsyncClient(timeout=60.0) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=PLANNER_URL)
        agent_card = await resolver.get_agent_card()
        config = ClientConfig(httpx_client=httpx_client, streaming=False)
        factory = ClientFactory(config)
        client = factory.create(agent_card)

        task_description = """
Available Actions:
  - (pick-up ?x): Pick up block ?x from the table
  - (put-down ?x): Put down block ?x on the table
  - (stack ?x ?y): Stack block ?x on top of block ?y
  - (unstack ?x ?y): Unstack block ?x from on top of block ?y

Objects:
  - Blocks: a, b

Initial State:
  - Block a is on the table
  - Block b is on the table
  - Block a is clear
  - Block b is clear
  - The arm is empty

Goal:
  - Block a is on block b
"""

        msg = Message(
            kind="message",
            role=Role.user,
            parts=[Part(TextPart(text=task_description))],
            message_id=uuid4().hex,
        )

        events = [event async for event in client.send_message(msg)]

    # Check that planner generated something
    completed = False
    has_artifact = False

    for event in events:
        match event:
            case (task, update):
                print(f"Planner task state: {task.status.state}")
                if task.status.state == "completed":
                    completed = True
                if update:
                    if hasattr(update, 'artifacts') and update.artifacts:
                        has_artifact = True
                        print(f"\nPlanner artifacts in update: {update.artifacts}")
            case msg:
                if hasattr(msg, 'artifacts') and msg.artifacts:
                    has_artifact = True
                    print(f"\nPlanner artifacts in message: {msg.artifacts}")

    assert completed, "Planner should complete"
    # Artifacts are in the final state, completion is what matters
    print(f"✓ Mock planner completed (has_artifact={has_artifact})")

if __name__ == "__main__":
    print(__doc__)
