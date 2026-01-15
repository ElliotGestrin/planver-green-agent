from typing import Any
import pytest
import httpx
from uuid import uuid4

from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart


# A2A validation helpers - adapted from https://github.com/a2aproject/a2a-inspector/blob/main/backend/validators.py

def validate_agent_card(card_data: dict[str, Any]) -> list[str]:
    """Validate the structure and fields of an agent card."""
    errors: list[str] = []

    # Use a frozenset for efficient checking and to indicate immutability.
    required_fields = frozenset(
        [
            'name',
            'description',
            'url',
            'version',
            'capabilities',
            'defaultInputModes',
            'defaultOutputModes',
            'skills',
        ]
    )

    # Check for the presence of all required fields
    for field in required_fields:
        if field not in card_data:
            errors.append(f"Required field is missing: '{field}'.")

    # Check if 'url' is an absolute URL (basic check)
    if 'url' in card_data and not (
        card_data['url'].startswith('http://')
        or card_data['url'].startswith('https://')
    ):
        errors.append(
            "Field 'url' must be an absolute URL starting with http:// or https://."
        )

    # Check if capabilities is a dictionary
    if 'capabilities' in card_data and not isinstance(
        card_data['capabilities'], dict
    ):
        errors.append("Field 'capabilities' must be an object.")

    # Check if defaultInputModes and defaultOutputModes are arrays of strings
    for field in ['defaultInputModes', 'defaultOutputModes']:
        if field in card_data:
            if not isinstance(card_data[field], list):
                errors.append(f"Field '{field}' must be an array of strings.")
            elif not all(isinstance(item, str) for item in card_data[field]):
                errors.append(f"All items in '{field}' must be strings.")

    # Check skills array
    if 'skills' in card_data:
        if not isinstance(card_data['skills'], list):
            errors.append(
                "Field 'skills' must be an array of AgentSkill objects."
            )
        elif not card_data['skills']:
            errors.append(
                "Field 'skills' array is empty. Agent must have at least one skill if it performs actions."
            )

    return errors


def _validate_task(data: dict[str, Any]) -> list[str]:
    errors = []
    if 'id' not in data:
        errors.append("Task object missing required field: 'id'.")
    if 'status' not in data or 'state' not in data.get('status', {}):
        errors.append("Task object missing required field: 'status.state'.")
    return errors


def _validate_status_update(data: dict[str, Any]) -> list[str]:
    errors = []
    if 'status' not in data or 'state' not in data.get('status', {}):
        errors.append(
            "StatusUpdate object missing required field: 'status.state'."
        )
    return errors


def _validate_artifact_update(data: dict[str, Any]) -> list[str]:
    errors = []
    if 'artifact' not in data:
        errors.append(
            "ArtifactUpdate object missing required field: 'artifact'."
        )
    elif (
        'parts' not in data.get('artifact', {})
        or not isinstance(data.get('artifact', {}).get('parts'), list)
        or not data.get('artifact', {}).get('parts')
    ):
        errors.append("Artifact object must have a non-empty 'parts' array.")
    return errors


def _validate_message(data: dict[str, Any]) -> list[str]:
    errors = []
    if (
        'parts' not in data
        or not isinstance(data.get('parts'), list)
        or not data.get('parts')
    ):
        errors.append("Message object must have a non-empty 'parts' array.")
    if 'role' not in data or data.get('role') != 'agent':
        errors.append("Message from agent must have 'role' set to 'agent'.")
    return errors


def validate_event(data: dict[str, Any]) -> list[str]:
    """Validate an incoming event from the agent based on its kind."""
    if 'kind' not in data:
        return ["Response from agent is missing required 'kind' field."]

    kind = data.get('kind')
    validators = {
        'task': _validate_task,
        'status-update': _validate_status_update,
        'artifact-update': _validate_artifact_update,
        'message': _validate_message,
    }

    validator = validators.get(str(kind))
    if validator:
        return validator(data)

    return [f"Unknown message kind received: '{kind}'."]


# A2A messaging helpers

async def send_text_message(text: str, url: str, context_id: str | None = None, streaming: bool = False):
    async with httpx.AsyncClient(timeout=10) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=url)
        agent_card = await resolver.get_agent_card()
        config = ClientConfig(httpx_client=httpx_client, streaming=streaming)
        factory = ClientFactory(config)
        client = factory.create(agent_card)

        msg = Message(
            kind="message",
            role=Role.user,
            parts=[Part(TextPart(text=text))],
            message_id=uuid4().hex,
            context_id=context_id,
        )

        events = [event async for event in client.send_message(msg)]

    return events


# A2A conformance tests

def test_agent_card(agent):
    """Validate agent card structure and required fields."""
    response = httpx.get(f"{agent}/.well-known/agent-card.json")
    assert response.status_code == 200, "Agent card endpoint must return 200"

    card_data = response.json()
    errors = validate_agent_card(card_data)

    assert not errors, f"Agent card validation failed:\n" + "\n".join(errors)

@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True, False])
async def test_message(agent, streaming):
    """Test that agent returns valid A2A message format."""
    events = await send_text_message("Hello", agent, streaming=streaming)

    all_errors = []
    for event in events:
        match event:
            case Message() as msg:
                errors = validate_event(msg.model_dump())
                all_errors.extend(errors)

            case (task, update):
                errors = validate_event(task.model_dump())
                all_errors.extend(errors)
                if update:
                    errors = validate_event(update.model_dump())
                    all_errors.extend(errors)

            case _:
                pytest.fail(f"Unexpected event type: {type(event)}")

    assert events, "Agent should respond with at least one event"
    assert not all_errors, f"Message validation failed:\n" + "\n".join(all_errors)

# PDDL Planner Evaluator Agent Tests

@pytest.mark.asyncio
async def test_agent_card_has_correct_skill(agent):
    """Verify agent card contains the PDDL planner evaluation skill."""
    response = httpx.get(f"{agent}/.well-known/agent-card.json")
    card_data = response.json()
    
    assert 'skills' in card_data
    assert len(card_data['skills']) > 0
    
    skill = card_data['skills'][0]
    assert skill['id'] == 'pddl-planner-evaluation'
    assert 'PDDL' in skill['name'] or 'planner' in skill['name'].lower()
    assert 'planning' in [tag.lower() for tag in skill.get('tags', [])]


@pytest.mark.asyncio
async def test_invalid_request_missing_roles(agent):
    """Test that agent rejects requests missing required participant roles."""
    request = {
        "participants": {},  # Missing 'planner' role
        "config": {
            "domains": "blocksworld",
            "easy_count": 1,
            "medium_count": 0,
            "hard_count": 0
        }
    }
    
    import json
    events = await send_text_message(json.dumps(request), agent)
    
    # Should reject with error message
    has_rejection = False
    for event in events:
        match event:
            case (task, update):
                if task.status.state == "rejected":
                    has_rejection = True
                    break
    
    assert has_rejection, "Agent should reject requests missing required roles"


@pytest.mark.asyncio
async def test_invalid_request_missing_config(agent):
    """Test that agent rejects requests missing required config keys."""
    request = {
        "participants": {
            "planner": "http://example.com/planner"
        },
        "config": {
            "domains": "blocksworld"
            # Missing easy_count, medium_count, hard_count
        }
    }
    
    import json
    events = await send_text_message(json.dumps(request), agent)
    
    has_rejection = False
    for event in events:
        match event:
            case (task, update):
                if task.status.state == "rejected":
                    has_rejection = True
                    break
    
    assert has_rejection, "Agent should reject requests missing required config keys"


@pytest.mark.asyncio
async def test_invalid_difficulty_count_out_of_range(agent):
    """Test that agent rejects difficulty counts outside 0-10 range."""
    request = {
        "participants": {
            "planner": "http://example.com/planner"
        },
        "config": {
            "domains": "blocksworld",
            "easy_count": 15,  # Invalid: > 10
            "medium_count": 0,
            "hard_count": 0
        }
    }
    
    import json
    events = await send_text_message(json.dumps(request), agent)
    
    has_rejection = False
    for event in events:
        match event:
            case (task, update):
                if task.status.state == "rejected":
                    has_rejection = True
                    break
    
    assert has_rejection, "Agent should reject difficulty counts outside 0-10 range"


@pytest.mark.asyncio
async def test_valid_request_single_domain(agent):
    """Test that agent accepts valid requests with single domain."""
    request = {
        "participants": {
            "planner": "http://example.com/planner"
        },
        "config": {
            "domains": "blocksworld",
            "easy_count": 1,
            "medium_count": 0,
            "hard_count": 0
        }
    }
    
    import json
    events = await send_text_message(json.dumps(request), agent)
    
    # Should not reject immediately
    has_immediate_rejection = False
    has_working_status = False
    
    for event in events:
        match event:
            case (task, update):
                if task.status.state == "rejected":
                    has_immediate_rejection = True
                elif task.status.state == "working":
                    has_working_status = True
                    break  # Stop checking once we see it's working
    
    assert not has_immediate_rejection, "Agent should accept valid requests"
    # Note: We don't check for has_working_status here because the planner URL
    # is fake and the agent may fail during execution, but it should at least
    # not reject the request format


@pytest.mark.asyncio
async def test_valid_request_multiple_domains(agent):
    """Test that agent accepts requests with multiple domains."""
    request = {
        "participants": {
            "planner": "http://example.com/planner"
        },
        "config": {
            "domains": ["blocksworld", "gripper"],
            "easy_count": 1,
            "medium_count": 0,
            "hard_count": 0
        }
    }
    
    import json
    events = await send_text_message(json.dumps(request), agent)
    
    has_immediate_rejection = False
    for event in events:
        match event:
            case (task, update):
                if task.status.state == "rejected":
                    has_immediate_rejection = True
                    break
    
    assert not has_immediate_rejection, "Agent should accept requests with domain lists"


def test_pddl_describer_exists():
    """Test that PDDLDescriber can be imported and used."""
    import sys
    from pathlib import Path
    
    # Add src to path
    src_path = Path(__file__).parent.parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    from describe_pddl import PDDLDescriber
    
    # Test with blocksworld
    pddl_path = Path(__file__).parent.parent / "pddl" / "blocksworld"
    assert pddl_path.exists(), "blocksworld domain should exist"
    
    describer = PDDLDescriber(str(pddl_path))
    assert describer is not None
    
    # Test getting problems by difficulty
    easy_problems = describer.get_problems_by_difficulty("easy")
    assert isinstance(easy_problems, list)
    assert len(easy_problems) > 0, "Should have some easy problems"


def test_pddl_validation_import():
    """Test that pddlval can be imported."""
    from pddlval import validate_plan
    assert validate_plan is not None
