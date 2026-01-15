# Mock PDDL Planner Agent

A minimal testing agent that returns a pre-written valid plan for blocksworld p01.
This allows for deterministic testing without relying on LLM API calls.

## Usage

### Start the mock planner:
```bash
cd tests/mock_planner
uv run server.py --port 9010
```

### Test with the evaluator:

In another terminal, start your evaluator agent:
```bash
uv run src/server.py --port 9009
```

Then run the integration tests:
```bash
uv run pytest tests/test_integration.py -v -s
```

## How it works

The agent always returns the same valid plan from `sample_plan.txt`, which solves the blocksworld p01 problem. This makes testing predictable and fast, without requiring OpenAI API access.
