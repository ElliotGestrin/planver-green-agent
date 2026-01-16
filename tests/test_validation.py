"""Tests for PDDL plan validation using pddlval."""
import pytest
from pathlib import Path
from pddlval import validate_plan


@pytest.fixture
def workspace_root():
    """Get the workspace root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def pddl_root(workspace_root):
    """Get the PDDL domain and problems root directory."""
    return workspace_root / "pddl"


@pytest.fixture
def sample_plan(workspace_root):
    """Get the sample plan file path."""
    plan_path = workspace_root / "tests" / "mock_planner" / "sample_plan.txt"
    assert plan_path.exists(), f"Sample plan not found at {plan_path}"
    return str(plan_path)


def test_blocksworld_plan_valid(pddl_root, sample_plan):
    """Test that the sample plan is valid for blocksworld/p01."""
    domain_file = str(pddl_root / "blocksworld" / "domain.pddl")
    problem_file = str(pddl_root / "blocksworld" / "p01.pddl")
    
    assert validate_plan(domain_file, problem_file, sample_plan), \
        "Sample plan should be valid for blocksworld/p01"


@pytest.mark.parametrize("domain", [
    "barman",
    "childsnack",
    "gripper",
    "depots",
])
def test_other_domains_plan_invalid(pddl_root, sample_plan, domain):
    """Test that the blocksworld sample plan fails for other domains' p01."""
    domain_file = str(pddl_root / domain / "domain.pddl")
    problem_file = str(pddl_root / domain / "p01.pddl")
    
    assert not validate_plan(domain_file, problem_file, sample_plan), \
        f"Blocksworld plan should not be valid for {domain}/p01"


def test_plan_as_content(pddl_root):
    """Test validation with plan content instead of file path."""
    domain_file = str(pddl_root / "blocksworld" / "domain.pddl")
    problem_file = str(pddl_root / "blocksworld" / "p01.pddl")
    
    # Read the sample plan content
    plan_path = Path(__file__).parent / "mock_planner" / "sample_plan.txt"
    with open(plan_path, 'r') as f:
        plan_content = f.read()
    
    # Validate using content directly
    assert validate_plan(domain_file, problem_file, plan_content), \
        "Validation should work with plan content string"
