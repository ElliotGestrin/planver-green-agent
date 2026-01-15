import logging
import random
from typing import Any, Literal
from pathlib import Path
from pydantic import BaseModel, HttpUrl, ValidationError
from dotenv import load_dotenv

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message
from pddlval import validate_plan

from messenger import Messenger
from describe_pddl import PDDLDescriber

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pddl_planner_evaluator")


class EvalRequest(BaseModel):
    """Request format sent by the AgentBeats platform to green agents."""
    participants: dict[str, HttpUrl]  # role -> agent URL
    config: dict[str, Any]


class TaskResult(BaseModel):
    """Result for a single planning task."""
    domain: str
    problem: str
    difficulty: Literal["easy", "medium", "hard"]
    valid: bool
    plan: str
    error: str | None = None


class DomainResults(BaseModel):
    """Aggregated results for a domain."""
    domain: str
    total_tasks: int
    valid_plans: int
    invalid_plans: int
    errors: int
    success_rate: float
    results_by_difficulty: dict[str, dict[str, int]]


class PlannerEvaluation(BaseModel):
    """Overall evaluation results."""
    total_tasks: int
    total_valid: int
    total_invalid: int
    total_errors: int
    overall_success_rate: float
    domain_results: list[DomainResults]


class Agent:
    required_roles: list[str] = ["planner"]
    required_config_keys: list[str] = ["domains", "easy_count", "medium_count", "hard_count"]

    def __init__(self):
        self.messenger = Messenger()
        self.pddl_base_path = Path("pddl")
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"

        missing_config_keys = set(self.required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"

        # Validate difficulty counts
        for key in ["easy_count", "medium_count", "hard_count"]:
            try:
                count = int(request.config[key])
                if not 0 <= count <= 10:
                    return False, f"{key} must be between 0 and 10"
            except ValueError:
                return False, f"{key} must be an integer"

        # Validate domains
        domains = request.config["domains"]
        if isinstance(domains, str):
            if domains != "all":
                # Single domain or list
                domains = [domains]
        elif not isinstance(domains, list):
            return False, "domains must be 'all', a domain name, or a list of domain names"

        return True, "ok"

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Run PDDL planning evaluation.

        Args:
            message: The incoming message
            updater: Report progress and results
        """
        input_text = get_message_text(message)

        try:
            request: EvalRequest = EvalRequest.model_validate_json(input_text)
            ok, msg = self.validate_request(request)
            if not ok:
                await updater.reject(new_agent_text_message(msg))
                return
        except ValidationError as e:
            await updater.reject(new_agent_text_message(f"Invalid request: {e}"))
            return

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Starting PDDL planning evaluation.\n{request.model_dump_json()}")
        )

        # Parse configuration
        domains_config = request.config["domains"]
        easy_count = int(request.config["easy_count"])
        medium_count = int(request.config["medium_count"])
        hard_count = int(request.config["hard_count"])

        # Determine which domains to test
        if domains_config == "all":
            domains = [d.name for d in self.pddl_base_path.iterdir() if d.is_dir()]
        elif isinstance(domains_config, str):
            domains = [domains_config]
        else:
            domains = domains_config

        logger.info(f"Testing {len(domains)} domains with {easy_count} easy, {medium_count} medium, {hard_count} hard tasks each")

        # Run evaluation
        evaluation = await self.evaluate_planner(
            request.participants["planner"],
            domains,
            easy_count,
            medium_count,
            hard_count,
            updater
        )

        # Report results
        result_text = self._format_results(evaluation)
        await updater.add_artifact(
            parts=[
                Part(root=TextPart(text=result_text)),
                Part(root=DataPart(data=evaluation.model_dump())),
            ],
            name="Evaluation Results",
        )
        
    async def evaluate_planner(
        self,
        planner_url: str,
        domains: list[str],
        easy_count: int,
        medium_count: int,
        hard_count: int,
        updater: TaskUpdater,
    ) -> PlannerEvaluation:
        """Evaluate a planner agent across multiple domains and difficulty levels."""
        all_results: list[TaskResult] = []
        domain_stats: dict[str, DomainResults] = {}

        for domain in domains:
            domain_path = self.pddl_base_path / domain
            if not domain_path.exists():
                logger.warning(f"Domain {domain} not found, skipping")
                continue

            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Testing domain: {domain}")
            )

            # Initialize describer for this domain
            try:
                describer = PDDLDescriber(str(domain_path))
            except Exception as e:
                logger.error(f"Failed to initialize describer for {domain}: {e}")
                continue

            # Test each difficulty level
            domain_results = []
            for difficulty, count in [("easy", easy_count), ("medium", medium_count), ("hard", hard_count)]:
                if count == 0:
                    continue

                problems = describer.get_problems_by_difficulty(difficulty)
                sampled = problems[:count]

                for problem_file in sampled:
                    result = await self.test_single_task(
                        describer,
                        domain,
                        problem_file,
                        difficulty,
                        planner_url,
                        updater
                    )
                    print(f"Result for {domain}/{problem_file}: {'Valid' if result.valid else 'Invalid' if result.error is None else 'Error'}")
                    domain_results.append(result)
                    all_results.append(result)

            # Aggregate domain statistics
            domain_stats[domain] = self._aggregate_domain_results(domain, domain_results)

        # Compute overall statistics
        total_tasks = len(all_results)
        total_valid = sum(1 for r in all_results if r.valid)
        total_invalid = sum(1 for r in all_results if not r.valid and r.error is None)
        total_errors = sum(1 for r in all_results if r.error is not None)
        
        return PlannerEvaluation(
            total_tasks=total_tasks,
            total_valid=total_valid,
            total_invalid=total_invalid,
            total_errors=total_errors,
            overall_success_rate=total_valid / total_tasks if total_tasks > 0 else 0.0,
            domain_results=list(domain_stats.values())
        )

    async def test_single_task(
        self,
        describer: PDDLDescriber,
        domain: str,
        problem_file: str,
        difficulty: str,
        planner_url: str,
        updater: TaskUpdater,
    ) -> TaskResult:
        """Test the planner on a single task."""
        problem_path = describer.get_problem_path(problem_file)
        
        try:
            # Generate NL task prompt
            task_prompt = describer.generate_task_prompt(problem_path)
            
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Testing {domain}/{problem_file} ({difficulty})")
            )
            
            # Get plan from planner agent
            plan_response = await self.messenger.talk_to_agent(
                task_prompt, str(planner_url), new_conversation=True
            )
            
            # Save the response to file
            result_file = self.results_dir / f"{domain}_{problem_file.replace('.pddl', '')}.txt"
            result_file.write_text(plan_response)
            
            # Validate the plan
            is_valid = validate_plan(
                domain=str(describer.domain_path),
                problem=problem_path,
                plan=plan_response
            )

            print(f"Domain: {domain}, Problem: {problem_file}, Difficulty: {difficulty}, Valid: {is_valid}")
            print(f"Plan: {plan_response}")
            
            logger.info(f"{domain}/{problem_file}: {'✓' if is_valid else '✗'}")
            
            return TaskResult(
                domain=domain,
                problem=problem_file,
                difficulty=difficulty,
                valid=is_valid,
                plan=plan_response,
                error=None
            )
            
        except Exception as e:
            logger.error(f"Error testing {domain}/{problem_file}: {e}")
            return TaskResult(
                domain=domain,
                problem=problem_file,
                difficulty=difficulty,
                valid=False,
                plan="",
                error=str(e)
            )

    def _aggregate_domain_results(self, domain: str, results: list[TaskResult]) -> DomainResults:
        """Aggregate results for a single domain."""
        total = len(results)
        valid = sum(1 for r in results if r.valid)
        invalid = sum(1 for r in results if not r.valid and r.error is None)
        errors = sum(1 for r in results if r.error is not None)
        
        # Statistics by difficulty
        by_difficulty = {}
        for diff in ["easy", "medium", "hard"]:
            diff_results = [r for r in results if r.difficulty == diff]
            if diff_results:
                by_difficulty[diff] = {
                    "total": len(diff_results),
                    "valid": sum(1 for r in diff_results if r.valid),
                    "invalid": sum(1 for r in diff_results if not r.valid and r.error is None),
                    "errors": sum(1 for r in diff_results if r.error is not None),
                }
        
        return DomainResults(
            domain=domain,
            total_tasks=total,
            valid_plans=valid,
            invalid_plans=invalid,
            errors=errors,
            success_rate=valid / total if total > 0 else 0.0,
            results_by_difficulty=by_difficulty
        )

    def _format_results(self, evaluation: PlannerEvaluation) -> str:
        """Format evaluation results as human-readable text."""
        lines = [
            "=" * 80,
            "PDDL PLANNER EVALUATION RESULTS",
            "=" * 80,
            "",
            f"Overall Statistics:",
            f"  Total Tasks: {evaluation.total_tasks}",
            f"  Valid Plans: {evaluation.total_valid}",
            f"  Invalid Plans: {evaluation.total_invalid}",
            f"  Errors: {evaluation.total_errors}",
            f"  Success Rate: {evaluation.overall_success_rate:.1%}",
            "",
            "Results by Domain:",
            ""
        ]
        
        for domain_result in evaluation.domain_results:
            lines.extend([
                f"  {domain_result.domain}:",
                f"    Tasks: {domain_result.total_tasks}",
                f"    Valid: {domain_result.valid_plans}",
                f"    Invalid: {domain_result.invalid_plans}",
                f"    Errors: {domain_result.errors}",
                f"    Success Rate: {domain_result.success_rate:.1%}",
            ])
            
            if domain_result.results_by_difficulty:
                lines.append("    By Difficulty:")
                for diff, stats in domain_result.results_by_difficulty.items():
                    success_rate = stats["valid"] / stats["total"] if stats["total"] > 0 else 0.0
                    lines.append(f"      {diff.capitalize()}: {stats['valid']}/{stats['total']} ({success_rate:.1%})")
            lines.append("")
        
        return "\n".join(lines)
