"""
PDDL Domain Description Generator

This module uses Unified Planning Framework to parse PDDL domain files
and OpenAI LLMs to generate natural language descriptions in YAML format.

Usage:
    # Generate domain description (creates description.yaml)
    python src/describe_pddl.py pddl/blocksworld/domain.pddl
    
    # Use the describer programmatically
    from src.describe_pddl import PDDLDescriber
    
    # Initialize with domain folder (loads existing description.yaml if present)
    describer = PDDLDescriber("pddl/blocksworld")
    
    # Generate natural language task prompt from a problem file
    prompt = describer.generate_task_prompt("pddl/blocksworld/p01.pddl")
    print(prompt)
"""

import os
import re
import yaml
from typing import Dict, List
from openai import OpenAI
from tqdm import tqdm

from unified_planning.io import PDDLReader
from unified_planning.model import Problem
from unified_planning import Environment
Environment.error_used_name = False  # Suppress errors when different parts of the domain use same names


class UnsupportedFeatureException(Exception):
    """Raised when domain contains unsupported features like axioms, fluents, or constants."""
    pass


class PDDLDescriber:
    """
    Generates natural language descriptions of PDDL domains using OpenAI LLMs.
    
    Attributes:
        domain_dir: Path to the PDDL domain directory
        domain_path: Path to the PDDL domain file
        client: OpenAI client instance
        types: List of type names
        predicates: Dict mapping predicate signatures to descriptions
        actions: Dict mapping action signatures to descriptions
    """
    
    def __init__(self, domain_folder_path: str, api_key: str = None):
        """
        Initialize the PDDL describer.
        
        Args:
            domain_folder_path: Path to the PDDL domain folder (or domain file for backwards compatibility)
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        # Handle both folder and file paths
        if os.path.isfile(domain_folder_path):
            self.domain_path = os.path.abspath(domain_folder_path)
            self.domain_dir = os.path.dirname(self.domain_path)
        else:
            self.domain_dir = os.path.abspath(domain_folder_path)
            # Look for domain.pddl in the directory
            self.domain_path = os.path.join(self.domain_dir, 'domain.pddl')
            if not os.path.exists(self.domain_path):
                raise FileNotFoundError(f"No domain.pddl found in {self.domain_dir}")
        
        # Check if description.yaml exists
        description_path = os.path.join(self.domain_dir, 'description.yaml')
        
        if os.path.exists(description_path):
            # Load from existing YAML
            with open(description_path, 'r') as f:
                description = yaml.safe_load(f)
            
            self.types = description.get('Types', ['object'])
            self.predicates = description.get('Predicates', {})
            self.actions = description.get('Actions', {})
            self.client = None  # No need for OpenAI if loading from file
        else:
            # Generate new descriptions using LLM
            # Find a problem file to parse the domain
            problem_files = [f for f in os.listdir(self.domain_dir) if f.startswith('p') and f.endswith('.pddl')]
            
            if not problem_files:
                raise FileNotFoundError(f"No problem files found in {self.domain_dir}")
            
            problem_path = os.path.join(self.domain_dir, problem_files[0])
            
            # Parse the domain
            reader = PDDLReader()
            problem: Problem = reader.parse_problem(self.domain_path, problem_path)
            
            # Initialize OpenAI client
            self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            
            # Check for unsupported features
            self._check_unsupported_features(problem)
            
            # Generate descriptions
            self.types = self._get_types(problem)
            self.predicates = self._get_predicates(problem)
            self.actions = self._get_actions(problem)
    
    def _check_unsupported_features(self, problem: Problem):
        """Check if domain contains unsupported features and raise exception if found."""
        # Check for derived predicates (axioms)
        if hasattr(problem, 'derived_predicates') and len(problem.derived_predicates) > 0:
            raise UnsupportedFeatureException("Domain contains derived predicates (axioms)")
        
        # Check for numeric fluents (different from boolean fluents/predicates)
        # Allow total-cost as it's commonly used for action costs
        for fluent in problem.fluents:
            if not fluent.type.is_bool_type():
                if fluent.name.lower().replace('-', '_') != 'total_cost':
                    raise UnsupportedFeatureException(f"Domain contains numeric fluent: {fluent.name}")
        
        # Check for constants (objects defined in domain)
        # In UP, constants are typically part of the domain but accessed via problem.objects()
        # We'll check if there are objects defined at the domain level
        # This is a conservative check - may need refinement based on actual UP behavior
    
    def _get_types(self, problem: Problem) -> List[str]:
        """
        Extract all type names from the domain.
        
        Returns:
            List of type names. Returns ["object"] if no types are defined.
        """
        if not problem.user_types:
            return ["object"]
        
        return [t.name for t in problem.user_types]
    
    def _describe_predicate_with_llm(self, signature: str) -> str:
        """
        Use LLM to generate a natural language description of a predicate.
        
        Args:
            signature: Full signature like "(holding ?x - block)"
            
        Returns:
            F-string style description like "Block {x} is held"
        """
        prompt = f"""Given a PDDL predicate with the following signature:
{signature}

Generate a natural language description that:
1. Uses f-string style placeholders (e.g., {{x}}, {{y}})
2. Describes what the predicate means in simple, clear English
3. Annotates types in a natural way (e.g., "Block {{x}}" instead of just "{{x}}")
4. Is concise (one sentence)

Return ONLY the description string, nothing else.

Examples:
- (holding ?x - block) → "Block {{x}} is held"
- (on ?x - block ?y - block) → "Block {{x}} is on block {{y}}"
- (clear ?x - block) → "Block {{x}} is clear"
- (arm-empty) → "The arm is empty"
"""
        response = self.client.responses.create(
            model="gpt-5-nano",
            input=prompt,
            text={"verbosity": "low"},
            reasoning={"effort": "low"},
        )
        
        result = response.output_text
        if result is None:
            # Fallback if LLM returns None
            return signature
        
        result = result.strip()
        # Remove surrounding quotes if present
        if result and ((result.startswith('"') and result.endswith('"')) or (result.startswith("'") and result.endswith("'"))):
            result = result[1:-1]
        
        return result if result else signature
    
    def _describe_action_with_llm(self, action_name: str, action_details: str) -> str:
        """
        Use LLM to generate a natural language description of an action.
        
        Args:
            action_name: Name of the action
            action_details: Full action definition including parameters, preconditions, and effects
            
        Returns:
            Natural language description of the action
        """
        prompt = f"""Given a PDDL action:

{action_details}

Generate a natural language description that:
1. Explains what the action does
2. Lists the requirements (preconditions) naturally
3. Describes the effects/consequences
4. Uses parameter names with curly braces for clarity (e.g., {{b1}}, {{b2}})
5. Annotates types naturally (e.g., "block {{b1}}")
6. Is clear and concise (2-4 sentences)

Return ONLY the description, nothing else.

Example format:
"Pick up block {{b1}} from block {{b2}}. Requires that {{b1}} is clear and on {{b2}}, and the arm is empty. Causes {{b1}} to be held, {{b2}} to become clear, and {{b1}} to no longer be on {{b2}}."
"""
        
        response = self.client.responses.create(
            model="gpt-5-nano",
            input=prompt,
            text={"verbosity": "low"},
            reasoning={"effort": "low"},
        )
        
        result = response.output_text
        if result is None:
            # Fallback if LLM returns None
            return action_details
        
        result = result.strip()
        # Remove surrounding quotes if present
        if result and ((result.startswith('"') and result.endswith('"')) or (result.startswith("'") and result.endswith("'"))):
            result = result[1:-1]
        
        return result if result else action_details
    
    def _get_predicates(self, problem: Problem) -> Dict[str, str]:
        """
        Extract all predicates and generate descriptions using LLM.
        
        Returns:
            Dictionary mapping predicate signatures to natural language descriptions
        """
        predicates = {}
        
        # Filter out non-boolean and total-cost fluents first
        valid_fluents = [
            f for f in problem.fluents 
            if f.type.is_bool_type() and f.name.lower().replace('-', '_') != 'total_cost'
        ]
        
        for fluent in tqdm(valid_fluents, desc="Generating predicate descriptions", unit="predicate", colour="green"):
            tqdm.write(f"Describing predicate: {fluent.name}")

            # Build signature
            if fluent.signature:
                params = []
                for param in fluent.signature:
                    param_name = param.name
                    param_type = param.type.name if hasattr(param.type, 'name') else 'object'
                    params.append(f"?{param_name} - {param_type}")
                signature = f"({fluent.name} {' '.join(params)})"
            else:
                signature = f"({fluent.name})"
            
            # Get LLM description
            description = self._describe_predicate_with_llm(signature)
            predicates[signature] = description
        
        return predicates
    
    def _get_actions(self, problem: Problem) -> Dict[str, str]:
        """
        Extract all actions and generate descriptions using LLM.
        
        Returns:
            Dictionary mapping action signatures to natural language descriptions
        """
        actions = {}
        
        for action in tqdm(problem.actions, desc="Generating action descriptions", unit="action", colour="magenta"):
            # Build signature
            if action.parameters:
                params = []
                for param in action.parameters:
                    param_name = param.name
                    param_type = param.type.name if hasattr(param.type, 'name') else 'object'
                    params.append(f"?{param_name} - {param_type}")
                signature = f"({action.name} {' '.join(params)})"
            else:
                signature = f"({action.name})"

            tqdm.write(f"Describing action: {signature}")
            
            # Build detailed action description for LLM
            details_lines = [signature]
            details_lines.append("  :parameters (" + " ".join([f"?{p.name} - {p.type.name if hasattr(p.type, 'name') else 'object'}" for p in action.parameters]) + ")")
            
            if action.preconditions:
                details_lines.append(f"  :precondition {action.preconditions}")
            
            if action.effects:
                details_lines.append(f"  :effect {action.effects}")
            
            action_details = "\n".join(details_lines)
            
            # Get LLM description
            description = self._describe_action_with_llm(action.name, action_details)
            actions[signature] = description
        
        return actions
    
    def _classify_problems_by_difficulty(self) -> dict:
        """
        Classify problems in the domain directory by difficulty based on object count.
        
        Returns:
            Dictionary with 'easy', 'medium', 'hard' keys containing lists of problem filenames
        """
        # Find all problem files
        problem_files = sorted([f for f in os.listdir(self.domain_dir) 
                                if f.startswith('p') and f.endswith('.pddl')])
        
        if not problem_files:
            return {"easy": [], "medium": [], "hard": []}
        
        # Count objects in each problem
        problem_counts = []
        for prob_file in problem_files:
            prob_path = os.path.join(self.domain_dir, prob_file)
            try:
                reader = PDDLReader()
                problem = reader.parse_problem(self.domain_path, prob_path)
                obj_count = len(list(problem.all_objects))
                problem_counts.append((prob_file, obj_count))
            except Exception as e:
                # Skip problems that can't be parsed
                continue
        
        if not problem_counts:
            return {"easy": [], "medium": [], "hard": []}
        
        # Sort by object count
        problem_counts.sort(key=lambda x: x[1])
        
        # Split into thirds
        n = len(problem_counts)
        third = n // 3
        remainder = n % 3
        
        # Distribute remainder to make splits as even as possible
        if remainder == 0:
            easy_end = third
            medium_end = 2 * third
        elif remainder == 1:
            easy_end = third + 1
            medium_end = 2 * third + 1
        else:  # remainder == 2
            easy_end = third + 1
            medium_end = 2 * third + 2
        
        easy = [p[0] for p in problem_counts[:easy_end]]
        medium = [p[0] for p in problem_counts[easy_end:medium_end]]
        hard = [p[0] for p in problem_counts[medium_end:]]
        
        return {"easy": easy, "medium": medium, "hard": hard}
    
    def generate_description(self, output_filename: str = "description.yaml"):
        """
        Generate a complete YAML description of the PDDL domain.
        
        Args:
            output_filename: Name of the YAML file (will be saved in domain directory)
        """
        # Classify problems by difficulty
        difficulties = self._classify_problems_by_difficulty()
        
        description = {
            "Types": self.types,
            "Predicates": self.predicates,
            "Actions": self.actions,
            "Problems": {
                "easy": difficulties["easy"],
                "medium": difficulties["medium"],
                "hard": difficulties["hard"]
            }
        }
        
        # Always save to the domain directory
        output_path = os.path.join(self.domain_dir, output_filename)
        
        # Save to YAML without quotes around strings
        with open(output_path, 'w') as f:
            yaml.dump(description, f, default_flow_style=False, sort_keys=False, 
                     allow_unicode=True, default_style='', width=float('inf'))
        
        print(f"Description saved to {output_path}")
        return description
    
    def get_problems_by_difficulty(self, difficulty: str) -> List[str]:
        """
        Get list of problem files for a specific difficulty level.
        
        Args:
            difficulty: One of 'easy', 'medium', or 'hard'
            
        Returns:
            List of problem filenames
        """
        difficulties = self._classify_problems_by_difficulty()
        return difficulties.get(difficulty, [])
    
    def get_problem_path(self, problem_filename: str) -> str:
        """
        Get full path to a problem file.
        
        Args:
            problem_filename: Name of the problem file (e.g., 'p01.pddl')
            
        Returns:
            Absolute path to the problem file
        """
        return os.path.join(self.domain_dir, problem_filename)
    
    def _format_predicate(self, predicate_str: str, obj_mapping: Dict = None) -> str:
        """
        Format a predicate instance using the natural language description.
        
        Args:
            predicate_str: String like "holding(block_a)" or "on(block_a, block_b)"
            obj_mapping: Optional mapping of parameter placeholders to object names
            
        Returns:
            Natural language description like "Block block_a is held"
        """
        # Parse predicate name and arguments from string format
        match = re.match(r'([a-zA-Z_-]+)\((.*?)\)', predicate_str.strip())
        if not match:
            # Handle predicates with no arguments (e.g., "arm-empty")
            predicate_name = predicate_str.strip()
            args = []
        else:
            predicate_name = match.group(1)
            args_str = match.group(2)
            args = [arg.strip() for arg in args_str.split(',')] if args_str else []
        
        # Find matching predicate signature (handle both "arm-empty" and "arm_empty" style)
        matching_desc = None
        matching_sig = None
        for sig, desc in self.predicates.items():
            # Extract predicate name from signature like "(holding ?x - block)" or "(arm-empty)"
            sig_name = re.match(r'\(([^\s\)]+)', sig)
            if sig_name:
                sig_pred_name = sig_name.group(1)
                # Match with both hyphen and underscore variations
                if (sig_pred_name == predicate_name or 
                    sig_pred_name.replace('-', '_') == predicate_name.replace('-', '_')):
                    matching_desc = desc
                    matching_sig = sig
                    break
        
        if not matching_desc:
            # Fallback if no description found
            return predicate_str
        
        # Extract parameter names from the signature
        param_names = re.findall(r'\?(\w+)', matching_sig)
        
        # Replace placeholders with actual values
        result = matching_desc
        for i, param_name in enumerate(param_names):
            if i < len(args):
                # Replace {param_name} with actual object name
                result = result.replace(f"{{{param_name}}}", args[i])
        
        return result
    
    def generate_task_prompt(self, problem_path: str) -> str:
        """
        Generate a natural language task prompt from a PDDL problem file.
        
        Args:
            problem_path: Path to the PDDL problem file
            
        Returns:
            Natural language task prompt string
        """
        # Parse the problem
        reader = PDDLReader()
        problem: Problem = reader.parse_problem(self.domain_path, problem_path)
        
        # Start building the prompt
        lines = []
        
        # Task description
        lines.append("Generate a plan for this task. Provide one action instruction per line.")
        lines.append("Format each action as: (action_name arg1 arg2 ...)")
        lines.append("")
        
        # Actions section
        lines.append("Available Actions:")
        for action_sig, action_desc in self.actions.items():
            # Keep the action description as-is (with {param} placeholders)
            lines.append(f"- {action_sig}: {action_desc}")
        lines.append("")
        
        # Objects section
        lines.append("The objects are:")
        all_objects = []
        for obj in problem.all_objects:
            all_objects.append(obj.name)
        for obj_name in all_objects:
            lines.append(f"- {obj_name}")
        lines.append("")
        
        # Initial state
        lines.append("Right now:")
        for fluent, value in problem.initial_values.items():
            if value.bool_constant_value():  # Only include true predicates
                # Format: predicate_name(arg1, arg2, ...)
                pred_name = fluent.fluent().name
                if fluent.args:
                    args_str = ", ".join([str(arg) for arg in fluent.args])
                    pred_str = f"{pred_name}({args_str})"
                else:
                    pred_str = pred_name
                
                nl_description = self._format_predicate(pred_str)
                lines.append(f"- {nl_description}")
        lines.append("")
        
        # Goal state
        lines.append("The goal is to reach a point where:")
        
        # Handle goals - check if it's an AND node with args
        for goal in problem.goals:
            if hasattr(goal, 'node_type') and str(goal.node_type) == 'OperatorKind.AND':
                # It's an AND conjunction - extract individual goals from args
                if hasattr(goal, 'args'):
                    for sub_goal in goal.args:
                        nl_desc = self._parse_goal_to_nl(sub_goal)
                        if nl_desc:
                            lines.append(f"- {nl_desc}")
            else:
                # Single goal
                nl_desc = self._parse_goal_to_nl(goal)
                if nl_desc:
                    lines.append(f"- {nl_desc}")

        lines.append("Reply only with the plan, no additional text.")
        
        return "\n".join(lines)
    
    def _parse_goal_to_nl(self, goal_fnode) -> str:
        """
        Parse a goal FNode into natural language.
        
        Args:, or None if it's a conjunction
        """
        # Get the string representation and parse it
        goal_str = str(goal_fnode)
        
        # If it's an AND conjunction, return None (parent will handle it)
        if goal_str.startswith('(') and ' and ' in goal_str:
            return None
        
        # Handle simple predicates like "on(b1, b2)"
        # Extract predicate name and arguments
        match = re.match(r'([a-zA-Z_-]+)\((.*?)\)', goal_str)
        if match:
            pred_name = match.group(1)
            args_str = match.group(2)
            args = [arg.strip() for arg in args_str.split(',')]
            
            # Build predicate string for formatting
            pred_str = f"{pred_name}({', '.join(args)})"
            return self._format_predicate(pred_str)
        else:
            # Handle no-argument predicates
            pred_name = goal_str.strip()
            return self._format_predicate(pred_name)


def describe_domain(domain_path: str, output_filename: str = "description.yaml", api_key: str = None):
    """
    Main function to describe a PDDL domain.
    
    Args:
        domain_path: Path to the PDDL domain file
        output_filename: Name of the YAML file (will be saved in domain directory)
        api_key: OpenAI API key (optional, defaults to OPENAI_API_KEY env var)
        
    Returns:
        Dictionary containing the domain description
    """
    describer = PDDLDescriber(domain_path, api_key)
    return describer.generate_description(output_filename)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python describe_pddl.py <domain_path> [output_filename]")
        sys.exit(1)
    
    domain_path = sys.argv[1]
    output_filename = sys.argv[2] if len(sys.argv) > 2 else "description.yaml"
    
    try:
        describe_domain(domain_path, output_filename)
    except UnsupportedFeatureException as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
