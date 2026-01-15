"""
Generate description.yaml files for all PDDL domains.

This script iterates through all domains in the pddl/ directory and generates
natural language descriptions using the PDDLDescriber class.
"""

import sys
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from describe_pddl import PDDLDescriber, UnsupportedFeatureException


def generate_all_descriptions(force_regenerate: bool = False):
    """
    Generate descriptions for all PDDL domains.
    
    Args:
        force_regenerate: If True, regenerate even if description.yaml exists
    """
    pddl_dir = Path("pddl")
    
    if not pddl_dir.exists():
        print("Error: pddl/ directory not found")
        return
    
    # Get all domain directories
    domains = sorted([d for d in pddl_dir.iterdir() if d.is_dir()])
    
    print(f"Found {len(domains)} domains")
    print("=" * 80)
    
    successful = []
    skipped = []
    failed = []
    
    for domain_path in tqdm(domains, desc="Processing domains", unit="domain"):
        domain_name = domain_path.name
        description_file = domain_path / "description.yaml"
        
        # Skip if already exists and not forcing regeneration
        if description_file.exists() and not force_regenerate:
            tqdm.write(f"⊘ {domain_name}: description.yaml already exists (skipping)")
            skipped.append(domain_name)
            continue
        
        try:
            tqdm.write(f"⟳ {domain_name}: Generating description...")
            describer = PDDLDescriber(str(domain_path))
            describer.generate_description()
            tqdm.write(f"✓ {domain_name}: Successfully generated")
            successful.append(domain_name)
            
        except UnsupportedFeatureException as e:
            tqdm.write(f"✗ {domain_name}: Unsupported features - {e}")
            failed.append((domain_name, str(e)))
            
        except Exception as e:
            tqdm.write(f"✗ {domain_name}: Error - {e}")
            failed.append((domain_name, str(e)))
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total domains: {len(domains)}")
    print(f"Successfully generated: {len(successful)}")
    print(f"Skipped (already exists): {len(skipped)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print(f"\n✓ Successfully generated descriptions for:")
        for name in successful:
            print(f"  - {name}")
    
    if skipped:
        print(f"\n⊘ Skipped (use --force to regenerate):")
        for name in skipped:
            print(f"  - {name}")
    
    if failed:
        print(f"\n✗ Failed domains:")
        for name, error in failed:
            print(f"  - {name}: {error}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate description.yaml files for all PDDL domains"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate descriptions even if they already exist"
    )
    
    args = parser.parse_args()
    
    print("PDDL Domain Description Generator")
    print("=" * 80)
    
    generate_all_descriptions(force_regenerate=args.force)
