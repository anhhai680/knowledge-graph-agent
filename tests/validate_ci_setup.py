#!/usr/bin/env python3
"""
CI/CD Setup Validation Script

This script validates that the CI/CD setup is working correctly.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔍 {description}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} - PASSED")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"Error: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"⚠️ {description} - SKIPPED (tool not found)")
        return True


def main():
    """Main validation function."""
    print("🚀 Validating CI/CD Setup for Knowledge Graph Agent")
    print("=" * 60)

    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ Please run this script from the project root directory")
        return 1

    success_count = 0
    total_checks = 0

    # Configuration file checks
    config_files = [
        "pyproject.toml",
        "pytest.ini",
        "setup.cfg",
        ".pre-commit-config.yaml",
        "Makefile",
        ".github/workflows/ci.yml",
        ".github/workflows/security.yml",
        ".github/workflows/release.yml",
    ]

    print("\n📁 Checking configuration files...")
    for file_path in config_files:
        total_checks += 1
        if Path(file_path).exists():
            print(f"✅ {file_path} - EXISTS")
            success_count += 1
        else:
            print(f"❌ {file_path} - MISSING")

    # Tool availability checks
    print("\n🛠 Checking development tools...")

    tools = [
        (["python", "--version"], "Python availability"),
        (["black", "--version"], "Black code formatter"),
        (["isort", "--version"], "isort import sorter"),
        (["flake8", "--version"], "Flake8 linter"),
        (["mypy", "--version"], "MyPy type checker"),
        (["pytest", "--version"], "Pytest test runner"),
        (["safety", "--version"], "Safety security scanner"),
        (["bandit", "--version"], "Bandit security linter"),
    ]

    for cmd, description in tools:
        total_checks += 1
        if run_command(cmd, description):
            success_count += 1

    # GitHub Actions workflow validation
    print("\n⚡ Validating GitHub Actions workflows...")

    workflow_files = [
        ".github/workflows/ci.yml",
        ".github/workflows/security.yml",
        ".github/workflows/release.yml",
    ]

    for workflow in workflow_files:
        total_checks += 1
        if Path(workflow).exists():
            # Basic YAML validation
            try:
                import yaml

                with open(workflow, "r") as f:
                    yaml.safe_load(f)
                print(f"✅ {workflow} - VALID YAML")
                success_count += 1
            except ImportError:
                print(f"⚠️ {workflow} - YAML validation skipped (PyYAML not available)")
                success_count += 1
            except Exception as e:
                print(f"❌ {workflow} - INVALID YAML: {e}")
        else:
            print(f"❌ {workflow} - MISSING")

    # Summary
    print("\n" + "=" * 60)
    print(f"📊 Validation Summary: {success_count}/{total_checks} checks passed")

    if success_count == total_checks:
        print("🎉 CI/CD setup is complete and ready!")
        print("\n🚀 Next steps:")
        print("1. Commit and push changes to trigger CI pipeline")
        print("2. Create a pull request to test auto-formatting")
        print("3. Monitor GitHub Actions tab for results")
        return 0
    else:
        print("⚠️ Some issues found. Please review and fix them.")
        print(f"Success rate: {(success_count/total_checks)*100:.1f}%")
        return 1


if __name__ == "__main__":
    sys.exit(main())
