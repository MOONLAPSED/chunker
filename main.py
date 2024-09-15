import asyncio
import logging
import platform
from dataclasses import dataclass, field
from functools import partial
from struct import Struct
from typing import Any, Callable, Dict, TypeVar, List, Tuple
from typing import Union, Type, Optional, ClassVar, Generic
from enum import Enum, auto
from pathlib import Path
from abc import ABC, abstractmethod
import argparse
import os
import subprocess
import sys
import logging
import pathlib

"""
We can assume that imperative deterministic source code, such as this file written in Python, is capable of reasoning about non-imperative non-deterministic source code as if it were a defined and known quantity. This is akin to nesting a function with a value in an S-Expression.

In order to expect any runtime result, we must assume that a source code configuration exists which will yield that result given the input.

The source code configuration is the set of all possible configurations of the source code. It is the union of the possible configurations of the source code.

This source code file can reason about or make assumptions about dynamical source code as if it were a defined and known quantity, like nesting a function with a value in an S-Expression.

Imperative programming specifies how to perform tasks (like procedural code), while non-imperative (e.g., functional programming in LISP) focuses on what to compute. We turn this on its head in our imperative non-imperative runtime by utilizing nominative homoiconistic reflection to create a runtime where dynamical source code is treated as both static and dynamic.

"Nesting a function with a value in an S-Expression":
    In the code, we nest the input value within different function expressions (configurations).
    Each function is applied to the input to yield results, mirroring the collapse of the wave function to a specific state upon measurement.

This nominative homoiconistic reflection combines the expressiveness of S-Expressions with the operational semantics of Python. In this paradigm, source code can be constructed, deconstructed, and analyzed in real-time, allowing for dynamic composition and execution. Each code configuration (or state) is akin to a function in an S-Expression that can be encapsulated, manipulated, and ultimately evaluated in the course of execution.

To illustrate, consider a Python function as a generalized S-Expression. This function can take other functions and values as arguments, forming a nested structure. Each invocation changes the system's state temporarily, just as evaluating an S-Expression alters the state of the LISP interpreter.

In essence, our approach ensures that:

1. **Composition**: Functions (or code segments) can be composed at runtime, akin to how S-Expressions can nest functions and values.
2. **Evaluation**: Upon invocation, these compositions are evaluated, reflecting the current configuration of the runtime.
3. **Reflection and Modification**: The runtime can reflect on its structure and make modifications dynamically, which allows it to reason about its state and adapt accordingly.

This synthesis of static and dynamic code concepts is akin to the Copenhagen interpretation of quantum mechanics, where the observation (or execution) collapses the superposition of states (or configurations) into a definite outcome based on the input.

Ultimately, this model provides a flexible approach to managing and executing complex code structures dynamically while maintaining the clarity and compositional advantages traditionally seen in non-imperative, functional paradigms like LISP, drawing inspiration from lambda calculus and functional programming principles.

The most advanced concept of all in this ontology is the dynamic rewriting of source code at runtime. Source code rewriting is achieved with a special runtime `Atom()` class with 'modified quine' behavior. This special Atom, aside from its specific function and the functions obligated to it by polymorphism, will always rewrite its own source code but may also perform other actions as defined by the source code in the runtime which invoked it. They can be nested in S-expressions and are homoiconic with all other source code. These modified quines can be used to dynamically create new code at runtime, which can be used to extend the source code in a way that is not known at the start of the program. This is the most powerful feature of the system and allows for the creation of a runtime of runtimes dynamically limited by hardware and the operating system.
"""
# non-homoiconic pre-runtime "ADMIN-SCOPED" source code:
logging.basicConfig(level=logging.INFO)
@dataclass
class AppState:
    pdm_installed: bool = False
    virtualenv_created: bool = False
    dependencies_installed: bool = False
    lint_passed: bool = False
    code_formatted: bool = False
    tests_passed: bool = False
    benchmarks_run: bool = False
    pre_commit_installed: bool = False

async def setup_app(state: AppState):
    logging.info("Starting app setup")

    if not state.pdm_installed:
        await ensure_pdm(state)
    if not state.virtualenv_created:
        await ensure_virtualenv(state)
    if not state.dependencies_installed:
        await ensure_dependencies(state)

    mode = await prompt_for_mode()
    if mode == "d":
        await ensure_lint(state)
        await ensure_format(state)
        await ensure_tests(state)
        await ensure_benchmarks(state)
        await ensure_pre_commit(state)

    logging.info("App setup complete")

async def ensure(state: AppState, attribute: str, func: Callable):
    if not getattr(state, attribute):
        await func(state)

async def run_command_async(command: str, shell: bool = False, timeout: int = 120) -> dict:
    logging.info(f"Running command: {command}")
    try:
        process = await asyncio.create_subprocess_shell(
            command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, shell=shell
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        return_code = process.returncode
        return {
            "return_code": return_code,
            "output": stdout.decode() if stdout else "",
            "error": stderr.decode() if stderr else "",
        }
    except asyncio.TimeoutError:
        logging.error(f"Command '{command}' timed out.")
        return {"return_code": -1, "output": "", "error": "Command timed out"}
    except Exception as e:
        logging.error(f"Error running command '{command}': {str(e)}")
        return {"return_code": -1, "output": "", "error": str(e)}

async def ensure_pdm(state: AppState):
    result = await run_command_async("pdm --version", shell=True)
    if result["return_code"] == 0:
        state.pdm_installed = True
        logging.info("PDM is already installed.")
    else:
        logging.info("PDM not found. Installing...")
        await run_command_async("pip install pdm", shell=True)
        state.pdm_installed = True

async def ensure_virtualenv(state: AppState):
    if not state.virtualenv_created:
        logging.info("Creating virtual environment with pdm")
        result = await run_command_async("pdm venv create", shell=True)
        if result["return_code"] == 0:
            state.virtualenv_created = True
            logging.info("Virtual environment created.")
        else:
            logging.error("Failed to create virtual environment.")
            state.virtualenv_created = False

async def ensure_dependencies(state: AppState):
    logging.info("Installing dependencies")
    result = await run_command_async("pdm install", shell=True)
    if result["return_code"] == 0:
        state.dependencies_installed = True
    else:
        logging.error("Failed to install dependencies")
        state.dependencies_installed = False

async def ensure_lint(state: AppState):
    logging.info("Running linting tools")
    result = await run_command_async("pdm run flake8 .", shell=True)
    if result["return_code"] == 0:
        result = await run_command_async("pdm run black --check .", shell=True)
    if result["return_code"] == 0:
        result = await run_command_async("pdm run mypy .", shell=True)
    state.lint_passed = result["return_code"] == 0

async def ensure_format(state: AppState):
    await run_command_async("pdm run black .", shell=True)
    await run_command_async("pdm run isort .", shell=True)
    state.code_formatted = True

async def ensure_tests(state: AppState):
    await run_command_async("pdm run pytest", shell=True)
    state.tests_passed = True

async def ensure_benchmarks(state: AppState):
    await run_command_async("pdm run python src/bench/bench.py", shell=True)
    state.benchmarks_run = True

async def ensure_pre_commit(state: AppState):
    await run_command_async("pdm run pre-commit install", shell=True)
    state.pre_commit_installed = True

async def prompt_for_mode() -> str:
    mode = await asyncio.to_thread(input, "Choose setup mode: [d]evelopment or [n]on-development? ")
    return mode.lower()


def safe_remove_path(path: Path):
    """Safely remove a file or directory after validation."""
    try:
        # Verify path is within allowed directory (e.g., project root)
        allowed_root = Path(__file__).resolve().parents[2]
        if not path.resolve().is_relative_to(allowed_root):
            logging.error(f"Attempt to delete outside allowed directory: {path}")
            return
        
        # Confirm deletion
        confirm = input(f"Are you sure you want to delete {path}? (y/N): ").strip().lower()
        if confirm != 'y':
            logging.info(f"Skipping deletion of {path}")
            return

        if path.is_dir():
            path.rmdir()
            logging.info(f"Removed directory: {path}")
        else:
            path.unlink()
            logging.info(f"Removed file: {path}")

    except FileNotFoundError:
        logging.error(f"Path not found: {path}")
    except PermissionError:
        logging.error(f"Permission denied: {path}")
    except Exception as e:
        logging.error(f"Error removing path {path}: {e}")

def cleanup():
    """Clean up project artifacts with additional safety checks."""
    logging.info("Starting cleanup")

    project_root = Path(__file__).resolve().parents[2]

    paths_to_remove = [
        project_root / ".venv",
        project_root / "build",
        project_root / "dist",
        project_root / ".pytest_cache",
        project_root / ".mypy_cache",
        project_root / ".tox",
        project_root / "cognosis.egg-info",
        project_root / ".pdm.toml",
        project_root / ".pdm-build",
        project_root / ".pdm-python",
        project_root / "pdm.lock",
    ]

    for path in paths_to_remove:
        if path.exists():
            safe_remove_path(path)

    # Remove __pycache__ directories and .pyc files
    for root, dirs, files in os.walk(project_root, topdown=False):
        for name in dirs:
            if name == "__pycache__":
                safe_remove_path(Path(root) / name)
        for name in files:
            if name.endswith(".pyc"):
                safe_remove_path(Path(root) / name)

async def remove_conda_env():
    """Remove the conda environment."""
    try:
        conda_env_path = Path(os.environ.get("CONDA_PREFIX", ""))
        if not conda_env_path:
            print("No active conda environment detected.")
            return

        env_name = conda_env_path.name

        # Check if it's a base environment
        base_env_names = ["base", "miniconda", "anaconda"]
        if any(base_name in env_name.lower() for base_name in base_env_names):
            print(f"Cannot remove the base conda environment: {env_name}")
            return

        # Deactivate the current environment
        result = await run_command_async("conda deactivate", shell=True)
        if result["return_code"] != 0:
            print(f"Failed to deactivate conda environment: {result['error']}")
            return

        # Remove the environment
        result = await run_command_async(f"conda env remove -n {env_name} --yes", shell=True)
        if result["return_code"] == 0:
            print(f"Removed conda environment: {env_name}")
        else:
            print(f"Failed to remove conda environment: {result['error']}")

    except KeyError as e:
        print(f"Error accessing environment variable: {e}")
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while removing conda environment: {e}")

def main():
    confirm = input(
        "This will remove all project artifacts and installations. Are you sure? (y/N): "
    )
    if confirm.lower() == "y":
        cleanup()
        try:
            asyncio.run(remove_conda_env())
        except Exception as e:
            print(f"Failed to remove conda environment: {e}")
    else:
        print("Cleanup aborted.")
    sys.exit(0)

if __name__ == "__main__":
    main()
