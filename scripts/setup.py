"""Cross-platform setup script for RAG-Tag Force.

Installs dependencies, runs data ingestion, and validates the setup.
Idempotent — safe to run multiple times.

Usage:
    python scripts/setup.py
"""

import subprocess
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent


def run_cmd(cmd: list[str], description: str) -> bool:
    """Run a command and report success/failure.

    Args:
        cmd: Command and arguments to run.
        description: Human-readable description of the step.

    Returns:
        True if command succeeded, False otherwise.
    """
    print(f"\n{'=' * 60}")
    print(f"  {description}")
    print(f"{'=' * 60}")
    try:
        result = subprocess.run(cmd, cwd=str(_project_root), check=True)
        print(f"  ✓ {description} — OK")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ {description} — FAILED (exit code {e.returncode})")
        return False
    except FileNotFoundError:
        print(f"  ✗ {description} — Command not found: {cmd[0]}")
        return False


def main() -> None:
    """Run the full setup pipeline."""
    print("🪖 RAG-Tag Force — Setup")
    print(f"   Project root: {_project_root}")

    # Check .env exists
    env_file = _project_root / ".env"
    if not env_file.exists():
        print("\n⚠️  No .env file found. Creating from .env.example...")
        example = _project_root / ".env.example"
        if example.exists():
            env_file.write_text(example.read_text())
            print("   Created .env from .env.example")
            print("   ⚠️  Edit .env and add your ANTHROPIC_API_KEY before running the app!")
        else:
            print("   ✗ No .env.example found either. Create .env manually.")

    # Step 1: Install dependencies
    ok = run_cmd(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--quiet"],
        "Installing Python dependencies",
    )
    if not ok:
        print("\n✗ Setup failed at dependency installation.")
        sys.exit(1)

    # Step 2: Run ingestion pipeline
    ok = run_cmd(
        [sys.executable, "scripts/ingest.py"],
        "Running data ingestion pipeline",
    )
    if not ok:
        print("\n✗ Setup failed at data ingestion.")
        sys.exit(1)

    # Step 3: Run tests
    print(f"\n{'=' * 60}")
    print("  Running tests")
    print(f"{'=' * 60}")
    test_result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=str(_project_root),
    )
    if test_result.returncode == 0:
        print("  ✓ All tests passed")
    else:
        print("  ⚠️ Some tests failed (non-blocking for setup)")

    # Summary
    print(f"\n{'=' * 60}")
    print("  Setup Complete!")
    print(f"{'=' * 60}")
    print("\n  To launch the app:")
    print(f"    cd {_project_root}")
    print("    python -m uvicorn src.api.main:app --port 8000  # API")
    print("    cd frontend && npm run dev                      # UI on :3000")
    print("\n  Make sure ANTHROPIC_API_KEY is set in .env")
    print()


if __name__ == "__main__":
    main()
