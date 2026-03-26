from pathlib import Path
import subprocess
import sys




def run_cp_runner(root_dir: Path, config_path: Path, ) -> None:
    
    python_path = _get_cp_runner_python()
    
    cmd = [
        python_path.as_posix(),
        "-m",
        "cp_runner.cli",
        root_dir.as_posix(),
        "--config",
        config_path.as_posix(),
    ]

    completed = subprocess.run(cmd, capture_output=True, text=True)

    if completed.returncode != 0:
        raise RuntimeError(f"cp_runner failed for {root_dir}\n"
                            f"STDOUT:\n{completed.stdout}\n"
                            f"STDERR:\n{completed.stderr}")
    

def _get_cp_runner_python() -> Path:
    venv_path = _get_cp_runner_venv()
    
    if sys.platform == "win32":
        python_path = venv_path / "Scripts" / "python.exe"
    else:
        python_path = venv_path / "bin" / "python"

    if not python_path.exists():
        raise FileNotFoundError(f"cp_runner python not found at {python_path}")

    return python_path

def _find_repo_root(start: Path | None = None) -> Path:
    start = start or Path(__file__).resolve()

    for parent in [start, *start.parents]:
        pyproject = parent / "pyproject.toml"
        if pyproject.exists():
            # check that cp_runner exists at this level
            if (parent / "cp_runner").exists():
                return parent

    raise RuntimeError("Could not find repo root with cp_runner")

def _get_cp_runner_venv() -> Path:
    venv = _find_repo_root() / "cp_runner" / ".venv"
    if not venv.exists():
        raise FileNotFoundError(f"cp_runner venv not found at {venv}")
    return venv