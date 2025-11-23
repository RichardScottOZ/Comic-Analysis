import os
from pathlib import Path

def get_env_path(var_name: str, default: str) -> Path:
    return Path(os.environ.get(var_name, default)).expanduser()

def project_root() -> Path:
    return Path(__file__).resolve().parents[3]

def paths():
    root = project_root()
    return {
        "root_dir": get_env_path("PSS_ROOT", str(root / "data" / "books")),
        "data_dir": get_env_path("PSS_DATA", str(root / "src" / "cosmo" / "data")),
        "precompute_dir": get_env_path("PSS_PRECOMPUTE", str(root / "cosmo" / "precomputed")),
        "checkpoint_dir": get_env_path("PSS_CHECKPOINTS", str(root / "cosmo" / "checkpoints")),
        "output_dir": get_env_path("PSS_OUTPUT", str(root / "cosmo" / "output")),
    }
