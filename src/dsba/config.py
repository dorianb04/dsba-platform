from pathlib import Path


def get_models_root_path() -> Path:
    """
    returns the path to the models registry directory.
    """
    path_str = "models/registry"

    models_dir = Path(path_str).expanduser().resolve(strict=False)

    return models_dir
