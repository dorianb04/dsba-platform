import logging
from pathlib import Path

import yaml

# Assume config.yaml is in the project root, relative to this file's location
# src/dsba/config.py -> project_root
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

logger = logging.getLogger(__name__)

_config_cache = None


def load_config() -> dict:
    """
    Loads the configuration from config.yaml located in the project root.
    Caches the loaded configuration to avoid repeated file reads.

    Returns:
        dict: The loaded configuration.

    Raises:
        FileNotFoundError: If config.yaml is not found.
        yaml.YAMLError: If there's an error parsing the config file.
    """
    global _config_cache  # noqa: PLW0603
    if _config_cache:
        return _config_cache

    if not CONFIG_PATH.is_file():
        logger.error(f"Configuration file not found at {CONFIG_PATH}.")
        # Reformat the multi-line string assignment for clarity and length
        part1 = f"Configuration file not found at {CONFIG_PATH}. "
        # Corrected E501: Split the long string literal for part2
        part2 = (
            "Please create it by copying 'config.example.yaml' "
            "and filling in your values."
        )
        error_msg = part1 + part2
        raise FileNotFoundError(error_msg)

    try:
        # Ensure file is opened with read mode explicitly for clarity
        # though it's default
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f)
        if config is None:  # Handle empty config file case
            config = {}
        _config_cache = config
        logger.info(f"Configuration loaded successfully from {CONFIG_PATH}")
        return config
    except yaml.YAMLError as e:
        logger.exception(f"Error parsing configuration file {CONFIG_PATH}: {e}")
        # Corrected B904: Use 'raise from e'
        raise yaml.YAMLError(
            f"Error parsing configuration file {CONFIG_PATH}: {e}"
        ) from e
    except Exception as e:
        logger.exception(f"An unexpected error occurred while loading config: {e}")
        raise  # Keep original exception context implicitly


def get_models_root_path() -> Path:
    """
    Gets the resolved, absolute path for storing models from the config file.

    Returns:
        Path: The absolute path to the models' root directory.

    Raises:
        ValueError: If the path is not configured in config.yaml.
        FileNotFoundError: If config.yaml doesn't exist.
    """
    config = load_config()
    try:
        models_path_str = config["registry"]["models_root_path"]
        if not models_path_str:
            raise ValueError("models_root_path is empty in config.yaml")
        # Resolve ~ and make absolute
        models_dir = Path(models_path_str).expanduser().resolve(strict=False)
        return models_dir
    except KeyError as e:
        logger.error("Missing 'registry' or 'models_root_path' in config.yaml")
        # Corrected B904: Use 'raise from e'
        raise ValueError(
            "Configuration file must contain ['registry']['models_root_path']"
        ) from e
    except TypeError as e:
        logger.error("Invalid structure for 'registry' section in config.yaml")
        # Corrected B904: Use 'raise from e'
        raise ValueError(
            "Invalid structure for 'registry' section in config.yaml"
        ) from e
