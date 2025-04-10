from pathlib import Path

import pytest
import yaml

# Ensure imports work correctly relative to project root/src layout
from dsba.config import get_models_root_path, load_config

# Test Suite for Configuration Loading


# Create a fixture for a temporary valid config file
@pytest.fixture(scope="function")
def temp_config_file(tmp_path):
    config_content = {
        "registry": {"models_root_path": str(tmp_path / "test_models")},
        "azure": {
            "subscription_id": "test-sub-id",
            "resource_group_name": "test-rg",
            "location": "TestLocation",
            "storage_account_name": "teststorage",
            "config_fileshare_name": "testconfigshare",
            "models_fileshare_name": "testmodelshare",
            "aci_cpu_cores": 1.0,
            "aci_memory_gb": 1.0,
        },
    }
    # Use a temporary path for the config file itself during testing
    config_path = tmp_path / "temp_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_content, f)

    # Make the configured models directory
    (tmp_path / "test_models").mkdir()

    # Patch the CONFIG_PATH used by the dsba.config module for this test
    # Import the module where the constant is defined/used
    dsba_config_module = __import__(
        "dsba.config", fromlist=["CONFIG_PATH", "_config_cache"]
    )
    original_path = dsba_config_module.CONFIG_PATH
    dsba_config_module.CONFIG_PATH = config_path
    # Also reset cache before test
    dsba_config_module._config_cache = None

    yield config_path  # Provide the path to the test

    # Teardown: Restore original path and clear cache after test
    dsba_config_module.CONFIG_PATH = original_path
    dsba_config_module._config_cache = None


def test_load_config_success(temp_config_file):
    """Test loading a valid configuration file."""
    config = load_config()
    assert "registry" in config
    assert "azure" in config
    # Check content based on the fixture's definition
    assert config["registry"]["models_root_path"] == str(
        Path(temp_config_file).parent / "test_models"
    )


def test_get_models_root_path_success(temp_config_file):
    """Test getting the models root path successfully."""
    path = get_models_root_path()
    assert isinstance(path, Path)
    assert path.is_absolute()
    assert path.name == "test_models"
    assert path.exists()  # Fixture should create it
    assert path.is_dir()


def test_load_config_file_not_found(tmp_path):
    """Test behavior when config file does not exist."""
    # Patch CONFIG_PATH to a non-existent file
    dsba_config_module = __import__(
        "dsba.config", fromlist=["CONFIG_PATH", "_config_cache"]
    )
    original_path = dsba_config_module.CONFIG_PATH
    dsba_config_module.CONFIG_PATH = tmp_path / "non_existent_config.yaml"
    dsba_config_module._config_cache = None

    with pytest.raises(FileNotFoundError):
        load_config()

    # Restore
    dsba_config_module.CONFIG_PATH = original_path
    dsba_config_module._config_cache = None


def test_get_models_path_missing_key(tmp_path):
    """Test getting model path when key is missing in config."""
    config_content = {"azure": {}}  # Missing 'registry'
    config_path = tmp_path / "invalid_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_content, f)

    # Patch
    dsba_config_module = __import__(
        "dsba.config", fromlist=["CONFIG_PATH", "_config_cache"]
    )
    original_path = dsba_config_module.CONFIG_PATH
    dsba_config_module.CONFIG_PATH = config_path
    dsba_config_module._config_cache = None

    with pytest.raises(ValueError, match="Configuration file must contain"):
        get_models_root_path()

    # Restore
    dsba_config_module.CONFIG_PATH = original_path
    dsba_config_module._config_cache = None
