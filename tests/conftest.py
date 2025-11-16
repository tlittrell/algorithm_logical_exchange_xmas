import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import pytest


@pytest.fixture
def sample_config() -> dict[str, Any]:
    """Sample configuration for testing."""
    return {
        "seed": 123,
        "current_year": 2025,
        "emails": {
            "Alice": "alice@example.com",
            "Bob": "bob@example.com",
            "Charlie": "charlie@example.com",
            "Diana": "diana@example.com",
        },
        "algorithm": {
            "eligible_people": ["Alice", "Bob", "Charlie", "Diana"],
            "couples": [["Alice", "Bob"], ["Charlie", "Diana"]],
            "families": [["Alice", "Bob"], ["Charlie", "Diana"]],
            "message": "Hello {giver}, give to {gift1} and {gift2}!",
            "gifts_per_person": 2,
            "max_gifts_to_family": 1,
            "max_gifts_from_family": 1,
            "max_couple_overlap": 1,
        },
    }


@pytest.fixture
def sample_ly_gifts() -> pd.DataFrame:
    """Sample last year's gifts data."""
    return pd.DataFrame(
        {
            "giver": ["Alice", "Bob", "Charlie"],
            "gift1": ["Charlie", "Diana", "Bob"],
            "gift2": ["Diana", "Alice", "Alice"],
            "year": [2024, 2024, 2024],
        }
    )


@pytest.fixture
def sample_signups() -> pd.DataFrame:
    """Sample signup database data."""
    return pd.DataFrame(
        {
            "person": ["Alice", "Bob", "Charlie", "Diana"],
            "is_secret_santa": [True, True, True, True],
            "is_stockings": [True, True, False, True],
            "year": [2025, 2025, 2025, 2025],
        }
    )


@pytest.fixture
def temp_config_file(sample_config):
    """Create a temporary TOML config file."""
    import tomli_w

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False) as f:
        tomli_w.dump(sample_config, f)
        return Path(f.name)


@pytest.fixture
def temp_csv_files(sample_ly_gifts, sample_signups):
    """Create temporary database files for testing."""
    db_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    signups_db_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)

    sample_ly_gifts.to_csv(db_file.name, index=False)
    sample_signups.to_csv(signups_db_file.name, index=False)

    return Path(db_file.name), Path(signups_db_file.name)


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for output files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        yield output_dir


@pytest.fixture
def sample_gift_preferences() -> pd.DataFrame:
    """Sample gift preferences data with mixed disallow and assign types."""
    return pd.DataFrame(
        {
            "person": ["Alice", "Bob", "Charlie"],
            "gift": ["Bob", "Alice", "Diana"],
            "preference_type": ["disallow", "disallow", "assign"],
            "year": [2025, 2025, 2025],
        }
    )


@pytest.fixture
def temp_gift_preferences_file(sample_gift_preferences):
    """Create a temporary gift preferences file for testing."""
    prefs_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    sample_gift_preferences.to_csv(prefs_file.name, index=False)
    return Path(prefs_file.name)
