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
        "manual_disallows": {"Alice": ["Bob"]},
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
def sample_ty_signup() -> pd.DataFrame:
    """Sample this year's signup data."""
    return pd.DataFrame(
        {
            "person": ["Alice", "Bob", "Charlie", "Diana"],
            "is_secret_santa": [True, True, True, True],
            "other_column": ["data1", "data2", "data3", "data4"],
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
def temp_csv_files(sample_ly_gifts, sample_ty_signup):
    """Create temporary CSV files for testing."""
    db_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    ty_signup_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)

    sample_ly_gifts.to_csv(db_file.name, index=False)
    sample_ty_signup.to_csv(ty_signup_file.name, index=False)

    return Path(db_file.name), Path(ty_signup_file.name)


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for output files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        yield output_dir
