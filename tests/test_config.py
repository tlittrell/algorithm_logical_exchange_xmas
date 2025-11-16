import tempfile
from pathlib import Path

import pytest

from src.algorithm_logical_exchange_xmas.run import load_config, validate_config


class TestLoadConfig:
    def test_load_config_success(self, temp_config_file):
        """Test successful config loading."""
        config = load_config(temp_config_file)

        assert "seed" in config
        assert "algorithm" in config
        assert "emails" in config
        assert config["seed"] == 123

    def test_load_config_file_not_found(self):
        """Test error handling when config file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_config(Path("nonexistent.toml"))

    def test_load_config_invalid_toml(self):
        """Test error handling with invalid TOML syntax."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("invalid toml syntax [[[")
            invalid_file = Path(f.name)

        with pytest.raises((OSError, ValueError)):  # tomllib.TOMLDecodeError
            load_config(invalid_file)


class TestValidateConfig:
    def test_validate_config_success(self, sample_config):
        """Test successful config validation."""
        # Should not raise any exceptions
        validate_config(sample_config)

    def test_validate_config_duplicate_eligible_people(self, sample_config):
        """Test validation fails with duplicate eligible people."""
        sample_config["algorithm"]["eligible_people"] = ["Alice", "Bob", "Alice"]

        with pytest.raises(AssertionError, match="eligible people contains duplicates"):
            validate_config(sample_config)

    def test_validate_config_duplicate_couples(self, sample_config):
        """Test validation fails with duplicate people in couples."""
        sample_config["algorithm"]["couples"] = [["Alice", "Bob"], ["Alice", "Charlie"]]

        with pytest.raises(AssertionError, match="Couples contains duplicates"):
            validate_config(sample_config)

    def test_validate_config_couples_not_eligible(self, sample_config):
        """Test validation fails when couples contain ineligible people."""
        sample_config["algorithm"]["couples"] = [["Alice", "Eve"]]  # Eve not in eligible_people

        with pytest.raises(AssertionError):
            validate_config(sample_config)

    def test_validate_config_duplicate_families(self, sample_config):
        """Test validation fails with duplicate people in families."""
        sample_config["algorithm"]["families"] = [["Alice", "Bob"], ["Bob", "Charlie"]]

        with pytest.raises(AssertionError, match="Families contains duplicates"):
            validate_config(sample_config)

    def test_validate_config_families_not_complete(self, sample_config):
        """Test validation fails when not everyone is assigned a family."""
        sample_config["algorithm"]["families"] = [["Alice", "Bob"]]  # Missing Charlie, Diana

        with pytest.raises(AssertionError, match="Not everyone assigned a family"):
            validate_config(sample_config)

    def test_validate_config_negative_seed(self, sample_config):
        """Test validation fails with negative seed."""
        sample_config["seed"] = -1

        with pytest.raises(AssertionError):
            validate_config(sample_config)

    def test_validate_config_non_integer_seed(self, sample_config):
        """Test validation fails with non-integer seed."""
        sample_config["seed"] = "not_an_integer"

        with pytest.raises(TypeError):
            validate_config(sample_config)

    def test_validate_config_missing_algorithm_section(self, sample_config):
        """Test validation fails when algorithm section is missing."""
        del sample_config["algorithm"]

        with pytest.raises(KeyError):
            validate_config(sample_config)

    def test_validate_config_missing_required_fields(self, sample_config):
        """Test validation fails when required fields are missing."""
        del sample_config["algorithm"]["eligible_people"]

        with pytest.raises(KeyError):
            validate_config(sample_config)

    def test_validate_config_with_max_total_intra_family_gifts(self, sample_config):
        """Test validation with max_total_intra_family_gifts parameter."""
        sample_config["algorithm"]["max_total_intra_family_gifts"] = 5
        # Should not raise any exceptions
        validate_config(sample_config)

    def test_validate_config_invalid_max_total_intra_family_gifts_type(self, sample_config):
        """Test validation fails with non-integer max_total_intra_family_gifts."""
        sample_config["algorithm"]["max_total_intra_family_gifts"] = "not_an_int"

        with pytest.raises(AssertionError, match="must be an integer"):
            validate_config(sample_config)

    def test_validate_config_negative_max_total_intra_family_gifts(self, sample_config):
        """Test validation fails with negative max_total_intra_family_gifts."""
        sample_config["algorithm"]["max_total_intra_family_gifts"] = -1

        with pytest.raises(AssertionError, match="must be non-negative"):
            validate_config(sample_config)
