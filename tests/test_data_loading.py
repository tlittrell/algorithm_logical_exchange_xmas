from pathlib import Path

import pandas as pd
import pytest

from src.algorithm_logical_exchange_xmas.run import load_data


class TestLoadData:
    def test_load_data_success(self, temp_csv_files, sample_config):
        """Test successful data loading."""
        ly_gifts_path, ty_signup_path = temp_csv_files
        eligible_people = sample_config["algorithm"]["eligible_people"]

        ly_gifts, people_signed_up = load_data(eligible_people, ly_gifts_path, ty_signup_path)

        # Verify last year's gifts DataFrame
        assert isinstance(ly_gifts, pd.DataFrame)
        assert "giver" in ly_gifts.columns
        assert "gift1" in ly_gifts.columns
        assert "gift2" in ly_gifts.columns
        assert ly_gifts["giver"].is_unique

        # Verify people signed up list
        assert isinstance(people_signed_up, list)
        assert len(people_signed_up) == 4
        assert set(people_signed_up).issubset(set(eligible_people))

    def test_load_data_ly_gifts_file_not_found(self, temp_csv_files, sample_config):
        """Test error handling when last year's gifts file doesn't exist."""
        _, ty_signup_path = temp_csv_files
        eligible_people = sample_config["algorithm"]["eligible_people"]

        with pytest.raises(FileNotFoundError):
            load_data(eligible_people, Path("nonexistent.csv"), ty_signup_path)

    def test_load_data_ty_signup_file_not_found(self, temp_csv_files, sample_config):
        """Test error handling when this year's signup file doesn't exist."""
        ly_gifts_path, _ = temp_csv_files
        eligible_people = sample_config["algorithm"]["eligible_people"]

        with pytest.raises((OSError, RuntimeError)):  # DuckDB will raise an exception
            load_data(eligible_people, ly_gifts_path, Path("nonexistent.csv"))

    def test_load_data_duplicate_ly_giver(self, temp_csv_files, sample_config):
        """Test validation fails with duplicate givers in last year's data."""
        ly_gifts_path, ty_signup_path = temp_csv_files
        eligible_people = sample_config["algorithm"]["eligible_people"]

        # Create invalid ly_gifts with duplicate giver
        invalid_ly_gifts = pd.DataFrame(
            {
                "giver": ["Alice", "Alice", "Charlie"],
                "gift1": ["Charlie", "Bob", "Bob"],
                "gift2": ["Diana", "Diana", "Alice"],
            }
        )
        invalid_ly_gifts.to_csv(ly_gifts_path, index=False)

        with pytest.raises(AssertionError):
            load_data(eligible_people, ly_gifts_path, ty_signup_path)

    def test_load_data_ineligible_ly_giver(self, temp_csv_files, sample_config):
        """Test validation fails with ineligible giver in last year's data."""
        ly_gifts_path, ty_signup_path = temp_csv_files
        eligible_people = sample_config["algorithm"]["eligible_people"]

        # Create invalid ly_gifts with ineligible giver
        invalid_ly_gifts = pd.DataFrame(
            {
                "giver": ["Eve", "Bob", "Charlie"],  # Eve not in eligible_people
                "gift1": ["Charlie", "Diana", "Bob"],
                "gift2": ["Diana", "Alice", "Alice"],
            }
        )
        invalid_ly_gifts.to_csv(ly_gifts_path, index=False)

        with pytest.raises(AssertionError, match="Ineligible LY gifter"):
            load_data(eligible_people, ly_gifts_path, ty_signup_path)

    def test_load_data_ineligible_signup(self, temp_csv_files, sample_config):
        """Test validation fails with ineligible person in signup data."""
        ly_gifts_path, ty_signup_path = temp_csv_files
        eligible_people = sample_config["algorithm"]["eligible_people"]

        # Create invalid signup with ineligible person
        invalid_signup = pd.DataFrame(
            {
                "person": ["Alice", "Bob", "Eve"],  # Eve not in eligible_people
                "is_secret_santa": [True, True, True],
                "other_column": ["data1", "data2", "data3"],
            }
        )
        invalid_signup.to_csv(ty_signup_path, index=False)

        with pytest.raises(AssertionError, match="People signed up aren't eligible"):
            load_data(eligible_people, ly_gifts_path, ty_signup_path)

    def test_load_data_duplicate_signup(self, temp_csv_files, sample_config):
        """Test validation fails with duplicate person in signup data."""
        ly_gifts_path, ty_signup_path = temp_csv_files
        eligible_people = sample_config["algorithm"]["eligible_people"]

        # Create invalid signup with duplicate person
        invalid_signup = pd.DataFrame(
            {
                "person": ["Alice", "Alice", "Bob"],
                "is_secret_santa": [True, True, True],
                "other_column": ["data1", "data2", "data3"],
            }
        )
        invalid_signup.to_csv(ty_signup_path, index=False)

        with pytest.raises(AssertionError):
            load_data(eligible_people, ly_gifts_path, ty_signup_path)

    def test_load_data_filters_non_participants(self, temp_csv_files, sample_config):
        """Test that only people with is_secret_santa=True are included."""
        ly_gifts_path, ty_signup_path = temp_csv_files
        eligible_people = sample_config["algorithm"]["eligible_people"]

        # Create signup with mixed participation
        filtered_signup = pd.DataFrame(
            {
                "person": ["Alice", "Bob", "Charlie", "Diana"],
                "is_secret_santa": [True, False, True, True],  # Bob not participating
                "other_column": ["data1", "data2", "data3", "data4"],
            }
        )
        filtered_signup.to_csv(ty_signup_path, index=False)

        ly_gifts, people_signed_up = load_data(eligible_people, ly_gifts_path, ty_signup_path)

        # Should only include Alice, Charlie, Diana (not Bob)
        assert len(people_signed_up) == 3
        assert "Bob" not in people_signed_up
        assert set(people_signed_up) == {"Alice", "Charlie", "Diana"}
