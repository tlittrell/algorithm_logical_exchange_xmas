from pathlib import Path

import pandas as pd
import pytest

from src.algorithm_logical_exchange_xmas.run import load_data


class TestLoadData:
    def test_load_data_success(self, temp_csv_files, sample_config):
        """Test successful data loading."""
        db_path, signups_db_path = temp_csv_files
        eligible_people = sample_config["algorithm"]["eligible_people"]
        current_year = sample_config["current_year"]

        ly_gifts, people_signed_up = load_data(eligible_people, current_year, signups_db_path, db_path)

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

    def test_load_data_db_file_not_found(self, temp_csv_files, sample_config):
        """Test error handling when gift database file doesn't exist."""
        _, signups_db_path = temp_csv_files
        eligible_people = sample_config["algorithm"]["eligible_people"]
        current_year = sample_config["current_year"]

        with pytest.raises(Exception):  # DuckDB will raise IOException  # noqa: B017, PT011
            load_data(eligible_people, current_year, signups_db_path, Path("nonexistent.csv"))

    def test_load_data_signups_file_not_found(self, temp_csv_files, sample_config):
        """Test error handling when signups database file doesn't exist."""
        db_path, _ = temp_csv_files
        eligible_people = sample_config["algorithm"]["eligible_people"]
        current_year = sample_config["current_year"]

        with pytest.raises(Exception):  # DuckDB will raise IOException  # noqa: B017, PT011
            load_data(eligible_people, current_year, Path("nonexistent.csv"), db_path)

    def test_load_data_duplicate_ly_giver(self, temp_csv_files, sample_config):
        """Test validation fails with duplicate givers in last year's data."""
        db_path, signups_db_path = temp_csv_files
        eligible_people = sample_config["algorithm"]["eligible_people"]
        current_year = sample_config["current_year"]

        # Create invalid database with duplicate giver
        invalid_db = pd.DataFrame(
            {
                "giver": ["Alice", "Alice", "Charlie"],
                "gift1": ["Charlie", "Bob", "Bob"],
                "gift2": ["Diana", "Diana", "Alice"],
                "year": [2024, 2024, 2024],
            }
        )
        invalid_db.to_csv(db_path, index=False)

        with pytest.raises(AssertionError):
            load_data(eligible_people, current_year, signups_db_path, db_path)

    def test_load_data_ineligible_ly_giver(self, temp_csv_files, sample_config):
        """Test validation fails with ineligible giver in last year's data."""
        db_path, signups_db_path = temp_csv_files
        eligible_people = sample_config["algorithm"]["eligible_people"]
        current_year = sample_config["current_year"]

        # Create invalid database with ineligible giver
        invalid_db = pd.DataFrame(
            {
                "giver": ["Eve", "Bob", "Charlie"],  # Eve not in eligible_people
                "gift1": ["Charlie", "Diana", "Bob"],
                "gift2": ["Diana", "Alice", "Alice"],
                "year": [2024, 2024, 2024],
            }
        )
        invalid_db.to_csv(db_path, index=False)

        with pytest.raises(AssertionError, match="Ineligible LY gifter"):
            load_data(eligible_people, current_year, signups_db_path, db_path)

    def test_load_data_ineligible_signup(self, temp_csv_files, sample_config):
        """Test validation fails with ineligible person in signup data."""
        db_path, signups_db_path = temp_csv_files
        eligible_people = sample_config["algorithm"]["eligible_people"]
        current_year = sample_config["current_year"]

        # Create invalid signup database with ineligible person
        invalid_signups = pd.DataFrame(
            {
                "person": ["Alice", "Bob", "Eve"],  # Eve not in eligible_people
                "is_secret_santa": [True, True, True],
                "is_stockings": [True, True, True],
                "year": [current_year, current_year, current_year],
            }
        )
        invalid_signups.to_csv(signups_db_path, index=False)

        with pytest.raises(AssertionError, match="People signed up aren't eligible"):
            load_data(eligible_people, current_year, signups_db_path, db_path)

    def test_load_data_duplicate_signup(self, temp_csv_files, sample_config):
        """Test validation fails with duplicate person in signup data."""
        db_path, signups_db_path = temp_csv_files
        eligible_people = sample_config["algorithm"]["eligible_people"]
        current_year = sample_config["current_year"]

        # Create invalid signup database with duplicate person
        invalid_signups = pd.DataFrame(
            {
                "person": ["Alice", "Alice", "Bob"],
                "is_secret_santa": [True, True, True],
                "is_stockings": [True, True, False],
                "year": [current_year, current_year, current_year],
            }
        )
        invalid_signups.to_csv(signups_db_path, index=False)

        with pytest.raises(AssertionError):
            load_data(eligible_people, current_year, signups_db_path, db_path)

    def test_load_data_filters_non_participants(self, temp_csv_files, sample_config):
        """Test that only people with is_secret_santa=True are included."""
        db_path, signups_db_path = temp_csv_files
        eligible_people = sample_config["algorithm"]["eligible_people"]
        current_year = sample_config["current_year"]

        # Create signup database with mixed participation
        filtered_signups = pd.DataFrame(
            {
                "person": ["Alice", "Bob", "Charlie", "Diana"],
                "is_secret_santa": [True, False, True, True],  # Bob not participating
                "is_stockings": [True, True, False, True],
                "year": [current_year, current_year, current_year, current_year],
            }
        )
        filtered_signups.to_csv(signups_db_path, index=False)

        ly_gifts, people_signed_up = load_data(eligible_people, current_year, signups_db_path, db_path)

        # Should only include Alice, Charlie, Diana (not Bob)
        assert len(people_signed_up) == 3
        assert "Bob" not in people_signed_up
        assert set(people_signed_up) == {"Alice", "Charlie", "Diana"}
