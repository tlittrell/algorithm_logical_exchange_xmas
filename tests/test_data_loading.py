from pathlib import Path

import pandas as pd
import pytest

from src.algorithm_logical_exchange_xmas.run import load_data, load_gift_preferences, load_people


class TestLoadData:
    def test_load_data_success(self, temp_csv_files, sample_config):
        """Test successful data loading."""
        db_path, signups_db_path = temp_csv_files
        eligible_people = ["Alice", "Bob", "Charlie", "Diana"]
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
        eligible_people = ["Alice", "Bob", "Charlie", "Diana"]
        current_year = sample_config["current_year"]

        with pytest.raises(Exception):  # DuckDB will raise IOException  # noqa: B017, PT011
            load_data(eligible_people, current_year, signups_db_path, Path("nonexistent.csv"))

    def test_load_data_signups_file_not_found(self, temp_csv_files, sample_config):
        """Test error handling when signups database file doesn't exist."""
        db_path, _ = temp_csv_files
        eligible_people = ["Alice", "Bob", "Charlie", "Diana"]
        current_year = sample_config["current_year"]

        with pytest.raises(Exception):  # DuckDB will raise IOException  # noqa: B017, PT011
            load_data(eligible_people, current_year, Path("nonexistent.csv"), db_path)

    def test_load_data_duplicate_ly_giver(self, temp_csv_files, sample_config):
        """Test validation fails with duplicate givers in last year's data."""
        db_path, signups_db_path = temp_csv_files
        eligible_people = ["Alice", "Bob", "Charlie", "Diana"]
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
        eligible_people = ["Alice", "Bob", "Charlie", "Diana"]
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
        eligible_people = ["Alice", "Bob", "Charlie", "Diana"]
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
        eligible_people = ["Alice", "Bob", "Charlie", "Diana"]
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
        eligible_people = ["Alice", "Bob", "Charlie", "Diana"]
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


class TestLoadGiftPreferences:
    def test_load_gift_preferences_success(self, temp_gift_preferences_file, sample_config):
        """Test successful gift preferences loading with mixed types."""
        eligible_people = ["Alice", "Bob", "Charlie", "Diana"]
        current_year = sample_config["current_year"]
        gifts_per_person = sample_config["algorithm"]["gifts_per_person"]

        disallows, assigns = load_gift_preferences(
            eligible_people, current_year, gifts_per_person, temp_gift_preferences_file
        )

        # Verify both are dictionaries
        assert isinstance(disallows, dict)
        assert isinstance(assigns, dict)

        # Should have Alice disallowing Bob and Bob disallowing Alice
        assert "Alice" in disallows
        assert "Bob" in disallows
        assert "Bob" in disallows["Alice"]
        assert "Alice" in disallows["Bob"]

        # Should have Charlie assigned to Diana
        assert "Charlie" in assigns
        assert "Diana" in assigns["Charlie"]

    def test_load_gift_preferences_file_not_found(self, sample_config):
        """Test that missing file returns empty dicts."""
        eligible_people = ["Alice", "Bob", "Charlie", "Diana"]
        current_year = sample_config["current_year"]
        gifts_per_person = sample_config["algorithm"]["gifts_per_person"]

        disallows, assigns = load_gift_preferences(
            eligible_people, current_year, gifts_per_person, Path("nonexistent_preferences.csv")
        )

        # Should return empty dicts when file doesn't exist
        assert disallows == {}
        assert assigns == {}

    def test_load_gift_preferences_no_current_year_data(self, sample_config):
        """Test that no data for current year returns empty dicts."""
        eligible_people = ["Alice", "Bob", "Charlie", "Diana"]
        gifts_per_person = sample_config["algorithm"]["gifts_per_person"]
        current_year = 2030  # Year with no data

        # Create preferences file with different year
        import tempfile

        prefs_data = pd.DataFrame(
            {
                "person": ["Alice"],
                "gift": ["Bob"],
                "preference_type": ["disallow"],
                "year": [2025],  # Different year
            }
        )
        prefs_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        prefs_data.to_csv(prefs_file.name, index=False)

        disallows, assigns = load_gift_preferences(
            eligible_people, current_year, gifts_per_person, Path(prefs_file.name)
        )

        # Should return empty dicts when no data for current year
        assert disallows == {}
        assert assigns == {}

    def test_load_gift_preferences_ineligible_person(self, sample_config):
        """Test validation fails with ineligible person in preferences."""
        eligible_people = ["Alice", "Bob", "Charlie", "Diana"]
        current_year = sample_config["current_year"]
        gifts_per_person = sample_config["algorithm"]["gifts_per_person"]

        # Create invalid preferences with ineligible person
        import tempfile

        invalid_prefs = pd.DataFrame(
            {
                "person": ["Eve"],  # Eve not in eligible_people
                "gift": ["Alice"],
                "preference_type": ["disallow"],
                "year": [current_year],
            }
        )
        prefs_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        invalid_prefs.to_csv(prefs_file.name, index=False)

        with pytest.raises(AssertionError, match="Ineligible person"):
            load_gift_preferences(eligible_people, current_year, gifts_per_person, Path(prefs_file.name))

    def test_load_gift_preferences_ineligible_gift(self, sample_config):
        """Test validation fails with ineligible gift recipient in preferences."""
        eligible_people = ["Alice", "Bob", "Charlie", "Diana"]
        current_year = sample_config["current_year"]
        gifts_per_person = sample_config["algorithm"]["gifts_per_person"]

        # Create invalid preferences with ineligible gift recipient
        import tempfile

        invalid_prefs = pd.DataFrame(
            {
                "person": ["Alice"],
                "gift": ["Eve"],  # Eve not in eligible_people
                "preference_type": ["disallow"],
                "year": [current_year],
            }
        )
        prefs_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        invalid_prefs.to_csv(prefs_file.name, index=False)

        with pytest.raises(AssertionError, match="Ineligible gift recipient"):
            load_gift_preferences(eligible_people, current_year, gifts_per_person, Path(prefs_file.name))

    def test_load_gift_preferences_duplicate_pairs(self, sample_config):
        """Test validation fails with duplicate person-gift pairs."""
        eligible_people = ["Alice", "Bob", "Charlie", "Diana"]
        current_year = sample_config["current_year"]
        gifts_per_person = sample_config["algorithm"]["gifts_per_person"]

        # Create invalid preferences with duplicate pairs
        import tempfile

        invalid_prefs = pd.DataFrame(
            {
                "person": ["Alice", "Alice"],
                "gift": ["Bob", "Bob"],  # Duplicate pair
                "preference_type": ["disallow", "disallow"],
                "year": [current_year, current_year],
            }
        )
        prefs_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        invalid_prefs.to_csv(prefs_file.name, index=False)

        with pytest.raises(AssertionError, match="Duplicate person-gift pairs"):
            load_gift_preferences(eligible_people, current_year, gifts_per_person, Path(prefs_file.name))

    def test_load_gift_preferences_multiple_disallows(self, sample_config):
        """Test that one person can have multiple disallowed recipients."""
        eligible_people = ["Alice", "Bob", "Charlie", "Diana"]
        current_year = sample_config["current_year"]
        gifts_per_person = sample_config["algorithm"]["gifts_per_person"]

        # Create preferences with Alice disallowing multiple people
        import tempfile

        prefs_data = pd.DataFrame(
            {
                "person": ["Alice", "Alice", "Alice"],
                "gift": ["Bob", "Charlie", "Diana"],
                "preference_type": ["disallow", "disallow", "disallow"],
                "year": [current_year, current_year, current_year],
            }
        )
        prefs_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        prefs_data.to_csv(prefs_file.name, index=False)

        disallows, assigns = load_gift_preferences(
            eligible_people, current_year, gifts_per_person, Path(prefs_file.name)
        )

        # Should have Alice with list of three disallowed people
        assert "Alice" in disallows
        assert len(disallows["Alice"]) == 3
        assert set(disallows["Alice"]) == {"Bob", "Charlie", "Diana"}
        assert assigns == {}  # No assigns in this test

    def test_load_gift_preferences_conflict_detection(self, sample_config):
        """Test that conflicts between disallow and assign are detected."""
        eligible_people = ["Alice", "Bob", "Charlie", "Diana"]
        current_year = sample_config["current_year"]
        gifts_per_person = sample_config["algorithm"]["gifts_per_person"]

        # Create conflicting preferences (Alice both disallows and must assign to Bob)
        import tempfile

        conflicting_prefs = pd.DataFrame(
            {
                "person": ["Alice", "Alice"],
                "gift": ["Bob", "Bob"],
                "preference_type": ["disallow", "assign"],
                "year": [current_year, current_year],
            }
        )
        prefs_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        conflicting_prefs.to_csv(prefs_file.name, index=False)

        with pytest.raises(AssertionError, match="Conflicting preferences"):
            load_gift_preferences(eligible_people, current_year, gifts_per_person, Path(prefs_file.name))

    def test_load_gift_preferences_over_constraint(self, sample_config):
        """Test that over-constraint is detected (more assigns than gifts_per_person)."""
        eligible_people = ["Alice", "Bob", "Charlie", "Diana"]
        current_year = sample_config["current_year"]
        gifts_per_person = 2  # Only 2 gifts per person

        # Create over-constrained preferences (Alice must assign to 3 people)
        import tempfile

        over_constrained_prefs = pd.DataFrame(
            {
                "person": ["Alice", "Alice", "Alice"],
                "gift": ["Bob", "Charlie", "Diana"],
                "preference_type": ["assign", "assign", "assign"],
                "year": [current_year, current_year, current_year],
            }
        )
        prefs_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        over_constrained_prefs.to_csv(prefs_file.name, index=False)

        with pytest.raises(AssertionError, match="more assign preferences than gifts_per_person"):
            load_gift_preferences(eligible_people, current_year, gifts_per_person, Path(prefs_file.name))

    def test_load_gift_preferences_assign_only(self, sample_config):
        """Test loading assign preferences only."""
        eligible_people = ["Alice", "Bob", "Charlie", "Diana"]
        current_year = sample_config["current_year"]
        gifts_per_person = sample_config["algorithm"]["gifts_per_person"]

        # Create assign-only preferences
        import tempfile

        assign_prefs = pd.DataFrame(
            {
                "person": ["Alice", "Bob"],
                "gift": ["Charlie", "Diana"],
                "preference_type": ["assign", "assign"],
                "year": [current_year, current_year],
            }
        )
        prefs_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        assign_prefs.to_csv(prefs_file.name, index=False)

        disallows, assigns = load_gift_preferences(
            eligible_people, current_year, gifts_per_person, Path(prefs_file.name)
        )

        # Should have no disallows but two assigns
        assert disallows == {}
        assert "Alice" in assigns
        assert "Bob" in assigns
        assert "Charlie" in assigns["Alice"]
        assert "Diana" in assigns["Bob"]


class TestLoadPeople:
    def test_load_people_success(self, temp_people_file):
        """Test successful people loading with all emails."""
        eligible_people, emails = load_people(temp_people_file)

        # Verify eligible people list
        assert isinstance(eligible_people, list)
        assert len(eligible_people) == 4
        assert set(eligible_people) == {"Alice", "Bob", "Charlie", "Diana"}

        # Verify emails dict
        assert isinstance(emails, dict)
        assert len(emails) == 4
        assert emails["Alice"] == "alice@example.com"
        assert emails["Bob"] == "bob@example.com"
        assert emails["Charlie"] == "charlie@example.com"
        assert emails["Diana"] == "diana@example.com"

    def test_load_people_with_missing_emails(self):
        """Test successful loading with some missing emails."""
        import tempfile

        # Create people data with some blank emails
        people_data = pd.DataFrame(
            {
                "person": ["Alice", "Bob", "Charlie", "Diana"],
                "email": ["alice@example.com", "", "charlie@example.com", ""],
            }
        )
        people_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        people_data.to_csv(people_file.name, index=False)

        eligible_people, emails = load_people(Path(people_file.name))

        # All people should be in eligible list
        assert len(eligible_people) == 4
        assert set(eligible_people) == {"Alice", "Bob", "Charlie", "Diana"}

        # Only people with emails should be in emails dict
        assert len(emails) == 2
        assert "Alice" in emails
        assert "Charlie" in emails
        assert "Bob" not in emails
        assert "Diana" not in emails

    def test_load_people_file_not_found(self):
        """Test error handling when people file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="People file not found"):
            load_people(Path("nonexistent_people.csv"))

    def test_load_people_duplicate_names(self):
        """Test validation fails with duplicate person names."""
        import tempfile

        # Create invalid people data with duplicate names
        invalid_people = pd.DataFrame(
            {
                "person": ["Alice", "Bob", "Alice"],  # Alice appears twice
                "email": [
                    "alice1@example.com",
                    "bob@example.com",
                    "alice2@example.com",
                ],
            }
        )
        people_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        invalid_people.to_csv(people_file.name, index=False)

        with pytest.raises(AssertionError, match="duplicate person names"):
            load_people(Path(people_file.name))

    def test_load_people_empty_file(self):
        """Test handling of empty CSV file."""
        import tempfile

        # Create empty CSV with just headers
        empty_people = pd.DataFrame({"person": [], "email": []})
        people_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        empty_people.to_csv(people_file.name, index=False)

        eligible_people, emails = load_people(Path(people_file.name))

        # Should return empty list and dict
        assert eligible_people == []
        assert emails == {}

    def test_load_people_whitespace_only_email(self):
        """Test that whitespace-only emails are treated as missing."""
        import tempfile

        # Create people data with whitespace-only email
        people_data = pd.DataFrame(
            {
                "person": ["Alice", "Bob"],
                "email": ["alice@example.com", "   "],  # Bob has whitespace-only email
            }
        )
        people_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        people_data.to_csv(people_file.name, index=False)

        eligible_people, emails = load_people(Path(people_file.name))

        # Bob should be in eligible_people but not in emails
        assert len(eligible_people) == 2
        assert "Bob" in eligible_people
        assert len(emails) == 1
        assert "Alice" in emails
        assert "Bob" not in emails
