import pandas as pd
import pytest

from src.algorithm_logical_exchange_xmas.run import generate_output


class TestGenerateOutput:
    def test_generate_output_creates_files(self, temp_output_dir, sample_config, sample_ly_gifts):
        """Test that output generation creates database and markdown files."""
        # Create sample result data
        result = pd.DataFrame(
            {
                "giver": ["Alice", "Bob", "Charlie"],
                "gift1": ["Bob", "Charlie", "Alice"],
                "gift2": ["Charlie", "Alice", "Bob"],
                "gift1_ly": [None, None, None],
                "gift2_ly": [None, None, None],
            }
        )

        message_template = "Hello {giver}, give to {gift1} and {gift2}!"
        emails = {
            "Alice": "alice@example.com",
            "Bob": "bob@example.com",
            "Charlie": "charlie@example.com",
            "Diana": "diana@example.com",
        }
        current_year = sample_config["current_year"]

        db_path = temp_output_dir / "secret_santa_db.csv"
        messages_path = temp_output_dir / "messages.md"

        # Create initial database with previous year's data
        sample_ly_gifts.to_csv(db_path, index=False)

        # Call generate_output
        generate_output(result, message_template, emails, current_year, messages_path, db_path)

        # Verify files were created
        assert db_path.exists()
        assert messages_path.exists()

    def test_generate_output_database_update(self, temp_output_dir, sample_config, sample_ly_gifts):
        """Test that database is correctly updated with new year's data."""
        result = pd.DataFrame(
            {
                "giver": ["Alice", "Bob"],
                "gift1": ["Bob", "Alice"],
                "gift2": ["Charlie", "Diana"],
                "gift1_ly": ["Old1", "Old2"],
                "gift2_ly": ["Old3", "Old4"],
            }
        )

        current_year = sample_config["current_year"]
        db_path = temp_output_dir / "secret_santa_db.csv"
        messages_path = temp_output_dir / "messages.md"

        # Create initial database with previous year's data
        sample_ly_gifts.to_csv(db_path, index=False)

        emails = {
            "Alice": "alice@example.com",
            "Bob": "bob@example.com",
            "Charlie": "charlie@example.com",
            "Diana": "diana@example.com",
        }
        generate_output(result, "template", emails, current_year, messages_path, db_path)

        # Read back the database and verify content
        saved_db = pd.read_csv(db_path)

        # Should have both years' data
        assert len(saved_db) == len(sample_ly_gifts) + len(result)

        # Verify new year's data
        new_year_data = saved_db[saved_db["year"] == current_year]
        assert len(new_year_data) == len(result)
        assert set(new_year_data["giver"]) == set(result["giver"])

    def test_generate_output_database_rerun(self, temp_output_dir, sample_config, sample_ly_gifts):
        """Test that re-running for the same year replaces existing data."""
        current_year = sample_config["current_year"]
        db_path = temp_output_dir / "secret_santa_db.csv"
        messages_path = temp_output_dir / "messages.md"

        # Create initial database with both years' data
        initial_db = pd.concat(
            [
                sample_ly_gifts,
                pd.DataFrame(
                    {
                        "giver": ["Alice", "Bob"],
                        "gift1": ["Old1", "Old2"],
                        "gift2": ["Old3", "Old4"],
                        "year": [current_year, current_year],
                    }
                ),
            ],
            ignore_index=True,
        )
        initial_db.to_csv(db_path, index=False)

        # New result for same year
        result = pd.DataFrame(
            {
                "giver": ["Alice", "Bob", "Charlie"],
                "gift1": ["Bob", "Charlie", "Alice"],
                "gift2": ["Charlie", "Alice", "Bob"],
                "gift1_ly": [None, None, None],
                "gift2_ly": [None, None, None],
            }
        )

        emails = {
            "Alice": "alice@example.com",
            "Bob": "bob@example.com",
            "Charlie": "charlie@example.com",
            "Diana": "diana@example.com",
        }
        generate_output(result, "template", emails, current_year, messages_path, db_path)

        # Read back the database
        saved_db = pd.read_csv(db_path)

        # Should have previous year + new current year (old current year should be replaced)
        assert len(saved_db) == len(sample_ly_gifts) + len(result)

        # Verify current year has new data (3 people, not 2)
        current_year_data = saved_db[saved_db["year"] == current_year]
        assert len(current_year_data) == 3
        assert set(current_year_data["giver"]) == {"Alice", "Bob", "Charlie"}

    def test_generate_output_markdown_content(self, temp_output_dir, sample_ly_gifts):
        """Test that markdown output contains correct message formatting."""
        result = pd.DataFrame(
            {
                "giver": ["Alice", "Bob"],
                "gift1": ["Bob", "Charlie"],
                "gift2": ["Charlie", "Diana"],
                "gift1_ly": [None, None],
                "gift2_ly": [None, None],
            }
        )

        message_template = "Hello {giver}, your assignments are {gift1} and {gift2}!"
        emails = {"Alice": "alice@test.com", "Bob": "bob@test.com"}
        current_year = 2025

        db_path = temp_output_dir / "secret_santa_db.csv"
        messages_path = temp_output_dir / "messages.md"

        # Create initial database
        sample_ly_gifts.to_csv(db_path, index=False)

        generate_output(result, message_template, emails, current_year, messages_path, db_path)

        # Read back the markdown and verify content
        with messages_path.open() as f:
            content = f.read()

        # Check that it contains expected structure
        assert "# Alice" in content
        assert "# Bob" in content
        assert "email: alice@test.com" in content
        assert "email: bob@test.com" in content
        assert "Hello Alice, your assignments are Bob and Charlie!" in content
        assert "Hello Bob, your assignments are Charlie and Diana!" in content

    def test_generate_output_message_template_formatting(self, temp_output_dir, sample_ly_gifts):
        """Test that message template placeholders are correctly replaced."""
        result = pd.DataFrame(
            {
                "giver": ["TestPerson"],
                "gift1": ["Person1"],
                "gift2": ["Person2"],
                "gift1_ly": [None],
                "gift2_ly": [None],
            }
        )

        message_template = "Greetings {giver}! Give gifts to {gift1} and {gift2}."
        emails = {"TestPerson": "test@example.com"}
        current_year = 2025

        db_path = temp_output_dir / "secret_santa_db.csv"
        messages_path = temp_output_dir / "messages.md"

        # Create initial database
        sample_ly_gifts.to_csv(db_path, index=False)

        generate_output(result, message_template, emails, current_year, messages_path, db_path)

        with messages_path.open() as f:
            content = f.read()

        # Verify exact message formatting
        expected_message = "Greetings TestPerson! Give gifts to Person1 and Person2."
        assert expected_message in content

    def test_generate_output_multiple_people(self, temp_output_dir, sample_config, sample_ly_gifts):
        """Test output generation with multiple people."""
        result = pd.DataFrame(
            {
                "giver": ["Alice", "Bob", "Charlie", "Diana"],
                "gift1": ["Bob", "Charlie", "Diana", "Alice"],
                "gift2": ["Charlie", "Diana", "Alice", "Bob"],
                "gift1_ly": [None, None, None, None],
                "gift2_ly": [None, None, None, None],
            }
        )

        current_year = sample_config["current_year"]
        db_path = temp_output_dir / "secret_santa_db.csv"
        messages_path = temp_output_dir / "messages.md"

        # Create initial database
        sample_ly_gifts.to_csv(db_path, index=False)

        emails = {
            "Alice": "alice@example.com",
            "Bob": "bob@example.com",
            "Charlie": "charlie@example.com",
            "Diana": "diana@example.com",
        }
        generate_output(result, "Hi {giver}!", emails, current_year, messages_path, db_path)

        with messages_path.open() as f:
            content = f.read()

        # Verify all people have sections
        for person in ["Alice", "Bob", "Charlie", "Diana"]:
            assert f"# {person}" in content
            assert f"email: {emails[person]}" in content

    def test_generate_output_missing_email(self, temp_output_dir, sample_ly_gifts):
        """Test that missing email addresses cause errors."""
        result = pd.DataFrame(
            {
                "giver": ["Alice"],
                "gift1": ["Bob"],
                "gift2": ["Charlie"],
                "gift1_ly": [None],
                "gift2_ly": [None],
            }
        )

        emails = {}  # Missing Alice's email
        current_year = 2025

        db_path = temp_output_dir / "secret_santa_db.csv"
        messages_path = temp_output_dir / "messages.md"

        # Create initial database
        sample_ly_gifts.to_csv(db_path, index=False)

        # Should raise KeyError for missing email
        with pytest.raises(KeyError):
            generate_output(result, "template", emails, current_year, messages_path, db_path)

    def test_generate_output_default_paths(self, temp_output_dir, sample_config, sample_ly_gifts, monkeypatch):
        """Test that default file paths work correctly."""
        # Change to temp directory so default paths go there
        monkeypatch.chdir(temp_output_dir)

        result = pd.DataFrame(
            {
                "giver": ["Alice"],
                "gift1": ["Bob"],
                "gift2": ["Charlie"],
                "gift1_ly": [None],
                "gift2_ly": [None],
            }
        )

        current_year = sample_config["current_year"]

        # Create the default data directory structure
        (temp_output_dir / "data").mkdir(parents=True)
        (temp_output_dir / "data" / "output").mkdir(parents=True)

        # Create initial database at default location
        db_path = temp_output_dir / "data" / "secret_santa_db.csv"
        sample_ly_gifts.to_csv(db_path, index=False)

        # Call without specifying paths (use defaults)
        emails = {
            "Alice": "alice@example.com",
            "Bob": "bob@example.com",
            "Charlie": "charlie@example.com",
            "Diana": "diana@example.com",
        }
        generate_output(result, "Hello {giver}!", emails, current_year)

        # Verify default files were created/updated
        assert db_path.exists()
        assert (temp_output_dir / "data" / "output" / "secret_santa_messages.md").exists()
