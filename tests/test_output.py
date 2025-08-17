import pandas as pd
import pytest

from src.algorithm_logical_exchange_xmas.run import generate_output


class TestGenerateOutput:
    def test_generate_output_creates_files(self, temp_output_dir, sample_config):
        """Test that output generation creates both CSV and markdown files."""
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
        emails = sample_config["emails"]

        assignments_path = temp_output_dir / "assignments.csv"
        messages_path = temp_output_dir / "messages.md"

        # Call generate_output
        generate_output(result, message_template, emails, assignments_path, messages_path)

        # Verify files were created
        assert assignments_path.exists()
        assert messages_path.exists()

    def test_generate_output_csv_content(self, temp_output_dir, sample_config):
        """Test that CSV output contains correct data."""
        result = pd.DataFrame(
            {
                "giver": ["Alice", "Bob"],
                "gift1": ["Bob", "Alice"],
                "gift2": ["Charlie", "Diana"],
                "gift1_ly": ["Old1", "Old2"],
                "gift2_ly": ["Old3", "Old4"],
            }
        )

        assignments_path = temp_output_dir / "assignments.csv"
        messages_path = temp_output_dir / "messages.md"

        generate_output(result, "template", sample_config["emails"], assignments_path, messages_path)

        # Read back the CSV and verify content
        saved_result = pd.read_csv(assignments_path)
        pd.testing.assert_frame_equal(result, saved_result)

    def test_generate_output_markdown_content(self, temp_output_dir):
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

        assignments_path = temp_output_dir / "assignments.csv"
        messages_path = temp_output_dir / "messages.md"

        generate_output(result, message_template, emails, assignments_path, messages_path)

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

    def test_generate_output_message_template_formatting(self, temp_output_dir):
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

        assignments_path = temp_output_dir / "assignments.csv"
        messages_path = temp_output_dir / "messages.md"

        generate_output(result, message_template, emails, assignments_path, messages_path)

        with messages_path.open() as f:
            content = f.read()

        # Verify exact message formatting
        expected_message = "Greetings TestPerson! Give gifts to Person1 and Person2."
        assert expected_message in content

    def test_generate_output_multiple_people(self, temp_output_dir, sample_config):
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

        assignments_path = temp_output_dir / "assignments.csv"
        messages_path = temp_output_dir / "messages.md"

        generate_output(result, "Hi {giver}!", sample_config["emails"], assignments_path, messages_path)

        with messages_path.open() as f:
            content = f.read()

        # Verify all people have sections
        for person in ["Alice", "Bob", "Charlie", "Diana"]:
            assert f"# {person}" in content
            assert f"email: {sample_config['emails'][person]}" in content

    def test_generate_output_missing_email(self, temp_output_dir):
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

        assignments_path = temp_output_dir / "assignments.csv"
        messages_path = temp_output_dir / "messages.md"

        # Should raise KeyError for missing email
        with pytest.raises(KeyError):
            generate_output(result, "template", emails, assignments_path, messages_path)

    def test_generate_output_default_paths(self, temp_output_dir, sample_config, monkeypatch):
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

        # Create the default data/output directory structure
        (temp_output_dir / "data" / "output").mkdir(parents=True)

        # Call without specifying paths (use defaults)
        generate_output(result, "Hello {giver}!", sample_config["emails"])

        # Verify default files were created
        assert (temp_output_dir / "data" / "output" / "assignments.csv").exists()
        assert (temp_output_dir / "data" / "output" / "secret_santa_messages.md").exists()
