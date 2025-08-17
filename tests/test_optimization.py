import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from src.algorithm_logical_exchange_xmas.run import process_results, solve_optimization_problem


class TestSolveOptimizationProblem:
    def test_solve_optimization_problem_success(self):
        """Test successful optimization problem solving with minimal constraints."""
        people = ["Alice", "Bob", "Charlie", "Diana"]
        # Use empty last year's gifts to avoid conflicts
        ly_gifts = pd.DataFrame(columns=["giver", "gift1", "gift2"])
        couples = []  # No couple constraints for simplicity
        families = [["Alice", "Bob", "Charlie", "Diana"]]  # All in one family
        manual_disallows = {}
        gifts_per_person = 1  # Each person gives/receives 1 gift
        max_gifts_to_family = 4  # Allow gifts within family
        max_gifts_from_family = 4
        max_couple_overlap = 1
        seed = 123

        gifts = solve_optimization_problem(
            people,
            ly_gifts,
            couples,
            families,
            manual_disallows,
            gifts_per_person,
            max_gifts_to_family,
            max_gifts_from_family,
            max_couple_overlap,
            seed,
        )

        # Verify the result is a CVXPY variable with a solution
        assert isinstance(gifts, cp.Variable)
        assert gifts.value is not None
        assert gifts.value.shape == (4, 4)

        # Verify it's a boolean solution
        assert np.all((gifts.value == 0) | (gifts.value == 1))

    def test_solve_optimization_problem_deterministic(self):
        """Test that same seed produces same result."""
        people = ["Alice", "Bob", "Charlie", "Diana"]
        ly_gifts = pd.DataFrame(columns=["giver", "gift1", "gift2"])
        couples = []
        families = [["Alice", "Bob", "Charlie", "Diana"]]
        manual_disallows = {}
        gifts_per_person = 1
        max_gifts_to_family = 4
        max_gifts_from_family = 4
        max_couple_overlap = 1
        seed = 456

        # Run twice with same seed
        gifts1 = solve_optimization_problem(
            people,
            ly_gifts,
            couples,
            families,
            manual_disallows,
            gifts_per_person,
            max_gifts_to_family,
            max_gifts_from_family,
            max_couple_overlap,
            seed,
        )

        gifts2 = solve_optimization_problem(
            people,
            ly_gifts,
            couples,
            families,
            manual_disallows,
            gifts_per_person,
            max_gifts_to_family,
            max_gifts_from_family,
            max_couple_overlap,
            seed,
        )

        # Results should be identical
        np.testing.assert_array_equal(gifts1.value, gifts2.value)

    def test_solve_optimization_problem_different_seeds(self):
        """Test that different seeds produce different results."""
        people = ["Alice", "Bob", "Charlie", "Diana"]
        ly_gifts = pd.DataFrame(columns=["giver", "gift1", "gift2"])
        couples = []
        families = [["Alice", "Bob", "Charlie", "Diana"]]
        manual_disallows = {}
        gifts_per_person = 1
        max_gifts_to_family = 4
        max_gifts_from_family = 4
        max_couple_overlap = 1

        # Run with different seeds
        gifts1 = solve_optimization_problem(
            people,
            ly_gifts,
            couples,
            families,
            manual_disallows,
            gifts_per_person,
            max_gifts_to_family,
            max_gifts_from_family,
            max_couple_overlap,
            111,
        )

        gifts2 = solve_optimization_problem(
            people,
            ly_gifts,
            couples,
            families,
            manual_disallows,
            gifts_per_person,
            max_gifts_to_family,
            max_gifts_from_family,
            max_couple_overlap,
            222,
        )

        # Results should be different (with very high probability)
        assert not np.array_equal(gifts1.value, gifts2.value)

    def test_solve_optimization_problem_infeasible(self):
        """Test behavior with infeasible constraints."""
        people = ["Alice", "Bob"]  # Only 2 people
        ly_gifts = pd.DataFrame(columns=["giver", "gift1", "gift2"])
        couples = []
        families = [["Alice"], ["Bob"]]  # Each person in separate family
        manual_disallows = {"Alice": ["Bob"], "Bob": ["Alice"]}  # No one can give to anyone
        gifts_per_person = 1
        max_gifts_to_family = 0  # Can't give within family
        max_gifts_from_family = 0
        max_couple_overlap = 0
        seed = 123

        # This should be infeasible and raise an error
        with pytest.raises(ValueError, match="No optimal solution found"):
            solve_optimization_problem(
                people,
                ly_gifts,
                couples,
                families,
                manual_disallows,
                gifts_per_person,
                max_gifts_to_family,
                max_gifts_from_family,
                max_couple_overlap,
                seed,
            )


class TestProcessResults:
    def test_process_results_success(self):
        """Test successful result processing."""
        people = ["Alice", "Bob", "Charlie", "Diana"]
        gifts_per_person = 2

        # Use empty last year data to avoid conflicts
        ly_gifts = pd.DataFrame(columns=["giver", "gift1", "gift2"])

        # Create a valid solution matrix (Alice->Charlie,Diana; Bob->Alice,Charlie; etc.)
        solution_matrix = np.array(
            [
                [0, 0, 1, 1],  # Alice gives to Charlie, Diana
                [1, 0, 1, 0],  # Bob gives to Alice, Charlie
                [0, 1, 0, 1],  # Charlie gives to Bob, Diana
                [1, 1, 0, 0],  # Diana gives to Alice, Bob
            ]
        )

        # Create mock CVXPY variable with this solution
        gifts = cp.Variable((4, 4), boolean=True)
        gifts.value = solution_matrix

        result = process_results(gifts, people, ly_gifts, gifts_per_person)

        # Verify result structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4
        assert list(result.columns) == ["giver", "gift1", "gift2", "gift1_ly", "gift2_ly"]

        # Verify all people are present
        assert set(result["giver"]) == set(people)
        assert result["giver"].is_unique

        # Verify gift assignments make sense
        for _, row in result.iterrows():
            assert row["gift1"] in people
            assert row["gift2"] in people
            assert row["gift1"] != row["gift2"]  # Can't give to same person twice

    def test_process_results_validates_gift_counts(self):
        """Test that result validation catches incorrect gift counts."""
        people = ["Alice", "Bob", "Charlie", "Diana"]
        gifts_per_person = 2
        ly_gifts = pd.DataFrame(columns=["giver", "gift1", "gift2"])

        # Create invalid solution where Charlie gets 3 gifts instead of 2
        solution_matrix = np.array(
            [
                [0, 0, 1, 1],  # Alice gives to Charlie, Diana
                [1, 0, 1, 0],  # Bob gives to Alice, Charlie
                [0, 1, 0, 1],  # Charlie gives to Bob, Diana
                [1, 0, 1, 0],  # Diana gives to Alice, Charlie (Charlie gets 3 total: from Alice, Bob, Diana)
            ]
        )

        gifts = cp.Variable((4, 4), boolean=True)
        gifts.value = solution_matrix

        # Should raise assertion error about gift counts
        with pytest.raises(AssertionError, match="not every person appears exactly"):
            process_results(gifts, people, ly_gifts, gifts_per_person)

    def test_process_results_validates_no_repeats(self):
        """Test that result validation catches repeat gifts from last year."""
        people = ["Alice", "Bob", "Charlie", "Diana"]
        gifts_per_person = 2

        # Last year: Alice gave to Bob and Charlie
        ly_gifts = pd.DataFrame(
            {
                "giver": ["Alice"],
                "gift1": ["Bob"],
                "gift2": ["Charlie"],
            }
        )

        # This year: Alice gives to Bob again (should be prevented)
        solution_matrix = np.array(
            [
                [0, 1, 1, 0],  # Alice gives to Bob, Charlie (repeat!)
                [1, 0, 0, 1],  # Bob gives to Alice, Diana
                [0, 1, 0, 1],  # Charlie gives to Bob, Diana
                [1, 0, 1, 0],  # Diana gives to Alice, Charlie
            ]
        )

        gifts = cp.Variable((4, 4), boolean=True)
        gifts.value = solution_matrix

        # Should raise assertion error about repeat gifts
        with pytest.raises(AssertionError, match="repeat gift detected"):
            process_results(gifts, people, ly_gifts, gifts_per_person)

    def test_process_results_handles_new_participants(self):
        """Test processing results when some participants are new (not in last year's data)."""
        people = ["Alice", "Bob", "Charlie", "Diana"]
        gifts_per_person = 2

        # Last year only had Alice and Bob (with different recipients)
        ly_gifts = pd.DataFrame(
            {
                "giver": ["Alice", "Bob"],
                "gift1": ["Eve", "Frank"],  # People not participating this year
                "gift2": ["Grace", "Henry"],  # People not participating this year
            }
        )

        # Valid solution this year (no repeats since last year's recipients aren't participating)
        solution_matrix = np.array(
            [
                [0, 0, 1, 1],  # Alice gives to Charlie, Diana
                [1, 0, 1, 0],  # Bob gives to Alice, Charlie
                [0, 1, 0, 1],  # Charlie gives to Bob, Diana
                [1, 1, 0, 0],  # Diana gives to Alice, Bob
            ]
        )

        gifts = cp.Variable((4, 4), boolean=True)
        gifts.value = solution_matrix

        result = process_results(gifts, people, ly_gifts, gifts_per_person)

        # Should work fine, with Charlie and Diana having NaN for last year's data
        assert len(result) == 4
        assert pd.isna(result[result["giver"] == "Charlie"]["gift1_ly"].iloc[0])
        assert pd.isna(result[result["giver"] == "Diana"]["gift1_ly"].iloc[0])
