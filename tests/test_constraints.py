import cvxpy as cp
import pandas as pd

from src.algorithm_logical_exchange_xmas.run import (
    create_basic_constraints,
    create_couple_constraints,
    create_cycle_constraints,
    create_family_constraints,
    create_last_year_constraints,
    create_manual_assign_constraints,
    create_manual_disallow_constraints,
)


class TestBasicConstraints:
    def test_create_basic_constraints(self):
        """Test basic constraints creation."""
        people = ["Alice", "Bob", "Charlie"]
        gifts_per_person = 2
        gifts = cp.Variable((3, 3), boolean=True)

        constraints = create_basic_constraints(gifts, people, gifts_per_person)

        # Should have self-gifting constraints + row sum + column sum constraints
        assert len(constraints) == 3 + 2  # 3 self-constraints + 2 sum constraints

        # Verify constraint types
        self_constraints = constraints[:3]

        # Check that self-gifting constraints exist
        for i in range(3):
            # Each constraint should be gifts[i,i] == 0
            assert str(self_constraints[i]).count("== 0") == 1

    def test_create_basic_constraints_different_sizes(self):
        """Test basic constraints with different numbers of people."""
        for n_people in [2, 4, 5]:
            people = [f"Person{i}" for i in range(n_people)]
            gifts_per_person = 1
            gifts = cp.Variable((n_people, n_people), boolean=True)

            constraints = create_basic_constraints(gifts, people, gifts_per_person)

            # Should have n_people self-constraints + 2 sum constraints
            assert len(constraints) == n_people + 2


class TestLastYearConstraints:
    def test_create_last_year_constraints(self):
        """Test last year constraints creation."""
        people = ["Alice", "Bob", "Charlie", "Diana"]
        gifts = cp.Variable((4, 4), boolean=True)

        ly_gifts = pd.DataFrame(
            {
                "giver": ["Alice", "Bob"],
                "gift1": ["Bob", "Charlie"],
                "gift2": ["Charlie", "Diana"],
            }
        )

        constraints = create_last_year_constraints(gifts, people, ly_gifts)

        # Should create constraints to prevent Alice->Bob, Alice->Charlie, Bob->Charlie, Bob->Diana
        assert len(constraints) == 4

    def test_create_last_year_constraints_partial_participation(self):
        """Test last year constraints when some people didn't participate this year."""
        people = ["Alice", "Charlie"]  # Bob and Diana not participating this year
        gifts = cp.Variable((2, 2), boolean=True)

        ly_gifts = pd.DataFrame(
            {
                "giver": ["Alice", "Bob"],  # Bob not in this year's participants
                "gift1": ["Bob", "Charlie"],
                "gift2": ["Charlie", "Diana"],
            }
        )

        constraints = create_last_year_constraints(gifts, people, ly_gifts)

        # Should only create constraint for Alice->Charlie (Bob and Diana not participating)
        assert len(constraints) == 1

    def test_create_last_year_constraints_empty_history(self):
        """Test last year constraints with empty history."""
        people = ["Alice", "Bob", "Charlie"]
        gifts = cp.Variable((3, 3), boolean=True)

        ly_gifts = pd.DataFrame(columns=["giver", "gift1", "gift2"])

        constraints = create_last_year_constraints(gifts, people, ly_gifts)

        # Should create no constraints
        assert len(constraints) == 0


class TestCoupleConstraints:
    def test_create_couple_constraints(self):
        """Test couple constraints creation."""
        people = ["Alice", "Bob", "Charlie", "Diana"]
        gifts = cp.Variable((4, 4), boolean=True)
        couples = [["Alice", "Bob"], ["Charlie", "Diana"]]
        max_overlap = 1

        constraints = create_couple_constraints(gifts, people, couples, max_overlap)

        # Each couple generates 3 constraints: 2 no-gifting + 1 overlap constraint
        # 2 couples * 3 constraints = 6 total
        assert len(constraints) == 6

    def test_create_couple_constraints_partial_participation(self):
        """Test couple constraints when only one person from couple participates."""
        people = ["Alice", "Charlie"]  # Bob and Diana not participating
        gifts = cp.Variable((2, 2), boolean=True)
        couples = [["Alice", "Bob"], ["Charlie", "Diana"]]
        max_overlap = 1

        constraints = create_couple_constraints(gifts, people, couples, max_overlap)

        # No constraints should be created since no complete couples are participating
        assert len(constraints) == 0

    def test_create_couple_constraints_no_couples(self):
        """Test couple constraints with no couples."""
        people = ["Alice", "Bob", "Charlie"]
        gifts = cp.Variable((3, 3), boolean=True)
        couples = []
        max_overlap = 1

        constraints = create_couple_constraints(gifts, people, couples, max_overlap)

        assert len(constraints) == 0


class TestFamilyConstraints:
    def test_create_family_constraints(self):
        """Test family constraints creation."""
        people = ["Alice", "Bob", "Charlie", "Diana"]
        gifts = cp.Variable((4, 4), boolean=True)
        families = [["Alice", "Bob"], ["Charlie", "Diana"]]
        max_to_family = 1
        max_from_family = 1

        constraints = create_family_constraints(gifts, people, families, max_to_family, max_from_family)

        # Each person generates 2 constraints (to/from family limits)
        # 4 people * 2 constraints = 8 total
        assert len(constraints) == 8

    def test_create_family_constraints_partial_participation(self):
        """Test family constraints with partial family participation."""
        people = ["Alice", "Charlie"]  # Bob and Diana not participating
        gifts = cp.Variable((2, 2), boolean=True)
        families = [["Alice", "Bob"], ["Charlie", "Diana"]]
        max_to_family = 1
        max_from_family = 1

        constraints = create_family_constraints(gifts, people, families, max_to_family, max_from_family)

        # Only Alice and Charlie are participating, so 2 * 2 = 4 constraints
        assert len(constraints) == 4

    def test_create_family_constraints_empty_families(self):
        """Test family constraints with no families."""
        people = ["Alice", "Bob"]
        gifts = cp.Variable((2, 2), boolean=True)
        families = []
        max_to_family = 1
        max_from_family = 1

        constraints = create_family_constraints(gifts, people, families, max_to_family, max_from_family)

        assert len(constraints) == 0


class TestCycleConstraints:
    def test_create_cycle_constraints(self):
        """Test cycle constraints creation."""
        people = ["Alice", "Bob", "Charlie"]
        gifts = cp.Variable((3, 3), boolean=True)

        constraints = create_cycle_constraints(gifts, people)

        # For n people, should create n² constraints (one for each pair including self)
        assert len(constraints) == 9

    def test_create_cycle_constraints_different_sizes(self):
        """Test cycle constraints with different numbers of people."""
        for n_people in [2, 4, 5]:
            people = [f"Person{i}" for i in range(n_people)]
            gifts = cp.Variable((n_people, n_people), boolean=True)

            constraints = create_cycle_constraints(gifts, people)

            # Should have n² constraints
            assert len(constraints) == n_people**2


class TestManualDisallowConstraints:
    def test_create_manual_disallow_constraints(self):
        """Test manual disallow constraints creation."""
        people = ["Alice", "Bob", "Charlie", "Diana"]
        gifts = cp.Variable((4, 4), boolean=True)
        manual_disallows = {
            "Alice": ["Bob", "Charlie"],
            "Diana": ["Alice"],
        }

        constraints = create_manual_disallow_constraints(gifts, people, manual_disallows)

        # Alice can't give to Bob, Charlie (2 constraints)
        # Diana can't give to Alice (1 constraint)
        # Total: 3 constraints
        assert len(constraints) == 3

    def test_create_manual_disallow_constraints_non_participant(self):
        """Test manual disallow constraints when disallowed person doesn't participate."""
        people = ["Alice", "Bob", "Charlie"]  # Diana not participating
        gifts = cp.Variable((3, 3), boolean=True)
        manual_disallows = {
            "Alice": ["Bob", "Diana"],  # Diana not participating
            "Eve": ["Alice"],  # Eve not participating
        }

        constraints = create_manual_disallow_constraints(gifts, people, manual_disallows)

        # Only Alice->Bob constraint should be created
        assert len(constraints) == 1

    def test_create_manual_disallow_constraints_empty(self):
        """Test manual disallow constraints with no disallows."""
        people = ["Alice", "Bob", "Charlie"]
        gifts = cp.Variable((3, 3), boolean=True)
        manual_disallows = {}

        constraints = create_manual_disallow_constraints(gifts, people, manual_disallows)

        assert len(constraints) == 0

    def test_create_manual_disallow_constraints_self_disallow(self):
        """Test manual disallow constraints including self-disallow (should be redundant)."""
        people = ["Alice", "Bob"]
        gifts = cp.Variable((2, 2), boolean=True)
        manual_disallows = {
            "Alice": ["Alice", "Bob"],  # Self-disallow should be redundant with basic constraints
        }

        constraints = create_manual_disallow_constraints(gifts, people, manual_disallows)

        # Should create 2 constraints (Alice->Alice and Alice->Bob)
        assert len(constraints) == 2


class TestManualAssignConstraints:
    def test_create_manual_assign_constraints(self):
        """Test manual assign constraints creation."""
        people = ["Alice", "Bob", "Charlie", "Diana"]
        gifts = cp.Variable((4, 4), boolean=True)
        manual_assigns = {
            "Alice": ["Charlie"],
            "Bob": ["Diana"],
        }

        constraints = create_manual_assign_constraints(gifts, people, manual_assigns)

        # Alice must give to Charlie (1 constraint)
        # Bob must give to Diana (1 constraint)
        # Total: 2 constraints
        assert len(constraints) == 2

    def test_create_manual_assign_constraints_non_participant(self):
        """Test manual assign constraints when assigned person doesn't participate."""
        people = ["Alice", "Bob", "Charlie"]  # Diana not participating
        gifts = cp.Variable((3, 3), boolean=True)
        manual_assigns = {
            "Alice": ["Charlie", "Diana"],  # Diana not participating
            "Eve": ["Bob"],  # Eve not participating
        }

        constraints = create_manual_assign_constraints(gifts, people, manual_assigns)

        # Only Alice->Charlie constraint should be created
        assert len(constraints) == 1

    def test_create_manual_assign_constraints_empty(self):
        """Test manual assign constraints with no assigns."""
        people = ["Alice", "Bob", "Charlie"]
        gifts = cp.Variable((3, 3), boolean=True)
        manual_assigns = {}

        constraints = create_manual_assign_constraints(gifts, people, manual_assigns)

        assert len(constraints) == 0

    def test_create_manual_assign_constraints_multiple_per_person(self):
        """Test manual assign constraints with multiple assigns for one person."""
        people = ["Alice", "Bob", "Charlie", "Diana"]
        gifts = cp.Variable((4, 4), boolean=True)
        manual_assigns = {
            "Alice": ["Bob", "Charlie"],  # Alice must give to both Bob and Charlie
        }

        constraints = create_manual_assign_constraints(gifts, people, manual_assigns)

        # Should create 2 constraints (Alice->Bob and Alice->Charlie)
        assert len(constraints) == 2
