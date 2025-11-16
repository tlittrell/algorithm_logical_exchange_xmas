"""Secret Santa gift assignment optimization algorithm.

This module implements a constraint-based integer programming solution for assigning
Secret Santa gifts. It uses CVXPY to solve an optimization problem that maximizes
assignment novelty while respecting multiple constraints:

- No self-gifting or repeat gifts from last year
- Equal distribution (everyone gives/receives the same number of gifts)
- Couple constraints (partners can't gift each other or the same people)
- Family limits (configurable within-family gifting restrictions)
- Cycle prevention (if A→B, then B cannot→A)
- Manual disallow lists for custom restrictions

The algorithm loads configuration from a TOML file, reads historical and current
participant data, builds and solves the optimization problem, then generates
CSV assignments and personalized Markdown messages.

Example:
    Run the algorithm from the command line::

        $ uv run python src/algorithm_logical_exchange_xmas/run.py

    Or as a module::

        $ uv run python -m algorithm_logical_exchange_xmas.run
"""

import itertools
import logging
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any

import cvxpy as cp
import duckdb
import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_config(config_path: Path = Path("local_config.toml")) -> dict[str, Any]:
    """Load configuration from TOML file.

    Reads and parses a TOML configuration file containing algorithm parameters,
    eligible participants, relationship constraints, and email addresses.

    Args:
        config_path: Path to the TOML configuration file. Defaults to
            "local_config.toml" in the current working directory.

    Returns:
        A dictionary containing the parsed configuration with keys:
            - "seed": Random seed for reproducibility
            - "algorithm": Algorithm configuration (people, couples, families, etc.)
            - "emails": Mapping of participant names to email addresses
            - "manual_disallows": Manual gift assignment restrictions

    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        tomllib.TOMLDecodeError: If the TOML file is malformed.
    """
    logging.info("Reading in config")
    with config_path.open("rb") as file:
        return tomllib.load(file)


def validate_config(config: dict[str, Any]) -> None:
    """Validate configuration data for consistency and completeness.

    Performs comprehensive validation checks on the configuration to ensure:
    - No duplicate entries in eligible people, couples, or families
    - All people in couples/families are in the eligible people list
    - Every eligible person is assigned to exactly one family
    - Seed is a non-negative integer

    Args:
        config: Configuration dictionary loaded from TOML file, containing
            "algorithm" (with eligible_people, couples, families) and "seed" keys.

    Returns:
        None. Validation is performed via assertions.

    Raises:
        AssertionError: If any validation check fails, with a descriptive message
            indicating which constraint was violated.
    """
    logging.info("Validating config")

    algorithm_config = config["algorithm"]
    eligible_people = algorithm_config["eligible_people"]
    couples = algorithm_config["couples"]
    families = algorithm_config["families"]
    seed = config["seed"]

    # Validate eligible people
    assert len(set(eligible_people)) == len(eligible_people), "eligible people contains duplicates"

    # Validate couples
    people_in_couples = list(itertools.chain(*couples))
    assert len(set(people_in_couples)) == len(people_in_couples), "Couples contains duplicates"
    assert set(people_in_couples).issubset(set(eligible_people))

    # Validate families
    people_in_families = list(itertools.chain(*families))
    assert len(set(people_in_families)) == len(people_in_families), "Families contains duplicates"
    assert set(people_in_families) == set(eligible_people), "Not everyone assigned a family"

    # Validate seed
    assert seed >= 0
    assert isinstance(seed, int)


def load_data(
    eligible_people: list[str],
    current_year: int,
    signups_db_path: Path = Path("data/signups.csv"),
    db_path: Path = Path("data/secret_santa_db.csv"),
) -> tuple[pd.DataFrame, list[str]]:
    """Load last year's gifts and this year's signups from databases.

    Queries the Secret Santa database for previous year's gift assignments and
    signups database for current year participants, validating that all participants
    are eligible and data integrity is maintained.

    Args:
        eligible_people: List of all people eligible to participate in Secret Santa.
        current_year: The year for which assignments are being generated. Previous
            year's data (current_year - 1) will be loaded for constraint creation.
        signups_db_path: Path to the signups database CSV file containing historical
            signup data. Must have columns: "person", "is_secret_santa", "is_stockings",
            "year". Defaults to "data/signups.csv".
        db_path: Path to the Secret Santa database CSV file containing historical
            assignments. Must have columns: "giver", "gift1", "gift2", "year".
            Defaults to "data/secret_santa_db.csv".

    Returns:
        A tuple containing:
            - DataFrame with last year's gift assignments (columns: giver, gift1, gift2)
            - List of people who signed up to participate this year

    Raises:
        AssertionError: If data validation fails (e.g., duplicate givers, ineligible
            participants, or duplicate signups).
        FileNotFoundError: If database files don't exist.
        KeyError: If required columns are missing from the files.
        duckdb.Error: If database query fails.
    """
    logging.info(f"Reading last year's gifts from database (year {current_year - 1})")
    ly_gifts = duckdb.sql(
        """
        SELECT giver, gift1, gift2
        FROM read_csv_auto(?)
        WHERE year = ?
        """,
        params=[str(db_path), current_year - 1],
    ).df()
    assert ly_gifts["giver"].is_unique
    assert set(ly_gifts["giver"]).issubset(eligible_people), "Ineligible LY gifter"

    logging.info(f"Reading this year's signups from database (year {current_year})")
    signups = duckdb.sql(
        """
        SELECT person
        FROM read_csv_auto(?)
        WHERE year = ? AND is_secret_santa = TRUE
        """,
        params=[str(signups_db_path), current_year],
    ).df()
    assert set(signups["person"]).issubset(eligible_people), "People signed up aren't eligible"
    assert signups["person"].is_unique

    # Get the people signed up this year
    people_signed_up = signups["person"].to_list()

    return ly_gifts, people_signed_up


def create_basic_constraints(
    gifts: cp.Variable, people_signed_up: list[str], gifts_per_person: int
) -> list[cp.Constraint]:
    """Create basic constraints: no self-gifting and gift count constraints.

    Establishes fundamental rules that everyone gives and receives the same number
    of gifts, and no one can gift themselves.

    Args:
        gifts: CVXPY boolean variable matrix (nxn) where gifts[i,j]=1 means
            person i gives to person j.
        people_signed_up: List of participant names (ordered, length n).
        gifts_per_person: Number of gifts each person should give and receive.

    Returns:
        List of CVXPY constraints enforcing:
            - No self-gifting: gifts[i,i] = 0 for all i
            - Equal giving: Each person gives exactly gifts_per_person gifts
            - Equal receiving: Each person receives exactly gifts_per_person gifts
    """
    constraints = []
    n_people = len(people_signed_up)

    # Can't give to yourself
    constraints.extend([gifts[i, i] == 0 for i in range(n_people)])

    # Each person gives and receives the configured number of gifts
    constraints.append(cp.sum(gifts, axis=0) == np.full(n_people, gifts_per_person))
    constraints.append(cp.sum(gifts, axis=1) == np.full(n_people, gifts_per_person))

    return constraints


def create_last_year_constraints(
    gifts: cp.Variable, people_signed_up: list[str], ly_gifts: pd.DataFrame
) -> list[cp.Constraint]:
    """Create constraints to prevent repeating last year's assignments.

    Ensures no participant receives gifts from the same person two years in a row,
    maintaining novelty and preventing patterns.

    Args:
        gifts: CVXPY boolean variable matrix (nxn) where gifts[i,j]=1 means
            person i gives to person j.
        people_signed_up: List of participant names currently signed up this year.
        ly_gifts: DataFrame with last year's assignments, containing columns:
            "giver", "gift1", "gift2".

    Returns:
        List of CVXPY constraints where gifts[i,j]=0 if person i gave to person j
        last year (applies only to participants signed up both years).
    """
    constraints = []

    # No repeats of last year
    for row in ly_gifts.iterrows():
        gifter = row[1]["giver"]
        receiver1 = row[1]["gift1"]
        receiver2 = row[1]["gift2"]

        giver_idx = people_signed_up.index(gifter) if gifter in people_signed_up else None
        gift1_idx = people_signed_up.index(receiver1) if receiver1 in people_signed_up else None
        gift2_idx = people_signed_up.index(receiver2) if receiver2 in people_signed_up else None

        if giver_idx is not None and gift1_idx is not None:
            constraints.append(gifts[giver_idx, gift1_idx] == 0)
        if giver_idx is not None and gift2_idx is not None:
            constraints.append(gifts[giver_idx, gift2_idx] == 0)

    return constraints


def create_couple_constraints(
    gifts: cp.Variable, people_signed_up: list[str], couples: list[list[str]], max_couple_overlap: int
) -> list[cp.Constraint]:
    """Create constraints for couples.

    Prevents partners from gifting each other and limits how many people both
    partners can gift, ensuring variety in couple gift assignments.

    Args:
        gifts: CVXPY boolean variable matrix (nxn) where gifts[i,j]=1 means
            person i gives to person j.
        people_signed_up: List of participant names currently signed up.
        couples: List of two-element lists, each containing partner names.
        max_couple_overlap: Maximum number of people both partners can gift
            (typically 0 or 1 to ensure variety).

    Returns:
        List of CVXPY constraints enforcing:
            - Partners cannot gift each other (gifts[i,j]=0 and gifts[j,i]=0)
            - Limited overlap in gift recipients between partners
    """
    constraints = []
    n_people = len(people_signed_up)

    # Constraints for couples
    for person1, person2 in couples:
        if person1 in people_signed_up and person2 in people_signed_up:
            # Couples can't give to each other
            idx1 = people_signed_up.index(person1)
            idx2 = people_signed_up.index(person2)
            constraints.append(gifts[idx1, idx2] == 0)
            constraints.append(gifts[idx2, idx1] == 0)

            # Couples can't give to the same people because that's not fun
            constraints.append(cp.sum(gifts[[idx1, idx2], :], axis=0) <= np.full(n_people, max_couple_overlap))

    return constraints


def create_family_constraints(
    gifts: cp.Variable,
    people_signed_up: list[str],
    families: list[list[str]],
    max_gifts_to_family: int,
    max_gifts_from_family: int,
) -> list[cp.Constraint]:
    """Create constraints to limit gifts within families.

    Prevents excessive within-family gifting to encourage cross-family exchanges
    and maintain variety in gift assignments.

    Args:
        gifts: CVXPY boolean variable matrix (nxn) where gifts[i,j]=1 means
            person i gives to person j.
        people_signed_up: List of participant names currently signed up.
        families: List of lists, each containing names of family members.
        max_gifts_to_family: Maximum number of gifts a person can give to their
            own family members.
        max_gifts_from_family: Maximum number of gifts a person can receive from
            their own family members.

    Returns:
        List of CVXPY constraints limiting within-family gift exchanges for
        each participant based on the specified maximums.
    """
    constraints = []

    # Limit gifts within families
    for family in families:
        family_idx = [people_signed_up.index(person) for person in family if person in people_signed_up]
        for person in set(family).intersection(set(people_signed_up)):
            idx = people_signed_up.index(person)
            constraints.append(cp.sum(gifts[idx, family_idx]) <= max_gifts_to_family)
            constraints.append(cp.sum(gifts[family_idx, idx]) <= max_gifts_from_family)

    return constraints


def create_cycle_constraints(gifts: cp.Variable, people_signed_up: list[str]) -> list[cp.Constraint]:
    """Create constraints to prevent cycles (A->B and B->A).

    Prevents reciprocal gifting where person A gifts to person B and person B
    gifts back to person A, ensuring more interesting gift patterns.

    Args:
        gifts: CVXPY boolean variable matrix (nxn) where gifts[i,j]=1 means
            person i gives to person j.
        people_signed_up: List of participant names currently signed up.

    Returns:
        List of CVXPY constraints enforcing that for any pair (i,j),
        at most one direction of gifting can occur: gifts[i,j] + gifts[j,i] <= 1.
    """
    constraints = []

    # No cycles e.g. if person 1 gives to person 2 then person 2 can't give to person 1
    for person1, person2 in itertools.product(people_signed_up, people_signed_up):
        idx1 = people_signed_up.index(person1)
        idx2 = people_signed_up.index(person2)
        constraints.append(gifts[idx1, idx2] + gifts[idx2, idx1] <= 1)

    return constraints


def create_manual_disallow_constraints(
    gifts: cp.Variable, people_signed_up: list[str], manual_disallows: dict[str, list[str]]
) -> list[cp.Constraint]:
    """Create constraints for manual disallows.

    Applies custom restrictions specified in the configuration to prevent
    specific gift assignments based on user-defined rules.

    Args:
        gifts: CVXPY boolean variable matrix (nxn) where gifts[i,j]=1 means
            person i gives to person j.
        people_signed_up: List of participant names currently signed up.
        manual_disallows: Dictionary mapping giver names to lists of people
            they cannot gift to (e.g., {"Alice": ["Bob", "Charlie"]}).

    Returns:
        List of CVXPY constraints where gifts[i,j]=0 for all manually disallowed
        (i,j) pairs (only applied if both participants are signed up).
    """
    constraints = []

    # Add any manual blocks (e.g. new person doesn't get other outlaws)
    for person, disallow_list in manual_disallows.items():
        if person not in people_signed_up:
            continue

        person_idx = people_signed_up.index(person)
        for p2 in [p for p in disallow_list if p in people_signed_up]:
            p2_idx = people_signed_up.index(p2)
            constraints.append(gifts[person_idx, p2_idx] == 0)

    return constraints


def solve_optimization_problem(  # noqa: PLR0913
    people_signed_up: list[str],
    ly_gifts: pd.DataFrame,
    couples: list[list[str]],
    families: list[list[str]],
    manual_disallows: dict[str, list[str]],
    gifts_per_person: int,
    max_gifts_to_family: int,
    max_gifts_from_family: int,
    max_couple_overlap: int,
    seed: int,
) -> cp.Variable:
    """Set up and solve the optimization problem.

    Constructs and solves an integer programming problem that finds an optimal
    Secret Santa gift assignment matrix. The objective maximizes a random novelty
    matrix subject to all constraints, ensuring fair, diverse, and interesting
    gift assignments.

    Args:
        people_signed_up: List of participant names (defines matrix dimensions).
        ly_gifts: DataFrame with last year's assignments (columns: giver, gift1, gift2).
        couples: List of two-element lists containing partner names.
        families: List of lists, each containing family member names.
        manual_disallows: Dictionary mapping giver names to lists of disallowed recipients.
        gifts_per_person: Number of gifts each person gives and receives.
        max_gifts_to_family: Maximum gifts a person can give within their family.
        max_gifts_from_family: Maximum gifts a person can receive from their family.
        max_couple_overlap: Maximum people both partners in a couple can gift.
        seed: Random seed for reproducible novelty matrix generation.

    Returns:
        CVXPY Variable containing the optimal gift assignment matrix (nxn boolean),
        where gifts.value[i,j]=1 means person i gives to person j.

    Raises:
        ValueError: If no optimal solution exists (constraints are infeasible).

    Example:
        >>> gifts = solve_optimization_problem(
        ...     people_signed_up=["Alice", "Bob", "Charlie"],
        ...     ly_gifts=ly_df,
        ...     couples=[["Alice", "Bob"]],
        ...     families=[["Alice", "Bob"], ["Charlie"]],
        ...     manual_disallows={},
        ...     gifts_per_person=1,
        ...     max_gifts_to_family=0,
        ...     max_gifts_from_family=0,
        ...     max_couple_overlap=0,
        ...     seed=42
        ... )
        >>> print(gifts.value)  # Optimal assignment matrix
    """
    n_people = len(people_signed_up)

    # Set up random generator with seed
    rng = np.random.default_rng(seed)
    # Set up novelty matrix
    novelty = rng.random((n_people, n_people))

    ### Decision variables. Row is person giving, column is person receiving
    gifts = cp.Variable((n_people, n_people), boolean=True)

    ### Objective. Maximize novelty
    objective = cp.Maximize(cp.sum(cp.multiply(gifts, novelty)))

    ### Constraints
    constraints = []
    constraints.extend(create_basic_constraints(gifts, people_signed_up, gifts_per_person))
    constraints.extend(create_last_year_constraints(gifts, people_signed_up, ly_gifts))
    constraints.extend(create_couple_constraints(gifts, people_signed_up, couples, max_couple_overlap))
    constraints.extend(
        create_family_constraints(gifts, people_signed_up, families, max_gifts_to_family, max_gifts_from_family)
    )
    constraints.extend(create_cycle_constraints(gifts, people_signed_up))
    constraints.extend(create_manual_disallow_constraints(gifts, people_signed_up, manual_disallows))

    ### Create the integer programming problem
    problem = cp.Problem(objective, constraints)

    # Solve the problem
    problem.solve()

    # Display the results
    if problem.status == cp.OPTIMAL:
        logging.info("Optimal solution found")
        logging.debug("Optimal matrix X: %s", gifts.value)
        logging.info("Optimal objective value = %s", problem.value)
    else:
        error_msg = "No optimal solution found"
        raise ValueError(error_msg)

    return gifts


def process_results(
    gifts: cp.Variable, people_signed_up: list[str], ly_gifts: pd.DataFrame, gifts_per_person: int
) -> pd.DataFrame:
    """Process optimization results into a DataFrame.

    Converts the optimal gift assignment matrix into a human-readable DataFrame
    with giver-receiver pairs, and validates that all constraints are satisfied.

    Args:
        gifts: CVXPY Variable with solved optimal assignment matrix (nxn boolean),
            where gifts.value[i,j]=1 means person i gives to person j.
        people_signed_up: List of participant names (ordered, same as used in optimization).
        ly_gifts: DataFrame with last year's assignments for validation
            (columns: giver, gift1, gift2).
        gifts_per_person: Expected number of gifts each person gives/receives (for validation).

    Returns:
        DataFrame with columns:
            - giver: Person giving gifts
            - gift1: First recipient
            - gift2: Second recipient
            - gift1_ly: First recipient last year (for comparison)
            - gift2_ly: Second recipient last year (for comparison)

    Raises:
        AssertionError: If validation fails (e.g., unequal distribution, repeat
            gifts from last year, missing participants).
    """
    giver = []
    gift1 = []
    gift2 = []
    for i, person in enumerate(people_signed_up):
        receivers = [people_signed_up[j] for j in np.where(gifts.value[i] == 1)[0]]
        giver.append(person)
        gift1.append(receivers[0])
        gift2.append(receivers[1])

    result = pd.DataFrame({"giver": giver, "gift1": gift1, "gift2": gift2}).merge(
        ly_gifts.rename(columns={"gift1": "gift1_ly", "gift2": "gift2_ly", "person": "giver"}),
        on="giver",
        how="left",
        validate="1:1",
    )

    # Validate results
    assert len(result) == len(people_signed_up)
    assert result[["giver", "gift1", "gift2"]].notna().any().all()
    assert result["giver"].is_unique
    all_assignment_list = list(itertools.chain(*[result["gift1"].to_list(), result["gift2"].to_list()]))
    assert all(
        i == gifts_per_person for i in Counter(all_assignment_list).values()
    ), f"not every person appears exactly {gifts_per_person} times in assignments"
    assert set(all_assignment_list) == set(people_signed_up), "not everyone signed up gets gifts"
    assert (
        duckdb.sql(
            """
            select
                gift1 not ilike gift1_ly as test1,
                gift1 not ilike gift2_ly as test2,
                gift2 not ilike gift1_ly as test3,
                gift2 not ilike gift2_ly as test4,
            from result
            """
        )
        .df()
        .all()
        .all()
    ), "repeat gift detected"

    return result


def generate_output(
    result: pd.DataFrame,
    message_template: str,
    emails: dict[str, str],
    current_year: int,
    messages_path: Path = Path("data/output/secret_santa_messages.md"),
    db_path: Path = Path("data/secret_santa_db.csv"),
) -> None:
    """Generate output files and update the Secret Santa database.

    Updates the Secret Santa database with this year's gift assignments (replacing
    any existing entries for the current year) and creates a Markdown file with
    personalized messages for each participant including their email addresses.

    Args:
        result: DataFrame containing gift assignments with columns:
            giver, gift1, gift2 (and optionally gift1_ly, gift2_ly).
        message_template: String template for personalized messages with placeholders
            {giver}, {gift1}, {gift2} that will be filled for each participant.
        emails: Dictionary mapping participant names to email addresses.
        current_year: The year for which assignments were generated. Used to update
            the database and allows re-running the algorithm for the same year.
        messages_path: Path where the Markdown messages will be saved.
            Defaults to "data/output/secret_santa_messages.md".
        db_path: Path to the Secret Santa database CSV file where assignments will
            be stored. Defaults to "data/secret_santa_db.csv".

    Returns:
        None. Database is updated and message file is written to disk.

    Raises:
        IOError: If output files cannot be written (e.g., permission denied,
            directory doesn't exist).
        KeyError: If a participant's email is missing from the emails dictionary.
        duckdb.Error: If database update fails.
    """
    logging.info(f"Updating database with results for year {current_year}")

    # Read existing database
    existing_db = duckdb.sql(f"SELECT * FROM read_csv_auto('{db_path}')").df()

    # Remove existing entries for current year (allows re-running)
    updated_db = existing_db[existing_db["year"] != current_year]

    # Prepare new assignments with year column
    new_assignments = result[["giver", "gift1", "gift2"]].copy()
    new_assignments["year"] = current_year

    # Append new assignments
    updated_db = pd.concat([updated_db, new_assignments], ignore_index=True)

    # Write back to database
    updated_db.to_csv(db_path, index=False)

    logging.info("Writing messages")
    markdown_output_list = []
    for row_data in result.itertuples():
        message = message_template.format(giver=row_data.giver, gift1=row_data.gift1, gift2=row_data.gift2)

        # Create a markdown section with a heading for the giver
        giver_name = str(row_data.giver)
        markdown_message = f"# {giver_name}\n\nemail: {emails[giver_name]}\n\n{message}\n"

        # Append the message to the markdown output list
        markdown_output_list.append(markdown_message)

    markdown_output = "\n".join(markdown_output_list)
    with messages_path.open("w") as file:
        file.write(markdown_output)
    logging.info("Done")


def main() -> None:
    """Main function to orchestrate the Secret Santa algorithm.

    Executes the complete Secret Santa gift assignment workflow:
    1. Loads and validates configuration from TOML file
    2. Queries database for historical data and loads current participant signups
    3. Solves the constraint optimization problem
    4. Processes and validates results
    5. Updates database and generates personalized messages

    This function serves as the entry point when running the module directly.

    Returns:
        None. Database is updated and message file is created in data/output/.

    Raises:
        FileNotFoundError: If configuration or input data files are missing.
        ValueError: If the optimization problem has no feasible solution.
        AssertionError: If configuration or results fail validation checks.
        duckdb.Error: If database operations fail.

    Example:
        Run from command line::

            $ uv run python src/algorithm_logical_exchange_xmas/run.py

        Or as a module::

            $ uv run python -m algorithm_logical_exchange_xmas.run
    """
    # Load and validate configuration
    config = load_config()
    validate_config(config)

    # Extract config values
    seed = config["seed"]
    current_year = config["current_year"]
    emails = config["emails"]
    manual_disallows = config["manual_disallows"]

    # Algorithm configuration
    algorithm_config = config["algorithm"]
    couples = algorithm_config["couples"]
    families = algorithm_config["families"]
    eligible_people = algorithm_config["eligible_people"]
    message_template = algorithm_config["message"]
    gifts_per_person = algorithm_config["gifts_per_person"]
    max_gifts_to_family = algorithm_config["max_gifts_to_family"]
    max_gifts_from_family = algorithm_config["max_gifts_from_family"]
    max_couple_overlap = algorithm_config["max_couple_overlap"]

    # Load data
    ly_gifts, people_signed_up = load_data(
        eligible_people,
        current_year,
        signups_db_path=Path("data/signups.csv"),
    )

    # Solve optimization problem
    gifts = solve_optimization_problem(
        people_signed_up,
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

    # Process results
    result = process_results(gifts, people_signed_up, ly_gifts, gifts_per_person)

    # Generate output
    generate_output(
        result,
        message_template,
        emails,
        current_year,
        messages_path=Path("data/output/secret_santa_messages.md"),
    )


if __name__ == "__main__":
    main()
