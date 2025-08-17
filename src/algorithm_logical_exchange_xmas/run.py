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
    """Load configuration from TOML file."""
    logging.info("Reading in config")
    with config_path.open("rb") as file:
        return tomllib.load(file)


def validate_config(config: dict[str, Any]) -> None:
    """Validate configuration data for consistency and completeness."""
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
    ly_gifts_path: Path = Path("data/input/ly_gifts.csv"),
    ty_signup_path: Path = Path("data/input/ty_signup.csv"),
) -> tuple[pd.DataFrame, list[str]]:
    """Load last year's gifts and this year's signups."""
    logging.info("Reading in last year gifts")
    ly_gifts = pd.read_csv(ly_gifts_path)
    assert ly_gifts["giver"].is_unique
    assert set(ly_gifts["giver"]).issubset(eligible_people), "Ineligible LY gifter"

    logging.info("Reading in this year's signups")
    signups = duckdb.read_csv(str(ty_signup_path)).filter("is_secret_santa").df()
    assert set(signups["person"]).issubset(eligible_people), "People signed up aren't eligible"
    assert signups["person"].is_unique

    # Get the people signed up this year
    people_signed_up = signups["person"].to_list()

    return ly_gifts, people_signed_up


def create_basic_constraints(
    gifts: cp.Variable, people_signed_up: list[str], gifts_per_person: int
) -> list[cp.Constraint]:
    """Create basic constraints: no self-gifting and gift count constraints."""
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
    """Create constraints to prevent repeating last year's assignments."""
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
    """Create constraints for couples."""
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
    """Create constraints to limit gifts within families."""
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
    """Create constraints to prevent cycles (A->B and B->A)."""
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
    """Create constraints for manual disallows."""
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
    """Set up and solve the optimization problem."""
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
    """Process optimization results into a DataFrame."""
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
    assignments_path: Path = Path("data/output/assignments.csv"),
    messages_path: Path = Path("data/output/secret_santa_messages.md"),
) -> None:
    """Generate and save output files."""
    logging.info("Writing out results")
    result.to_csv(assignments_path, index=False)

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
    """Main function to orchestrate the Secret Santa algorithm."""
    # Load and validate configuration
    config = load_config()
    validate_config(config)

    # Extract config values
    seed = config["seed"]
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
        ly_gifts_path=Path("data/input/ly_gifts.csv"),
        ty_signup_path=Path("data/input/ty_signup.csv"),
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
        assignments_path=Path("data/output/assignments.csv"),
        messages_path=Path("data/output/secret_santa_messages.md"),
    )


if __name__ == "__main__":
    main()
