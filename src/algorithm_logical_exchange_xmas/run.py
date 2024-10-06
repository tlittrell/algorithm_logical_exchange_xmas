import tomllib
import itertools
import pandas as pd
import duckdb
import numpy as np
import cvxpy as cp

if __name__ == "__main__":
    print("Reading in config")
    with open("local_config.toml", "rb") as file:
        local_config = tomllib.load(file)

    seed = local_config["seed"]
    couples = local_config["couples"]
    families = local_config["families"]
    eligible_people = local_config["eligible_people"]

    print("Validating config")
    assert len(set(eligible_people)) == len(
        eligible_people
    ), "eligible people contains duplicates"

    people_in_couples = list(itertools.chain(*couples))
    assert len(set(people_in_couples)) == len(
        people_in_couples
    ), "Couples contains duplicates"
    assert set(people_in_couples).issubset(set(eligible_people))

    people_in_families = list(itertools.chain(*families))
    assert len(set(people_in_families)) == len(
        people_in_families
    ), "Couples contains duplicates"
    assert set(people_in_families) == set(
        eligible_people
    ), f"Not everyone assigned a family"

    assert seed >= 0
    assert isinstance(seed, int)

    print("Reading in last year gifts")
    ly_gifts = pd.read_csv("data/input/ly_gifts.csv")
    assert ly_gifts["giver"].is_unique
    assert set(ly_gifts["giver"]).issubset(eligible_people), "Ineligible LY gifter"

    print("Reading in this year's signups")
    query = """
    select
        *
    from read_csv('data/input/ty_signup.csv')
    where is_secret_santa
    """
    signups = duckdb.sql(query).df()
    assert set(signups["person"]).issubset(
        eligible_people
    ), "People signed up aren't eligible"
    assert signups["person"].is_unique

    # Get the people signed up this year
    people_signed_up = signups["person"].to_list()
    n_people = len(people_signed_up)

    np.random.seed(seed)
    # Set up novelty matrix
    novelty = np.random.random((n_people, n_people))

    ### Decision variables. Row is person giving, column is person receiving
    gifts = cp.Variable((n_people, n_people), boolean=True)

    ### Objective. Maximize novelty
    objective = cp.Maximize(cp.sum(cp.multiply(gifts, novelty)))

    ### Constraints
    constraints = []
    # Can't give to yourself
    for i, person in enumerate(people_signed_up):
        constraints.append(gifts[i, i] == 0)

    # Each person gives 2 gifts and receives 2 gifts
    constraints.append(cp.sum(gifts, axis=0) == np.full(n_people, 2))
    constraints.append(cp.sum(gifts, axis=1) == np.full(n_people, 2))

    # No repeats of last year
    for row in ly_gifts.iterrows():
        gifter = row[1]["giver"]
        receiver1 = row[1]["gift1"]
        receiver2 = row[1]["gift2"]

        giver_idx = (
            people_signed_up.index(gifter) if gifter in people_signed_up else None
        )
        gift1_idx = (
            people_signed_up.index(receiver1) if receiver1 in people_signed_up else None
        )
        gift2_idx = (
            people_signed_up.index(receiver2) if receiver2 in people_signed_up else None
        )

        if giver_idx is not None and gift1_idx is not None:
            constraints.append(gifts[giver_idx, gift1_idx] == 0)
        if giver_idx is not None and gift2_idx is not None:
            constraints.append(gifts[giver_idx, gift2_idx] == 0)

    # Couples can't give to each other
    for person1, person2 in couples:
        if person1 in people_signed_up and person2 in people_signed_up:
            idx1 = people_signed_up.index(person1)
            idx2 = people_signed_up.index(person2)
            constraints.append(gifts[idx1, idx2] == 0)
            constraints.append(gifts[idx2, idx1] == 0)

    # No person can give 2 gifts within their family or receive 2 gifts within their family
    for family in families:
        family_idx = [
            people_signed_up.index(person)
            for person in family
            if person in people_signed_up
        ]
        for person in set(family).intersection(set(people_signed_up)):
            idx = people_signed_up.index(person)
            constraints.append(cp.sum(gifts[idx, family_idx]) <= 0)
            constraints.append(cp.sum(gifts[family_idx, idx]) <= 0)

    # No cycles e.g. if person 1 gives to person 2 then person 2 can't
    # give to person 1
    for person1, person2 in itertools.product(people_signed_up, people_signed_up):
        idx1 = people_signed_up.index(person1)
        idx2 = people_signed_up.index(person2)
        constraints.append(gifts[idx1, idx2] + gifts[idx2, idx1] <= 1)

    ### Create the integer programming problem
    problem = cp.Problem(objective, constraints)

    # Solve the problem
    problem.solve()

    # Display the results
    if problem.status == cp.OPTIMAL:
        print("Optimal solution found")
        print("Optimal matrix X:")
        print(gifts.value)
        print("Optimal objective value =", problem.value)
    else:
        print("No optimal solution found")

    giver = []
    gift1 = []
    gift2 = []
    for i, person in enumerate(people_signed_up):
        receivers = [people_signed_up[j] for j in np.where(gifts.value[i] == 1)[0]]
    giver.append(person)
    gift1.append(receivers[0])
    gift2.append(receivers[1])
    result = pd.DataFrame({"giver": giver, "gift1": gift1, "gift2": gift2}).merge(
        ly_gifts.rename(
            columns={"gift1": "gift1_ly", "gift2": "gift2_ly", "person": "giver"}
        ),
        on="giver",
        validate="1:1",
    )
    print("Writing out results")
    result.to_csv("data/output/assignments.csv", index=False)