import tomllib
import itertools
import pandas as pd
import duckdb
import numpy as np
import cvxpy as cp

if __name__ == 'main':
    
    print("Reading in config")
    with open("local_config.toml", "rb") as file:
        local_config = tomllib.load(file)


    seed = local_config["seed"]
    couples = local_config["couples"]
    families = local_config["families"]
    eligible_people = local_config["eligible_people"]

    print("Validating config")
    assert len(set(eligible_people)) == len(eligible_people), "eligible people contains duplicates"

    people_in_couples = list(itertools.chain(*couples))
    assert len(set(people_in_couples)) == len(people_in_couples), "Couples contains duplicates"
    assert set(people_in_couples).issubset(set(eligible_people))

    people_in_families = list(itertools.chain(*families))
    assert len(set(people_in_families)) == len(people_in_families), "Couples contains duplicates"
    assert set(people_in_families) == set(eligible_people), f"Not everyone assigned a family"

    assert seed >= 0
    assert isinstance(seed, int)

    print("Reading in last year gifts")
    ly_gifts = pd.read_csv("data/input/ly_gifts.csv")
    assert ly_gifts["gifter"].is_unique
    assert set(ly_gifts["gifter"]).issubset(eligible_people), "Ineligible LY gifter"

    print("Reading in this year's signups")
    query = """
    select
        *
    from read_csv('data/input/ty_signup.csv')
    where is_secret_santa
    """
    signups = duckdb.sql(query).df()
    assert set(signups["person"]).issubset(eligible_people), "People signed up aren't eligible"
    assert signups["person"].is_unique

    # Get the people signed up this year
    people = signups["person"].to_list()
    n_people = len(people)

    np.random.seed(seed)
    # Set up novelty matrix
    novelty = np.random.random((n_people, n_people))