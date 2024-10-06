import tomllib
import itertools
import pandas as pd
import duckdb


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

    print("Reading in signups")
    query = """
    select
        *
    from read_csv('data/input/ty_signup.csv')
    where is_secret_santa
    """
    signups = duckdb.sql(query).df()
    assert set(signups["person"]).issubset(eligible_people), "People signed up aren't eligible"
    assert signups["person"].is_unique