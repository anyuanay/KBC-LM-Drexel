from MyTools import Args, run_single_relation
from MyHelpers import is_dead, chem_helper, border_helper


def person_place_of_death(input_rows: list, logger):
    input_rows, not_dead = is_dead(input_rows, logger)
    return run_single_relation(Args(input_rows, logger, top_k=10, threshold=1)) + not_dead


def person_employer(input_rows: list, logger):
    return run_single_relation(Args(input_rows, logger, top_k=500, threshold=0.06))


def person_instrument(input_rows: list, logger):
    return run_single_relation(Args(input_rows, logger, top_k=5, threshold=0.1055))


def state_shares_border_state(input_rows: list, logger):
    return border_helper(Args(input_rows, logger, top_k=80, threshold=0.04))


def country_official_language(input_rows: list, logger):
    return run_single_relation(Args(input_rows, logger, top_k=50, threshold=0.2))


def country_borders_with_country(input_rows: list, logger):
    return run_single_relation(Args(input_rows, logger, top_k=40, threshold=0.04))


def person_language(input_rows: list, logger):
    return run_single_relation(Args(input_rows, logger, top_k=1, threshold=0.184))


def person_profession(input_rows: list, logger):
    return run_single_relation(Args(input_rows, logger, top_k=5, threshold=0.01))


def river_basins_country(input_rows: list, logger):
    return run_single_relation(Args(input_rows, logger, top_k=4, threshold=0.071))


def person_cause_of_death(input_rows: list, logger):
    input_rows, not_dead = is_dead(input_rows, logger)
    return run_single_relation(Args(input_rows, logger, top_k=10, threshold=0.8)) + not_dead


def chemical_compound_element(input_rows: list, logger):
    return chem_helper(Args(input_rows, logger, top_k=300, threshold=0.055))


def company_parent_organization(input_rows: list, logger):
    return run_single_relation(Args(input_rows, logger, top_k=20, threshold=0.6))
