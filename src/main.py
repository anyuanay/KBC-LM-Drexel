import json
from file_io import read_lm_kbc_jsonl
from MyTools import Args, evaluator
from Processors import *


def run(args: Args):
    logger = args.logger

    # Load the input file
    logger.info(f"Loading the input file \"{args.input}\"...")
    input_rows = read_lm_kbc_jsonl(args.input)
    logger.info(f"Loaded {len(input_rows):,} rows.")

    # create buckets
    logger.info(f"Creating buckets.")
    buckets = {relation: [] for relation in args.selected_relations}
    for row in input_rows:
        if row["Relation"] in args.selected_relations:
            buckets[row["Relation"]].append(row)
    logger.info("Created {} buckets successful.\n".format(len(buckets)))

    # process each relation based on different strategies
    results = []
    for relation, processor in args.pipelines.items():
        # test some relations
        if processor is not None:
            logger.info("Processing {} relation...".format(relation))
            results.extend(processor(buckets[relation], logger))
            logger.info("Finished {} relation.\n".format(relation))
        else:
            logger.info("Jump over {} relation!\n".format(relation))

    # Save the results
    logger.info(f"Saving the results to \"{args.output}\"...")
    with open(args.output, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


if __name__ == '__main__':
    my_pipelines = {
        "ChemicalCompoundElement": chemical_compound_element,  # None,
        "PersonCauseOfDeath": person_cause_of_death,  # None,
        "CompanyParentOrganization": company_parent_organization,  # None,

        "CountryBordersWithCountry": country_borders_with_country,  # None,
        "CountryOfficialLanguage": country_official_language,  # None,
        "StateSharesBorderState": state_shares_border_state,   # None,

        "RiverBasinsCountry": river_basins_country,  # None,
        "PersonLanguage": person_language,  # None,
        "PersonProfession": person_profession,  # None,

        "PersonInstrument": person_instrument,   # None,
        "PersonEmployer": person_employer,   # None,
        "PersonPlaceOfDeath": person_place_of_death,   # None,
    }
    my_args = Args(pipelines=my_pipelines)
    my_args.input = "./data/test.jsonl"
    my_args.output = "./data/predictions.jsonl"

    run(my_args)
    # evaluator(my_args)
