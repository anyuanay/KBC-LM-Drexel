from MyTools import Args, create_prompt, run_single_relation
from baseline import PromptSet
from pandas import DataFrame
from spacy.lang.en import English
from tqdm.auto import tqdm
from transformers import pipeline


def chem_helper(args):
    args.logger.info(f"Processing...")
    match_prob = 0.7
    elements, elem = chem_elements()

    mask_token = args.tokenizer.mask_token
    tokenizer = English().tokenizer
    pipe = pipeline(task="fill-mask", model=args.model, tokenizer=args.tokenizer, top_k=args.top_k, device=args.gpu)
    results = []
    subject_entities = [row["SubjectEntity"] for row in args.input_rows]
    for subject_entity in tqdm(subject_entities):
        compounds = set()
        # tokenize and linguistic match ---------------------------------------------
        tokens = tokenizer(subject_entity)
        if len(tokens) > 1:
            for token in tokens:
                token = str(token)
                token_l = str(token).lower()

                if token_l[:4] in elem:
                    compound = elements[elem.index(token_l[:4])]
                    if compound not in compounds:
                        compounds.add(compound)
                        results.append({
                            "Prompt": "linguistic match",
                            "SubjectEntity": subject_entity,
                            "ObjectEntity": compound,
                            "Probability": match_prob})
                # sub-test -----------------------------------------------------------------
                prompt = create_prompt(token, args.relation, mask_token)
                probe_outputs = pipe(prompt)
                for sequence in probe_outputs:
                    if sequence["token_str"][:4] in elem and sequence["token_str"] not in compounds:
                        compounds.add(sequence["token_str"])
                        results.append(
                            {
                                "Prompt": "sub_test: " + prompt,
                                "SubjectEntity": subject_entity,
                                "ObjectEntity": sequence["token_str"],
                                "Probability": round(sequence["score"], 4),
                            }
                        )
        # total-test ---------------------------------------------------------------
        prompt = create_prompt(subject_entity, args.relation, mask_token)
        probe_outputs = pipe(prompt)
        for sequence in probe_outputs:
            if sequence["token_str"][:4] in elem and sequence["token_str"] not in compounds:
                results.append(
                    {
                        "Prompt": prompt,
                        "SubjectEntity": subject_entity,
                        "ObjectEntity": sequence["token_str"],
                        "Probability": round(sequence["score"], 4),
                    }
                )

    results_df = DataFrame(results)
    results = []
    results_df[results_df["Probability"] > args.threshold] \
        .groupby("SubjectEntity") \
        .apply(
        lambda row: results.append(
            {
                "SubjectEntity": row["SubjectEntity"].unique()[0],
                "Relation": args.relation,
                "Prompt": row["Prompt"].unique()[0],
                "ObjectEntities": list(row["ObjectEntity"])
            }
        )
    )
    return results


def chem_elements():
    elements = ['Hydrogen', 'Helium', 'Lithium', 'Beryllium', 'Boron', 'Carbon', 'Nitrogen', 'Oxygen', 'Fluorine',
                'Neon', 'Sodium', 'Magnesium', 'Aluminum', 'Silicon', 'Phosphorus', 'Sulfur', 'Chlorine', 'Argon',
                'Potassium', 'Calcium', 'Scandium', 'Titanium', 'Vanadium', 'Chromium', 'Manganese', 'Iron', 'Cobalt',
                'Nickel', 'Copper', 'Zinc', 'Gallium', 'Germanium', 'Arsenic', 'Selenium', 'Bromine', 'Krypton',
                'Rubidium', 'Strontium', 'Yttrium', 'Zirconium', 'Niobium', 'Molybdenum', 'Technetium', 'Ruthenium',
                'Rhodium', 'Palladium', 'Silver', 'Cadmium', 'Indium', 'Tin', 'Antimony', 'Tellurium', 'Iodine',
                'Xenon', 'Caesium', 'Barium', 'Lanthanum', 'Cerium', 'Praseodymium', 'Neodymium', 'Promethium',
                'Samarium', 'Europium', 'Gadolinium', 'Terbium', 'Dysprosium', 'Holmium', 'Erbium', 'Thulium',
                'Ytterbium', 'Lutetium', 'Hafnium', 'Tantalum', 'Tungsten', 'Rhenium', 'Osmium', 'Iridium', 'Platinum',
                'Gold', 'Mercury', 'Thallium', 'Lead', 'Bismuth', 'Polonium', 'Astatine', 'Radon', 'Francium', 'Radium',
                'Actinium', 'Thorium', 'Protactinium', 'Uranium', 'Neptunium', 'Plutonium', 'Americium', 'Curium',
                'Berkelium', 'Californium', 'Einsteinium', 'Fermium', 'Mendelevium', 'Nobelium', 'Lawrencium',
                'Rutherfordium', 'Dubnium', 'Seaborgium', 'Bohrium', 'Hassium', 'Meitnerium', 'Darmstadtium',
                'Roentgenium', 'Copernicium', 'Ununtrium', 'Flerovium', 'Ununpentium', 'Livermorium', 'Ununseptium',
                'Ununoctium', 'glyceryl']

    elem = ['hydr', 'heli', 'lith', 'bery', 'boro', 'carb', 'nitr', 'oxyg', 'fluo', 'neon', 'sodi', 'magn', 'alum',
            'sili', 'phos', 'sulf', 'chlo', 'argo', 'pota', 'calc', 'scan', 'tita', 'vana', 'chro', 'mang', 'iron',
            'coba', 'nick', 'copp', 'zinc', 'gall', 'germ', 'arse', 'sele', 'brom', 'kryp', 'rubi', 'stro', 'yttr',
            'zirc', 'niob', 'moly', 'tech', 'ruth', 'rhod', 'pall', 'silv', 'cadm', 'indi', 'tin', 'anti', 'tell',
            'iodi', 'xeno', 'caes', 'bari', 'lant', 'ceri', 'pras', 'neod', 'prom', 'sama', 'euro', 'gado', 'terb',
            'dysp', 'holm', 'erbi', 'thul', 'ytte', 'lute', 'hafn', 'tant', 'tung', 'rhen', 'osmi', 'irid', 'plat',
            'gold', 'merc', 'thal', 'lead', 'bism', 'polo', 'asta', 'rado', 'fran', 'radi', 'acti', 'thor', 'prot',
            'uran', 'nept', 'plut', 'amer', 'curi', 'berk', 'cali', 'eins', 'ferm', 'mend', 'nobe', 'lawr', 'ruth',
            'dubn', 'seab', 'bohr', 'hass', 'meit', 'darm', 'roen', 'cope', 'unun', 'fler', 'unun', 'live', 'unun',
            'unun', 'glyc']

    return elements, elem


def is_dead(input_rows: list, logger):
    args = Args(input_rows, logger, top_k=200, threshold=0.055)
    args.relation = "IsDead"

    output = run_single_relation(args)
    input_rows = []
    results = []

    for entity in output:
        if "died" not in entity["ObjectEntities"]:
            input_rows.append(entity)
        else:
            entity["ObjectEntities"] = []
            results.append(entity)

    return input_rows, results


def border_prompt(subject_entity: str, object_entity: str, mask_token: str) -> str:
    if object_entity:
        object_entity = object_entity[0]
        prompt = f"{subject_entity} and {mask_token} are neighboring states in {object_entity}."
    else:
        prompt = f"{subject_entity} is close to a region: {mask_token}."
    return prompt


def border_helper(args):
    args.logger.info(f"Pre-processing...")
    pipe = pipeline(task="fill-mask", model=args.model, tokenizer=args.tokenizer, top_k=40, device=args.gpu)
    mask_token = args.tokenizer.mask_token
    pre_input_rows = args.input_rows

    args.logger.info(f"Creating prompts...")
    pre_prompts = PromptSet([
        f"{row['SubjectEntity']} is a state in {mask_token}, which is a country."
        for row in pre_input_rows])

    args.logger.info(f"Running the model...")
    pre_outputs = []
    for pre_out in tqdm(pipe(pre_prompts, batch_size=8), total=len(pre_prompts)):
        pre_outputs.append(pre_out)

    pre_results = []
    for row, prompt, output in zip(pre_input_rows, pre_prompts, pre_outputs):
        token_dict = {}
        for seq in output:
            token_dict[seq["token_str"]] = seq['score']

        result = {
            "SubjectEntity": row["SubjectEntity"],
            "Relation": row["Relation"],
            "Prompt": prompt,
            "ObjectEntities": [k for k, v in token_dict.items() if v == max(token_dict.values())],
        }
        pre_results.append(result)

    args.logger.info(f"Secondary-processing...")
    pipe = pipeline(task="fill-mask", model=args.model, tokenizer=args.tokenizer, top_k=args.top_k, device=args.gpu)

    args.logger.info(f"Creating prompts...")
    prompts = PromptSet([border_prompt(
        subject_entity=row["SubjectEntity"],
        object_entity=row["ObjectEntities"],
        mask_token=mask_token,
    ) for row in pre_results])

    args.logger.info(f"Running the model...")
    outputs = []
    for out in tqdm(pipe(prompts, batch_size=8), total=len(prompts)):
        outputs.append(out)

    results = []
    for row, prompt, output in zip(pre_results, prompts, outputs):
        result = {
            "SubjectEntity": row["SubjectEntity"],
            "Relation": row["Relation"],
            "Prompt": prompt,
            "ObjectEntities": [seq["token_str"] for seq in output if seq["score"] > 0.05]
        }
        results.append(result)

    return results
