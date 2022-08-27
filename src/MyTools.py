from baseline import PromptSet, logger as baseline_logger
from evaluate import evaluate_per_sr_pair, combine_scores_per_relation
from pandas import DataFrame
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
from nltk.corpus import stopwords

stop = stopwords.words('english')
stop.extend([
    "first", "company", "Taiwan", "architect", "author", "cancer", "engineer"
])


def create_prompt(subject_entity: str, relation: str, mask_token: str) -> str:
    prompt = mask_token

    if relation == "ChemicalCompoundElement":
        prompt = f"{subject_entity} consits of {mask_token}, which is a chemical element."

    elif relation == "CompanyParentOrganization":
        prompt = f"The parent organization of {subject_entity} is {mask_token} company."

    elif relation == "IsDead":
        prompt = f"{subject_entity} has already {mask_token}."

    elif relation == "PersonCauseOfDeath":
        prompt = f"{subject_entity} died due to {mask_token}."

    elif relation == "CountryBordersWithCountry":
        prompt = f"{mask_token} and {subject_entity} are neighboring country. They share the border."

    elif relation == "CountryOfficialLanguage":
        prompt = f"{subject_entity}'s official language is {mask_token}."

    elif relation == "PersonEmployer":
        prompt = f"{subject_entity} joined and work at {mask_token} as an employer, which is a company"

    elif relation == "PersonInstrument":
        prompt = f"{subject_entity} loves playing {mask_token}, which is a instrument."

    elif relation == "PersonPlaceOfDeath":
        prompt = f"{subject_entity} died at home or hospital in {mask_token}."

    elif relation == "PersonLanguage":
        prompt = f"{subject_entity} speaks in {mask_token}, which is a language."

    elif relation == "RiverBasinsCountry":
        prompt = f"{subject_entity} river basins in {mask_token}."

    elif relation == "PersonProfession":
        prompt = f"{subject_entity} is (a or an) {mask_token}, which is a profession."

    return prompt


class Args:
    model_type = "bert-large-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    model = AutoModelForMaskedLM.from_pretrained(model_type)

    def __init__(self, input_rows=None, logger=None, pipelines=None, top_k=100, threshold=0.5):
        # for greenline
        self.input = "data/dev.jsonl"
        self.output = "data/dev.pred.jsonl"
        self.pipelines = pipelines
        self.selected_relations = [relation
                                   for relation, processor in pipelines.items()
                                   if processor is not None] if pipelines is not None else None

        self.input_rows = input_rows
        self.relation = input_rows[0]["Relation"] if input_rows is not None else None
        self.top_k = top_k
        self.threshold = threshold

        self.gpu = -1
        self.logger = logger if logger is not None else baseline_logger
        self.model_type = "bert-large-cased"
        self.tokenizer = Args.tokenizer
        self.model = Args.model


def run_single_relation(args):
    logger = args.logger
    mask_token = args.tokenizer.mask_token

    # Load the input rows
    input_rows = args.input_rows

    # Load the pipeline
    logger.info("Loading the pipeline with top_k = {0} and threshold = {1}.".format(args.top_k, args.threshold))
    pipe = pipeline(task="fill-mask", model=args.model, tokenizer=args.tokenizer, top_k=args.top_k, device=args.gpu)

    # Create prompts
    logger.info(f"Creating prompts...")
    prompts = PromptSet([create_prompt(
        subject_entity=row["SubjectEntity"],
        relation=args.relation,
        mask_token=mask_token,
    ) for row in input_rows])

    # Run the model
    logger.info(f"Running the model...")
    outputs = []
    for out in tqdm(pipe(prompts, batch_size=8), total=len(prompts)):
        outputs.append(out)

    # filter by threshold
    results = []
    n = 0  # how many stopwords removed
    for row, prompt, output in zip(input_rows, prompts, outputs):
        result = {
            "SubjectEntity": row["SubjectEntity"],
            "Relation": row["Relation"],
            "Prompt": prompt,
            "ObjectEntities": [
                seq["token_str"]
                for seq in output
                if (seq["score"] > args.threshold) and (seq["token_str"] not in stop)
            ]
        }
        # try stopwords removal------------------------------------------------------------
        for seq in output:
            if seq["score"] > args.threshold:
                if seq["token_str"] not in stop:
                    result["ObjectEntities"].append(seq["token_str"])
                else:
                    n = n + 1
        # ---------------------------------------------------------------------------------
        results.append(result)
    logger.info(f" - removed stopwords: {n}.")
    return results


def evaluator(args: Args):
    scores_per_sr_pair = evaluate_per_sr_pair(args.output,  # predictions
                                              args.input)  # ground_truth
    scores_per_relation_unfiltered = combine_scores_per_relation(scores_per_sr_pair)
    scores_per_relation = {r: scores_per_relation_unfiltered[r] for r in args.selected_relations}
    scores_per_relation["*** Average ***"] = {
        "p": sum([x["p"] for x in scores_per_relation.values()]) / len(
            scores_per_relation),
        "r": sum([x["r"] for x in scores_per_relation.values()]) / len(
            scores_per_relation),
        "f1": sum([x["f1"] for x in scores_per_relation.values()]) / len(
            scores_per_relation),
    }

    print(DataFrame(scores_per_relation).transpose().round(3))
