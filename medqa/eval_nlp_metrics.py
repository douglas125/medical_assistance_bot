import os
import ast
import numpy as np
import pandas as pd
import warnings

from tqdm.auto import tqdm

from medqa.classical_metrics import get_bleu_score
from medqa.classical_metrics import get_rouge_scores

warnings.filterwarnings("ignore", module="nltk")


def compute_nlp_metrics(df):
    """Receives a dataframe with "answer" and "candidate_answer"
    and computes BLEU and rouge scores
    """
    scores = {"LLM_judge": [], "bleu": [], "rouge1": [], "rouge2": [], "rougeL": []}
    # judge metrics
    for idx, row in df.iterrows():
        eval_string = row["metrics"]

        # since the LLM is small, sometimes it forgets the brackets []
        if not eval_string.startswith("["):
            eval_string = f"[{eval_string}]"
        judge_score = ast.literal_eval(eval_string)

        # since the LLM is small, sometimes the LLM judge assigns score > 1
        judge_score = min(1.0, np.sum(judge_score) / 6.0)
        scores["LLM_judge"].append(judge_score)

    # BLEU and rouge
    for idx, row in tqdm(df.iterrows(), desc="Computing scores", total=len(df)):
        ref_answers = ast.literal_eval(row["answer"])
        cand_answer = row["candidate_ans"]
        bleu_score = get_bleu_score(ref_answers, cand_answer)
        scores["bleu"].append(bleu_score)
        rouge_scores = get_rouge_scores(ref_answers, cand_answer)
        scores["rouge1"].append(rouge_scores[0])
        scores["rouge2"].append(rouge_scores[1])
        scores["rougeL"].append(rouge_scores[2])
    report = {
        "number_of_qa_pairs": len(df),
        "LLM_judge_avg": str(np.mean(scores["LLM_judge"])) + " N/A",
        "LLM_judge_std": str(np.std(scores["LLM_judge"])) + " N/A",
        "bleu_avg": np.mean(scores["bleu"]),
        "bleu_std": np.std(scores["bleu"]),
        "rouge1_avg": np.mean(scores["rouge1"]),
        "rouge1_std": np.std(scores["rouge1"]),
        "rouge2_avg": np.mean(scores["rouge2"]),
        "rouge2_std": np.std(scores["rouge2"]),
        "rougeL_avg": np.mean(scores["rougeL"]),
        "rougeL_std": np.std(scores["rougeL"]),
    }
    return report


def perform_eval(best_qa_agent, df_test):
    all_metrics = []
    metrics_files = os.listdir("metrics")
    # don't pick up files that used test data
    metrics_files = [
        os.path.join("metrics", x) for x in metrics_files if "test" not in x
    ]
    for file in metrics_files:
        df = pd.read_csv(file)
        metrics = compute_nlp_metrics(df)
        metrics["file"] = file
        all_metrics.append(metrics)

    df_results = pd.DataFrame(all_metrics).set_index("file").T
    print(df_results)

    # The GloVe method has better scores across the board in validation so it should be chosen
    # this code was added a posteriori, after noting that GloVe had the highest val score

    glove_test_file = os.path.join("metrics", "glove_test_ans.zip")
    if not os.path.isfile(glove_test_file):
        all_answers = []
        # inference - get the answers
        for idx, row in tqdm(
            df_test.iterrows(),
            desc="Inference on test set",
            total=len(df_test),
        ):
            candidate_ans = best_qa_agent(row["question"])
            all_answers.append(candidate_ans)
        df_test["candidate_ans"] = all_answers
        df_test["metrics"] = "[0,0,0,0]"
        df_test.to_csv(glove_test_file, index=False)
    df_test = pd.read_csv(glove_test_file)
    print("Estimated scores on test set for the best model (GloVe retrieval):")
    test_metrics = compute_nlp_metrics(df_test)
    print(test_metrics)
