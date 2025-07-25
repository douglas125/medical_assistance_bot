""" Script to automate the evaluation of the models
"""
import os
import time

import pandas as pd
from tqdm.auto import tqdm

import medqa.judge
from medqa.datasplit import get_split
from .zero_shot_agent import ZeroShotAgent
from .qa_retrieval_agent import QARetrievalAgent
from .glove_retriever import GloVeSimilarity
from .eval_nlp_metrics import perform_eval

# Unfortunately, due to the long time it takes
# to run the LLM judge (~6h), it won't be possible
# to compute this metric for all cases
USE_JUDGE_EVAL = False


def main():
    judge_llm = medqa.judge.AnswerJudge()

    # load data
    print("Loading data...")
    file = os.path.join("data", "intern_screening_dataset.csv")
    df = pd.read_csv(file)
    row_split = [get_split(x) for x in list(df.index)]
    df["split"] = row_split

    # splits
    df_train = df[df.split == "train"]
    df_val = df[df.split == "val"]
    df_test = df[df.split == "test"]

    # groups all reference answers in validation
    df_val_agg = df_val.groupby("question")["answer"].apply(list).reset_index()
    df_test_agg = df_test.groupby("question")["answer"].apply(list).reset_index()

    # get QA bots and where to save their evaluation
    qa_agents = [
        {
            "agent": ZeroShotAgent(),
            "metric_file": os.path.join("metrics", "zero_shot_metrics.zip"),
        },
        {
            "agent": QARetrievalAgent(df_train),
            "metric_file": os.path.join("metrics", "qa_retrieval_metrics.zip"),
        },
        {
            "agent": GloVeSimilarity(df_train),
            "metric_file": os.path.join("metrics", "glove_model_metrics.zip"),
        },
    ]

    for model in qa_agents:
        # do not recompute metrics if they exist
        if not os.path.isfile(model["metric_file"]):
            t0 = time.time()
            all_answers = []
            all_metrics = []
            all_reasoning = []
            # inference - get the answers
            for idx, row in tqdm(
                df_val_agg.iterrows(),
                desc="Inference: " + model["metric_file"],
                total=len(df_val_agg),
            ):
                candidate_ans = model["agent"](row["question"])
                if "</think>" in candidate_ans:
                    candidate_ans = candidate_ans.split("</think>")[-1]
                all_answers.append(candidate_ans)
            inference_time = time.time() - t0
            print(f"Inference time: {inference_time:.2f} s")

            t0 = time.time()
            for idx, row in tqdm(
                df_val_agg.iterrows(),
                desc="Judge: " + model["metric_file"],
                total=len(df_val_agg),
            ):
                if USE_JUDGE_EVAL:
                    reasoning, metrics = judge_llm.judge_answer(
                        row["question"], row["answer"], all_answers[idx]
                    )
                else:
                    reasoning = ""
                    metrics = "[0, 0, 0, 0]"
                all_metrics.append(metrics)
                all_reasoning.append(reasoning)
            eval_time = time.time() - t0
            print(f"Evaluation time: {eval_time:.2f} s")

            df_val_agg_copy = df_val_agg.copy()
            df_val_agg_copy["candidate_ans"] = all_answers
            df_val_agg_copy["metrics"] = all_metrics
            df_val_agg_copy["reasoning"] = all_reasoning

            df_val_agg_copy.to_csv(model["metric_file"], index=False)

    # compute and show classical metrics
    perform_eval(qa_agents[2]["agent"], df_test_agg.copy())


if __name__ == "__main__":
    main()
