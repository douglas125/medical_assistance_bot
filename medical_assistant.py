import os
import pandas as pd

import gradio as gr

from medqa.datasplit import get_split
from medqa.zero_shot_agent import ZeroShotAgent
from medqa.qa_retrieval_agent import QARetrievalAgent
from medqa.glove_retriever import GloVeSimilarity

# load data
print("Loading data...")
file = os.path.join("data", "intern_screening_dataset.csv")
df = pd.read_csv(file)
row_split = [get_split(x) for x in list(df.index)]
df["split"] = row_split

# splits
df_train = df[df.split == "train"]

qa_agents = [
    ZeroShotAgent(),
    GloVeSimilarity(df_train.copy()),
    QARetrievalAgent(df_train.copy()),
]


def get_answer(question):
    # the words with ? at the end are not found in the GloVe embeddings. split it
    question = question.replace("?", " ?")

    answer1 = qa_agents[0](question)
    answer3 = qa_agents[2](question)
    answer2 = qa_agents[1](question)
    if "</think>" in answer1:
        answer1 = answer1.split("</think>")[-1]
    if "</think>" in answer2:
        answer2 = answer2.split("</think>")[-1]
    if "</think>" in answer3:
        answer3 = answer3.split("</think>")[-1]

    # log answers
    with open('log.md', 'a') as file:
        # Write the content to be appended
        file.write("# Question: \n")
        file.write(f"{question} \n\n")
        file.write("## Zero-shot LLM answer: \n\n")
        file.write(f"{answer1} \n\n")
        file.write("## GloVe model answer: \n\n")
        file.write(f"{answer2} \n\n")
        file.write("## LLM + GloVe model answer: \n\n")
        file.write(f"{answer3} \n\n")
        file.write("\n\n")

    return answer1, answer2, answer3


def main():
    with gr.Blocks(title="Medical Assistant") as demo:
        gr.Markdown("# Medical Assistant")

        with gr.Row():
            question_input = gr.Textbox(
                label="Medical question",
                placeholder="Type your question and click Submit",
                scale=4,
            )
            submit_btn = gr.Button("Submit", scale=1)

        txt_zero_shot = gr.Textbox(label="Zero-shot LLM")
        txt_glove = gr.Textbox(label="Answer retrieved with GloVe Embeddings")
        txt_llm_glove = gr.Textbox(label="LLM using GloVe reference QA pairs")

        submit_btn.click(
            fn=get_answer,
            inputs=question_input,
            outputs=[txt_zero_shot, txt_glove, txt_llm_glove],
        )
    demo.launch()


if __name__ == "__main__":
    main()
