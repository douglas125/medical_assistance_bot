""" QA Retrieval Agent

Based on a set of questions whose answers are known, perform a 2-step inference by:

- Selecting which keywords should be used to locate relevant answers
- Retrieving the relevant QA pairs to serve as basis to create an answer for the user
"""
import gat_llm.llm_invoker as inv
from .llm_caller import call_llm
from .glove_retriever import GloVeSimilarity


medqa_prompt = """You are medical question-answering system expert at providing medical information.

Here are some reference QA pairs that may be useful to help answer the user's question:

[[REF_QA_PAIRS]]

Your goal is to effectively answer user queries related to medical diseases.
Use information from the question/answer pairs if and only if the <qa_pairs></qa_pairs> contain information related to the user's question. You are allowed to add more information if needed to answer adequately.
Do not mention the words "QA pairs" in the final answer.
Refuse to answer questions not related to your expertise."""


class QARetrievalAgent:
    def __init__(self, df_ref_qa, model="Qwen 3 1.7b - Ollama"):
        """Initializes the QA retrieval agent.
        Arguments:
            df_ref_qa - the reference dataframe containing question/answer pairs (training data)
        """
        self.model = model
        self.llm = inv.LLM_Provider.get_llm(None, model)
        self.sys_prompt = medqa_prompt
        self.glove_sim = GloVeSimilarity(df_ref_qa)

    def __call__(self, user_question):
        """Provides the answer to a user question based on the QA document"""
        df_relevant_qa = self.glove_sim.find_documents(user_question)[
            ["question", "answer"]
        ]
        qa_xml = df_relevant_qa.to_xml(
            index=False, xml_declaration=False, root_name=f"qa_pairs"
        )
        cur_prompt = self.sys_prompt.replace("[[REF_QA_PAIRS]]", qa_xml)
        ans = call_llm(
            self.llm,
            system_prompt=cur_prompt,
            prompt=user_question,
        )
        return ans
