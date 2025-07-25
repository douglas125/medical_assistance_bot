""" Zero shot agent. Doesn't use any training data
"""
import gat_llm.llm_invoker as inv
from .llm_caller import call_llm


class ZeroShotAgent:
    def __init__(self):
        llm_name = "Qwen 3 1.7b - Ollama"
        self.llm = inv.LLM_Provider.get_llm(None, llm_name)

    def __call__(self, question):
        medqa_prompt = """You are medical question-answering system expert at providing medical information.
Your goal is to effectively answer user queries related to medical diseases.
Refuse to answer questions not related to your expertise."""

        return call_llm(self.llm, medqa_prompt, question)
