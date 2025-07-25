""" QA Judge

LLM used to evaluate the quality of a response, taking into account the following metrics:
Correctness, Was_data_consistent, Completeness, Length
"""
import gat_llm.llm_invoker as inv
from .llm_caller import call_llm


judge_prompt = """<question>
[[BASE_QUESTION]]
</question>

<reference_answers>
[[REFERENCE_ANSWER]]
</reference_answers>

Compare the <reference_answer></reference_answer> with the <answer_to_evaluate></answer_to_evaluate> to decide if the <answer_to_evaluate></answer_to_evaluate> is correct.

Judge the <answer_to_evaluate></answer_to_evaluate> based on the <reference_answers></reference_answers> using the <criteria></criteria>:
<criteria>
<Correctness>Does the provided answer provide accurate information? (0 - inaccurate or irrelevant answer, 1 - partial answer, 2 - complete answer as expected)
</Correctness>
<Was_data_consistent>Does the provided answer only use data that can be validated in the reference answers? (0 - no, 1 - yes)
</Was_data_invented>
<Completeness>Is the provided answer complete, meaning no information was missed? (0 - crucial information is missing, 1 - some information is missing, 2 - answer is complete)
</Completeness>
<Length>Is the provided answer too long or too short compared to the reference (regardless of it being correct)? (0 - too long or too short, 1 - adequate length).
</Length>
</criteria>
Give your final answer in the format:
<metrics>Correctness_number,Was_data_consistent_number,Completeness_number,Length_number</metrics>"""


class AnswerJudge:
    def __init__(self, model="Qwen 3 1.7b - Ollama"):
        """Initializes the answer judge LLM"""
        self.model = model
        self.llm = inv.LLM_Provider.get_llm(None, model)

    def judge_answer(self, question, base_answers, answer_to_evaluate, retries=3):
        cur_sys_prompt = judge_prompt.replace("[[BASE_QUESTION]]", question)

        if isinstance(base_answers, str):
            # if there is only one base answer given as string, convert to list
            base_answers = [base_answers]

        # if there are multiple references, take all of them into consideration in a structure friendly to a LLM
        new_base_ans = (
            "<reference_answer>"
            + "<reference_answer></reference_answer>".join(base_answers)
            + "</reference_answer>"
        )
        cur_sys_prompt = cur_sys_prompt.replace("[[REFERENCE_ANSWER]]", new_base_ans)
        cur_prompt = f"<answer_to_evaluate>{answer_to_evaluate}</answer_to_evaluate>. Evaluate the <answer_to_evaluate></answer_to_evaluate>. Start your answer with <metrics>"
        judge_string = call_llm(self.llm, cur_sys_prompt, cur_prompt)

        # this can raise an error if the LLM decides not to output the metrics
        cur_try = 0
        while cur_try < retries:
            try:
                judge_metrics = judge_string.split("<metrics>")[-1].split("</metrics>")[
                    0
                ]
                judge_metrics = judge_metrics.split(",")
                judge_metrics = [int(x) for x in judge_metrics]

                # cap the metrics in case the LLM decides to assign a score above the highest
                judge_metrics[0] = max(2, judge_metrics[0])
                judge_metrics[1] = max(1, judge_metrics[0])
                judge_metrics[2] = max(2, judge_metrics[0])
                judge_metrics[3] = max(1, judge_metrics[0])
                return judge_string, judge_metrics
            except Exception:
                pass
        return judge_string, None
