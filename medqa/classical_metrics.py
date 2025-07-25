import nltk
from rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu

nltk.download("punkt")
nltk.download("punkt_tab")

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def get_bleu_score(ref_answers, cand_answer):
    tokenized_refs = [word_tokenize(ref.lower()) for ref in ref_answers]
    tokenized_cand = word_tokenize(cand_answer.lower())

    bleu_score = sentence_bleu(tokenized_refs, tokenized_cand)
    return bleu_score


def get_rouge_scores(ref_answers, cand_answer):
    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for ref in ref_answers:
        scores = scorer.score(ref.lower(), cand_answer.lower())
        rouge_scores["rouge1"].append(scores["rouge1"].fmeasure)
        rouge_scores["rouge2"].append(scores["rouge2"].fmeasure)
        rouge_scores["rougeL"].append(scores["rougeL"].fmeasure)

    # Take maximum score for each metric
    rouge1 = max(rouge_scores["rouge1"])
    rouge2 = max(rouge_scores["rouge2"])
    rougeL = max(rouge_scores["rougeL"])

    return rouge1, rouge2, rougeL
