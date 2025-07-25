# Medical QA

Answer bot from a given set of reference medical answers

# Installation

- Clone this repository and go to its folder
- Create the folder `data` and place file `intern_screening_dataset.csv` in it
- Download GloVe embeddings from `https://nlp.stanford.edu/data/glove.6B.zip` and unzip them into the `glove_embeddings` folder. This code uses `glove.6B.50d.txt`.
- Create the environment: `conda env create -f environment.yml`
- Install Ollama: https://ollama.com/ and run it

# Running the code (inference)

After the Installation steps, activate the environment and run the demo UI:

```
conda activate medqa_env
python medical_assistant.py
```

Sample interactions can be found in the [interaction log file](log.md).
These samples demonstrate that LLM based models have more flexibility to handle out-of-distribution questions while the search-based model (using GloVe to retrieve relevant QA pairs) works better when the structure of the questions is more similar to what is presented in the base data set.

# Solution Approach

In order to expedite the development of the solution and obtain high-quality results, a pre-trained LLM model was used. The LLM model qwen3:1.7b was chosen because it provides good compromise considering runtime, time allowed for developing the solution and quality of responses.

The first model is a zero-shot approach without using any training data with the LLM.
The second model uses GloVe Embeddings to compute the similarity of the user query with the questions in the training set. This approach tends to give less flexible results, but it is also significantly faster.
The third approach combines QA pairs retrieved using GloVe embeddings and uses them as extra context for the LLM. If reference QA pairs are found, they should improve the quality of the responses.

## Evaluation

BLEU and Rouge metrics are used to evaluate the models. These are widely used metrics that capture precision and recall using text similarity strategies.

Modern LLMs introduce the possibility of using LLM judges. The code to use a LLM as judge has been implemented in this repository. However, due to time constraints (the LLM judge takes many hours to run in this dataset), its results were not used. Please disconsider the LLM judge metrics (marked with N/A).

The approach presented in https://moonshotai.github.io/Kimi-K2/ was adapted to create a more reliable evaluation based on:
```
Based on the reference answers:
Correctness: Does the provided answer provide accurate information? (0 - inaccurate or irrelevant answer, 1 - partial answer, 2 - complete answer as expected)
No data invented: Does the provided answer invent new data not available in the reference answers? (0 - yes, 1 - no)
Completeness: Is the provided answer complete, meaning no information was missed? (0 - crucial information is missing, 1 - some information is missing, 2 - answer is complete)
Conciseness: Is the provided answer too long or too short compared to the reference? (0 - too long or too short, 1 - adequate length).
```

Running the evaluation script: `python -m medqa.evaluate_models`

## Data preprocessing

The available data is split into 80/10/10 for training, validation and test. All proposed models may use data from the training set. The model with the best performance on the validation set is selected and the performance on real data is estimated using the test data (only once).

In order to maintain consistency on which samples are selected for each split, the hash trick is applied to the row number (assumed to be the sample id). This allows the models to accomodate future samples without mixing data from different splits.

## Models

State-of-the-art LLMs provide excellent performance on multiple language tasks and are usually an excellent starting point for natural language applications.
However, in this particular case, medical data may be sensitive and there may be concerns about sending data to a LLM provider. Considering the inference costs and assuming that it may be necessary to evaluate and generate answers locally, only models that can run using a single personal computer were chosen.

The following approaches were used:

- Data-retrieval using GloVe: the training set is explicitly used as a knowledge base using GloVe embeddings on the questions (to compare user questions with available QA pairs in the training set)
- Zero-shot LLM: the base LLM is used without any extra instruction
- LLM with data-retrieval: the base LLM receives QA pairs suggested by GloVe to inform its answer decision (only using questions from the training set)

## Summary of the results

Data-retrieval using GloVe:
- Pros:
    - Very fast inference
    - Results are reliable since they are verbatim copies of a curated set
    - Best overall scores (BLEU and rouge)
- Cons:
    - Not conversational (only answers one isolated question at a time)
    - Cannot adapt to the user question. E.g. (compare condition X with condition Y)

Zero-shot LLM:
- Pros:
    - Conversational
    - Can adapt to the user question. E.g. (compare condition X with condition Y)
- Cons:
    - Slow inference
    - Results may not be reliable

LLM with data retrieval:
- Pros:
    - Conversational
    - Can adapt to the user question. E.g. (compare condition X with condition Y)
    - Can take advantage of up-to-date information retrieved from reliable QA pairs
    - Higher scores than the zero-shot LLM
- Cons:
    - Slow inference
    - Did not score as high as GloVe

Result output from `python -m medqa.evaluate_models` (validation set):

```
file               metrics\glove_model_metrics.zip metrics\qa_retrieval_metrics.zip metrics\zero_shot_metrics.zip
number_of_qa_pairs                            1634                             1634                          1634
LLM_judge_avg                              0.0 N/A                          1.0 N/A        0.8524071807425541 N/A
LLM_judge_std                              0.0 N/A                          0.0 N/A       0.23124609755940842 N/A
bleu_avg                                  0.157639                         0.034624                      0.013353
bleu_std                                  0.265394                         0.056064                      0.017178
rouge1_avg                                0.382012                          0.30111                      0.266303
rouge1_std                                 0.23621                         0.110183                      0.089088
rouge2_avg                                0.211948                         0.087507                      0.057408
rouge2_std                                0.278014                         0.072245                      0.033116
rougeL_avg                                0.289674                         0.178179                      0.137508
rougeL_std                                0.257585                         0.080305                      0.040645
```

It would be useful to evaluate the models's performance on the training set as well. However, the time limit would not be enough to run the LLM on all the training sets and the GloVe performance would be 100% perfect (since the questions are matched exactly).

The estimated scores for the model using GloVe for data retrieval are generally acceptable scores for NLP models (~ 0.3 on a scale from 0 to 1):

```
Data-retrieval using GloVe (test scores)
'bleu_avg': 0.1698362096365479
'rouge1_avg': 0.3983421243546301
'rouge2_avg': 0.22713763940435833
'rougeL_avg': 0.3023308305075954
```

## Potential Improvements

In this work, no weight fine-tuning was performed.
Further improvements to speed and model specialization could be obtained by:
- directly fine-tuning the open weight models
- the LLM with data retrieval is a promising and modern approach. It can provide up-to-date, reliable information in a conversational manner by combining the versatility of the LLM and pre-defined relevant QA pairs
- using techniques such as low-rank adaptation (LoRA)
- using pre-trained word embeddings or document embeddings to retrieve the most relevant QA pairs
- the GloVe embeddings approach acts as a primitive RAG. To boost runtime and performance, it would be better to use more advanced vector databases and embedding models
- use GloVe embeddings of higher dimension than 50
- training a custom model using Tensorflow or PyTorch. This was not feasible due to time and processing constraints
- using better metrics (e.g. LLM judge) and out-of-distribution questions to better assess the quality of the answers

## Responsible AI

In a real use case, it is very important to inform users that the AI answer does not dispense with medical advice and is not meant to replace doctors.


## Summary of premises

- Take advantage of modern LLM architectures
- No data should be sent to external LLM providers (i.e., the LLM should run locally)
- The computational constraint is that inference needs to run in a single PC (i.e., constrains models to ~1B-14B parameters)

### Note - Disclaimer

This code was written without third-party AI systems or coding assistants in any part.
No AI assistants were used to develop the code. LLM models were used to create the solution, but not to write any of the code.
Documentation and sample code available online has been consulted and used as reference.
