import numpy as np
from numpy.linalg import norm

from tqdm.auto import tqdm


class GloVeSimilarity:
    def __init__(self, df_qa, emb_file="glove_embeddings/glove.6B.50d.txt"):
        """Constructor.
        Arguments:
            df_qa: dataframe containing question and answer pairs
        """
        self.df_qa = df_qa
        with open(emb_file, "r", encoding="utf-8") as f:
            emb_data = f.read().splitlines()
        self.embeddings = {}
        self.embdim = None
        for line in tqdm(emb_data, desc="Reading GloVe"):
            line_split = line.split()
            self.embeddings[line_split[0]] = np.array(
                [float(x) for x in line_split[1:]]
            )
            if self.embdim is None:
                self.embdim = len(self.embeddings[line_split[0]])

        # precompute the qa embeddings
        self.qa_embeddings = []
        for idx, row in tqdm(
            df_qa.iterrows(), desc="Precomputing QA embeddings", total=len(df_qa)
        ):
            # Using the answer seems detrimental -  + " " + str(row["answer"])
            cur_emb = str(row["question"])
            cur_emb = self._get_sentence_emb(cur_emb)
            self.qa_embeddings.append(cur_emb)

    def _get_emb(self, x):
        """Retrieve the embedding of a token"""
        return self.embeddings.get(x.lower().strip(), np.zeros(self.embdim))

    def similarity(self, x, y):
        """Cosine similarity"""
        return np.dot(x, y) / (norm(x) * norm(y))

    def _get_sentence_emb(self, sentence):
        """Return the embedding of the sentence.
        A simple way of doing it is to average the embedding values
        """
        return np.mean([self._get_emb(x) for x in sentence.split()], axis=0)

    def find_documents(self, search_string, top_n=5):
        """Returns documents that match the search_string according to
        GloVe embeddings + similarity score
        """
        search_emb = self._get_sentence_emb(search_string)
        similarity_vec = [-self.similarity(search_emb, v) for v in self.qa_embeddings]
        match_order = np.argsort(similarity_vec)
        return self.df_qa.iloc[match_order[0:top_n]]

    def __call__(self, question):
        ans = self.find_documents(question, top_n=1)
        return ans.iloc[0].answer
