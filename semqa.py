from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
import time

class SemQA:
    def __init__(self, alpha=0.5, sentence_transformer_model_name="all-mpnet-base-v2", nli_model_name="facebook/bart-large-mnli", cache_dir="/tmp"):
        # For question encoding
        self.q_encoder_model_name = sentence_transformer_model_name
        self.q_encoder = SentenceTransformer(sentence_transformer_model_name, cache_folder=cache_dir)

        # For answer encoding, and NLI scoring
        self.nli_model_name = nli_model_name
        self.tok = AutoTokenizer.from_pretrained(nli_model_name, cache_dir=cache_dir)
        self.nli = AutoModelForSequenceClassification.from_pretrained(nli_model_name, cache_dir=cache_dir).eval()


        # For computing the combined score, alpha is the weight that balances the two components
        # A higher alpha means more weight on Q_recall, and a lower alpha means more weight on A_entail
        self.alpha = alpha


    def prepare_dataset(self, gold: dict, pred: dict):
        # Extracting the questions and answers from the gold and pred datasets
        gold_qs, gold_as = [], []
        pred_qs, pred_as = [], []

        # Pull out the raw lists
        current_gold_qs = gold.get("questions", [])
        if not current_gold_qs:
            print("Warning: The gold dataset does not contain any questions.")
            print("gold:", gold)
            print("pred:", pred)
            raise ValueError("The gold dataset must contain at least one question.")

        # Build gold question and answer lists
        for item in current_gold_qs:
            q = item["question"]
            a = item["answers"][0]["answer"]

            if not a or not q:
                raise ValueError("Gold: question or answer is empty: {} {}".format(q, a))

            gold_qs.append(q)
            gold_as.append(a)

        # sanity check
        if len(gold_qs) != len(gold_as):
            raise ValueError("Gold: #questions != #answers")

        # Now do the same for predictions
        current_pred_qs = pred.get("evidence", [])
        if not current_pred_qs:
            raise ValueError("The pred dataset must contain at least one evidence item.")

        for ev in current_pred_qs:
            q = ev["question"]
            a = ev.get("answer", "")
            pred_qs.append(q)
            pred_as.append(a)

        if len(pred_qs) != len(pred_as):
            raise ValueError("Pred: #questions != #answers")

        return gold_qs, gold_as, pred_qs, pred_as

    def _calculate_Q_recall(self, gold_qs, pred_qs):
        # Encoding the questions
        q_emb_g = self.q_encoder.encode(gold_qs, convert_to_tensor=True)
        q_emb_p = self.q_encoder.encode(pred_qs, convert_to_tensor=True)

        # Calculating the cost
        # This is based on cosine similarity between the question embeddings
        cost = 1 - util.cos_sim(q_emb_g, q_emb_p).cpu().numpy()

        # Hungarian match to minimize cost (maximize sim)
        # Linear sum assignment is the Hungarian algorithm
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
        # The row and column indices of the optimal assignment are returned
        sim_matrix = linear_sum_assignment(cost)
        self.row, self.col = sim_matrix

        # Similarity scores for the matched pairs
        sims = 1 - cost[self.row, self.col] 

        # Q_recall is the mean of the similarity scores
        return sims.mean()

    def calculate_A_entail(self, gold_as, pred_as):
        # Check that the similarity matrix has been computed
        if not hasattr(self, 'row') or not hasattr(self, 'col'):
            raise ValueError("Q_recall must be calculated before A_entail.")

        # now for each matched pair compute NLI entail/neutral probability
        entail_scores = []
        for i,j in zip(self.row, self.col):
            premise, hypothesis = pred_as[j], gold_as[i]
            enc = self.tok(premise, hypothesis, return_tensors="pt", truncation=True)
            with torch.no_grad():
                logits = self.nli(**enc).logits.squeeze()
                probs = torch.softmax(logits, dim=-1)
            # bart-mnli: labels are [contradiction, neutral, entailment]
            entail_scores.append((probs[1] + probs[2]).item())

        A_entail = np.mean(entail_scores)

        return A_entail

    def score(self, gold_qs, pred_qs, gold_as, pred_as):
        # Compute Q_recall
        start = time.perf_counter()
        Q_recall = self._calculate_Q_recall(gold_qs, pred_qs)
        q_recall_elapsed = time.perf_counter() - start

        # Compute A_entail
        start = time.perf_counter()
        A_entail = self.calculate_A_entail(gold_as, pred_as)    
        entail_elapsed = time.perf_counter() - start


        # Calculate the composite score, by using the alpha parameter
        composite = self.alpha * Q_recall + (1 - self.alpha) * A_entail

        return {
            "Q_recall":  Q_recall,
            "A_entail":  A_entail,
            "composite": composite,
            "Q_recall_elapsed": q_recall_elapsed,
            "A_entail_elapsed": entail_elapsed,
        }