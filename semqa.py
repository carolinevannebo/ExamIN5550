from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
import time

class SemQA:
    def __init__(self, encoder_model_name="all-mpnet-base-v2", cache_dir="/tmp"):
        # For question encoding
        self.encoder_model_name = encoder_model_name
        self.encoder = SentenceTransformer(self.encoder_model_name, cache_folder=cache_dir)


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

    def _q_score_hungarian(self, gold_qs, pred_qs):
        """
        Compute the Hungarian matching score between gold and pred questions.
        """
        # Encode the questions
        em_g = self.encoder.encode(gold_qs, convert_to_tensor=True)
        em_p = self.encoder.encode(pred_qs, convert_to_tensor=True)

        # Calculate the cost matrix based on hungarian matching
        cost = 1 - util.cos_sim(em_g, em_p).cpu().numpy()
        row, col = linear_sum_assignment(cost)
        sims = 1 - cost[row, col]

        # Return the mean similarity, tha matched pairs, and the raw similarities
        return sims.mean(), list(zip(row, col)), sims

    def _q_score_softmax(self, gold_qs, pred_qs):
        """
        Compute the softmax of the cosine similarity between gold and pred questions.
        """
        # Encode the questions
        em_g = self.encoder.encode(gold_qs, convert_to_tensor=True)
        em_p = self.encoder.encode(pred_qs, convert_to_tensor=True)
        

        # Compute cosine similarity
        sim = util.cos_sim(em_g, em_p)

        # Calculate the softmax of the cosine similarity
        probs = torch.softmax(sim, dim=1)
        
        # Get the mean of the probabilities
        return probs.mean().item()

    
    def _a_score_hungarian_filtered( self, gold_as, pred_as, matched_pairs, raw_sims, threshold: float = 0.5, top_k: int = None):

        # Filter out any pairs below threshold
        filtered = [
            (g_idx, p_idx, sim)
            for (g_idx, p_idx), sim in zip(matched_pairs, raw_sims)
            if sim >= threshold
        ]

        # If top_k specified, take top_k by sim descending
        if top_k is not None:
            filtered = sorted(filtered, key=lambda x: x[2], reverse=True)[:top_k]

        if not filtered:
            return 0.0

        # Build filtered answer lists
        gold_f = [gold_as[g] for g, _, _ in filtered]
        pred_f = [pred_as[p] for _, p, _ in filtered]

        em_g = self.encoder.encode(gold_f, convert_to_tensor=True)
        em_p = self.encoder.encode(pred_f, convert_to_tensor=True)

        cost = 1 - util.cos_sim(em_g, em_p).cpu().numpy()
        row, col = linear_sum_assignment(cost)
        sims = 1 - cost[row, col]

        return sims.mean()

    def score(self, gold_qs, pred_qs, gold_as, pred_as, alpha:float = 0.5, variation: str = "hungarian", threshold: float = 0.5, top_k: int = None):

        # Check that the variation is valid
        if variation not in ("hungarian", "softmax"):
            raise ValueError("variation must be 'hungarian' or 'softmax'")

        # Question score
        t0 = time.perf_counter()
        if variation == "hungarian":
            q_score, matched_pairs, raw_sims = self._q_score_hungarian(gold_qs, pred_qs)
        else:
            q_score = self._q_score_softmax(gold_qs, pred_qs)
            # dummy placeholders so code compiles
            matched_pairs, raw_sims = [], []
        q_time = time.perf_counter() - t0

        # Answer score 
        t1 = time.perf_counter()
        a_score = self._a_score_hungarian_filtered(
            gold_as, pred_as,
            matched_pairs=matched_pairs,
            raw_sims=raw_sims,
            threshold=threshold,
            top_k=top_k
        )
        a_time = time.perf_counter() - t1


        # Calulate the composite score
        composite = alpha * q_score + (1 - alpha) * a_score


        # Return the scores
        return {
            "semqa_variation":     variation,
            "semqa_threshold":     threshold,
            "semqa_top_k":         top_k,
            "semqa_alpha":         alpha,
            "semqa_q_score":       q_score,
            "semqa_a_score":       a_score,
            "semqa_q_time":        q_time,
            "semqa_a_time":        a_time,
            "semqa_composite":     composite,
        }