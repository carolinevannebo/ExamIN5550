from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
import time

class SemQA:
    def __init__(self, encoder_model_name="all-mpnet-base-v2", nli_model_name="facebook/bart-large-mnli", cache_dir="/tmp"):
        # For question encoding
        self.encoder_model_name = encoder_model_name
        self.encoder = SentenceTransformer(self.encoder_model_name, cache_folder=cache_dir)

        # NLI model 
        self.nli_model_name = nli_model_name
        self.tok = AutoTokenizer.from_pretrained(
            nli_model_name, 
            cache_dir=cache_dir
        )
        self.nli = AutoModelForSequenceClassification.from_pretrained(
            nli_model_name, 
            cache_dir=cache_dir
        ).eval()


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

        # Matched pairs
        matched_pairs = list(zip(row, col))

        # Return the mean similarity, the matched pairs, and the raw similarities
        return sims.mean(), matched_pairs

    def _q_score_softmax(self, gold_qs, pred_qs, threshold:float = 0.5):
        """
        Compute the softmax of the cosine similarity between gold and pred questions.
        """
        # Encode the questions
        em_g = self.encoder.encode(gold_qs, convert_to_tensor=True)
        em_p = self.encoder.encode(pred_qs, convert_to_tensor=True)
        
        # Compute cosine similarity between the embeddings 
        sim = util.cos_sim(em_g, em_p)

        # Calculate the softmax of the cosine similarity
        probs = torch.softmax(sim, dim=0)

        # Apply thresholding to the softmax score
        # Only get similartity score that are about the threshold
        thresholded_probs = probs[probs >= threshold]

        # Calculate the mean of the probabilities 
        mean_prob = thresholded_probs.mean().item()

        # Indexes from where the probability of the probabilities
        mask = probs >= threshold
        i_idxs, j_idxs = mask.nonzero(as_tuple=True)
        matched_pairs = list(zip(i_idxs.tolist(), j_idxs.tolist()))

        # Get the k and the number of batches 
        # Penealize when we dont have many matches 
        k = min(len(gold_qs), len(pred_qs))
        number_matches = len(thresholded_probs)
        
        # Calculate the coefficent 
        coefficent = min(1.0, number_matches / k)

        # Calculate the final score 
        q_score_softmax = coefficent * mean_prob

        # Return the question score for the softmac 
        return q_score_softmax, matched_pairs 
    

    def _calculate_entailment(self, premise, hypothesis):
        # Do entailment direction 
        # See docs: https://huggingface.co/facebook/bart-large-mnli
        enc = self.tok(premise, hypothesis,  return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = self.nli(**enc)[0]

        # From the docomentation of the NLI model, it is the "entailment dim"
        entail_logits = logits[:,[0,2]]
        prob = torch.softmax(entail_logits, dim=1)
        prob_of_entailment = prob[:,1]

        # Return the probability of entailment 
        return prob_of_entailment


    def _a_score_entailment(self, gold_as, pred_as, matched_pairs):
        """
        For each matched pair, calculate the entailment in both direction
        """
        
        # bidirectional entailment
        scores = []
        for (i, j) in matched_pairs: 
            # Get the goal and pred 
            gold = gold_as[i]
            pred = pred_as[j]

            # Forward and backward pass 
            forward = self._calculate_entailment(premise=gold, hypothesis=pred)
            backward = self._calculate_entailment(premise=pred, hypothesis=gold)

            # Add the maximum to the list
            scores.append(max(forward, backward))
        
        # Return the mean of the scores 
        return float(np.mean(scores)) if scores else 0.0

    def score(self, gold_qs, pred_qs, gold_as, pred_as,  alpha:float = 0.5, threshold: float = 0.5, variation: str = "hungarian"):

        # Check that the variation is valid
        if variation not in ("hungarian", "softmax"):
            raise ValueError("variation must be 'hungarian' or 'softmax'")

        # Question score
        t0 = time.perf_counter()
        if variation == "hungarian":
            # Get the score and the mached pairs
            q_score, matched_pairs = self._q_score_hungarian(gold_qs, pred_qs)
        else:
            q_score, matched_pairs = self._q_score_softmax(gold_qs, pred_qs, threshold)

        q_time = time.perf_counter() - t0

        # Answer score 
        t1 = time.perf_counter()

        # Calculate answer score using entailment for the given matched pairs
        a_score = self._a_score_entailment(gold_as, pred_as, matched_pairs)

        # Calculate time for answer 
        a_time = time.perf_counter() - t1

        # Calulate the composite score
        composite = alpha * q_score + (1 - alpha) * a_score

        # Return the scores
        return {
            "semqa_variation":     variation,
            "semqa_threshold":     threshold,
            "semqa_alpha":         alpha,
            "semqa_q_score":       q_score,
            "semqa_a_score":       a_score,
            "semqa_q_time":        q_time,
            "semqa_a_time":        a_time,
            "semqa_composite":     composite,
        }