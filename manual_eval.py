#!/usr/bin/env python3
import json
import argparse
from nltk.translate.meteor_score import single_meteor_score
from nltk import word_tokenize
import nltk
import time

from typing import List, Tuple, Dict
import random
import sys
import os
import importlib
import pandas as pd

from averitec_evaluate import AVeriTeCEvaluator, EV2REvaluator, download_nltk_data


def serialize_av_pairs_from_gold(gold_example):
    """
    Turns gold_example['questions'] into the single-string format
    AVeriTeCEvaluator expects in a column named 'evi'.
    """
    lines = []
    for q in gold_example["questions"]:
        question = q["question"]
        # take first answer if multiple
        answer = q["answers"][0]["answer"] if q["answers"] else "No answer"
        lines.append(f"{question}\t\t\n{answer}")
    return "\t\t\n\n".join(lines)

def serialize_av_pairs_from_pred(pred_example):
    """
    Turns pred_example['evidence'] into the same single-string format.
    """
    lines = []
    for ev in pred_example["evidence"]:
        lines.append(f"{ev['question']}\t\t\n{ev['answer']}")
    return "\t\t\n\n".join(lines)

def compute_metrics_for_example(gold_ex, pred_ex, averi_scorer, ev2r_scorer, times):
    # One level dataframes 
    serial_gold = {
        "claim": gold_ex["claim"],
        "evi":    serialize_av_pairs_from_gold(gold_ex),
        "label":  gold_ex["label"],
        "id":     gold_ex["id"],
    }
    serial_pred = {
        "claim": pred_ex["claim"],
        "evi":    serialize_av_pairs_from_pred(pred_ex),
        "label":  pred_ex["pred_label"],
        "id":     pred_ex["id"],
    }
    df_gold = pd.DataFrame([serial_gold])
    df_pred = pd.DataFrame([serial_pred])


    # Output dir
    out = {}

    #  AVeriTeC Q-only (Hungarian-Meteor)
    print("\tCalculating Q-only (Hungarian-Meteor)…")
    start = time.perf_counter()
    q_only_score, [q_only_list] = averi_scorer.evaluate_questions_only(df_pred, df_gold)
    elapsed = time.perf_counter() - start
    out["Q-only (Hungarian-Meteor)"] = (q_only_list, elapsed)
    print(f"\tQ-only took {elapsed:.4f}s, score = {q_only_list:.4f}")

    # AVeriTeC Q+A (Hungarian-Meteor)
    print("\tCalculating Q+A (Hungarian-Meteor)…")
    start = time.perf_counter()
    qa_score, [qa_list] = averi_scorer.evaluate_questions_and_answers(df_pred, df_gold)
    elapsed = time.perf_counter() - start
    out["Q+A (Hungarian-Meteor)"] = (qa_list, elapsed)
    print(f"\tQ+A took {elapsed:.4f}s, score = {qa_list:.4f}")

    # AVeriTeC end-to-end (label+evi)
    print("\tCalculating AVeriTeC end-to-end…")
    start = time.perf_counter()
    ave_score, [ave_list] = averi_scorer.evaluate_averitec_score(df_pred, df_gold)
    elapsed = time.perf_counter() - start
    out["AVeriTeC end-to-end"] = (ave_list, elapsed)
    print(f"\tEnd-to-end took {elapsed:.4f}s, score = {ave_list:.4f}")

    #  EV2R Q-only recall
    print("\tCalculating EV2R Q-only recall…")
    start = time.perf_counter()
    pred_q, ref_q, pred_qa, ref_qa = ev2r_scorer.prepare_dataset(df_pred, df_gold)
    q_resps = ev2r_scorer.prompt_api_model(pred_q, ref_q, input_type="question")
    q_scores = ev2r_scorer.calculate_question_scores(q_resps)
    ev2r_q_recall, [q_recalls] = ev2r_scorer.extract_recall_score(q_scores)
    elapsed = time.perf_counter() - start
    out["EV2R Q-only recall"] = (q_recalls, elapsed)
    print(f"\t→ EV2R Q-only took {elapsed:.4f}s, score = {q_recalls:.4f}")

    #  EV2R Q+A recall
    print("\tCalculating EV2R Q+A recall…")
    start = time.perf_counter()
    print("\t\tPrompting model")
    qa_resps = ev2r_scorer.prompt_api_model(pred_qa, ref_qa, input_type="qa_pair")
    print("\t\tCalculating score")
    qa_scores = ev2r_scorer.calculate_prediction_scores(qa_resps)
    ev2r_qa_recall, [qa_recalls] = ev2r_scorer.extract_recall_score(qa_scores)
    elapsed = time.perf_counter() - start
    out["EV2R Q+A recall"] = (qa_recalls, elapsed)
    print(f"\tEV2R Q+A took {elapsed:.4f}s, score = {qa_recalls:.4f}")


    # Set times dict for calculating averages
    # Will append the new time, or create a new dict for key
    for k, (_, t) in results.items():
        times.setdefault(k, []).append(t)

    return out


def check_claim(current_gold, current_pred):
    # Check that both claims are identical
    gold_claim = current_gold["claim"]
    pred_claim = current_pred["claim"]
    assert gold_claim == pred_claim , f"Gold claim was: {gold_claim}. Pred claim was: {pred_claim}"


# Constants 
DATASET_SPLIT = "dev_super_small"
ROOT_PATH = "/cluster/work/projects/ec403/ec-kjetiki/ExamIN5550/"
DATASTORE_PATH = "/cluster/work/projects/ec403/ec-kjetiki/ExamIN5550/data_store"
BASELINE_PREDICTION_PATH = f"{DATASTORE_PATH}/baseline/{DATASET_SPLIT}_veracity_prediction.json"
GOLD_PREDICTION_PATH = f"{DATASTORE_PATH}/averitec/{DATASET_SPLIT}.json"


# Main file 
def main():
    parser = argparse.ArgumentParser(description="Manual per-example evaluation")
    parser.add_argument("--prediction_file", type=str, help="JSON file with a list of examples to evaluate.", default=BASELINE_PREDICTION_PATH)
    parser.add_argument("--gold_file", type=str, help="JSON file with a list of examples to evaluate.", default=GOLD_PREDICTION_PATH)
    parser.add_argument("--examples", type=int, help="Number of examples to iterate over.", default=5)
    args = parser.parse_args()


    # Download nltk data
    print("Downloading nltk data")
    download_nltk_data('punkt')
    download_nltk_data('punkt_tab')
    download_nltk_data('wordnet')


    # Create Averitec Scorer
    print("Creating AVeritec Evaluator")
    averitec_scorer = AVeriTeCEvaluator()

    # Create Ev2r scorer
    print("Creating Ev2R scorer")
    sys.path.append(os.path.dirname("properties.py"))
    properties = importlib.import_module("properties")
    ev2r_scorer = EV2REvaluator(properties)


    # Load prediciton file
    print("Reading prediction file")
    with open(args.prediction_file, "r") as f:
        preds = json.load(f)


    # Read the goal path
    print("Reading gold file")
    with open(args.gold_file, "r") as f:
        gold = json.load(f)

    print(f"Loaded {len(preds)} predictions, {len(gold)} gold entries.\n")
    assert len(preds) == len(gold), "There was not enough gold for all predicitons"
    

    # Pick random example indices
    max_examples = len(preds)
    n = min(args.examples, max_examples)
    example_idxs = random.sample(range(max_examples), n)

    # Keep track metric calculation time
    times = {}


    # Print each example
    for idx in example_idxs:
        print("\n" + "="*30 + f" Example {idx} " + "="*30 + "\n")
        
        # Get the current example
        current_gold = gold[idx]
        current_gold["id"] = idx
        current_pred = preds[idx]
        current_pred["id"] = idx

        # Check that both claims are identical
        check_claim(current_gold, current_pred)


        # Print the claim 
        claim = current_gold["claim"]
        print(f"CLAIM : '{claim}'\n")

        # Print the number of question-answer pairs 
        num_gold_questions = len(current_gold["questions"])
        num_pred_questions = len(current_pred["evidence"])

        print(f"Gold QA pairs: {num_gold_questions}, Predicted QA pairs: {num_pred_questions}")


        # Compute scores
        print("Computing for example")
        metrics = compute_metrics_for_example(current_gold, current_pred, averitec_scorer, ev2r_scorer, times)

        print("Computing done")
        
        # Print the values 
        for name, val_list in metrics.items():
            print(f"{name:25s} => {val_list[0]:.4f}")

    print("\nDone.")


    # Calculate average times per metric, and print them 
    print("\nAverage computation times per metric:")
    for name, tlist in times.items():
        avg = sum(tlist) / len(tlist)
        print(f"  {name:20s} : {avg:.3f}s over {len(tlist)} examples")


        


if __name__ == "__main__":
    main()
