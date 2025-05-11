#!/usr/bin/env python3
import json
import argparse
from nltk.translate.meteor_score import single_meteor_score
from nltk import word_tokenize
from typing import List, Tuple, Dict
import random
import sys
import os
import importlib

from averitec_evaluate import AVeriTeCEvaluator, EV2REvaluator 

def compute_metrics_for_example():
    pass 


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
    parser.add_argument("--examples", type=int, help="Number of examples to iterate over.", default=2)
    args = parser.parse_args()


    # Create Averitec Scorer
    print("Create AVeritec Evaluator")
    averitec_scorer = AVeriTeCEvaluator()

    # Create Ev2r scorer
    print("Create Ev2R scorer")
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
    

    # 3) Pick random example indices
    max_examples = len(preds)
    n = min(args.examples, max_examples)
    example_idxs = random.sample(range(max_examples), n)

    # 4) Print each example
    for idx in example_idxs:
        print("\n" + "="*30 + f" Example {idx} " + "="*30 + "\n")

        print(">>> GOLD ENTRY:")
        print(json.dumps(gold[idx], indent=2, ensure_ascii=False))

        print("\n>>> PREDICTION ENTRY:")
        print(json.dumps(preds[idx], indent=2, ensure_ascii=False))

    print("\nDone.")


        


if __name__ == "__main__":
    main()
