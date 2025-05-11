#!/usr/bin/env python3
import json
import argparse
import random
import time
import sys
import os
import importlib

import pandas as pd
from averitec_evaluate import (
    AVeriTeCEvaluator,
    EV2REvaluator,
    download_nltk_data
)

def serialize_av_pairs_from_gold(gold_example):
    lines = []
    for q in gold_example["questions"]:
        ans = q["answers"][0]["answer"] if q["answers"] else "No answer"
        lines.append(f"{q['question']}\t\t\n{ans}")
    return "\t\t\n\n".join(lines)

def serialize_av_pairs_from_pred(pred_example):
    lines = []
    for ev in pred_example["evidence"]:
        lines.append(f"{ev['question']}\t\t\n{ev['answer']}")
    return "\t\t\n\n".join(lines)

def compute_av_metrics_for_example(g, p, averi, times):
    # wrap into 1-row DataFrames
    serial_gold = {
        "claim": g["claim"],
        "evi": serialize_av_pairs_from_gold(g),
        "label": g["label"],
        "id": g["id"],
    }
    serial_pred = {
        "claim": p["claim"],
        "evi": serialize_av_pairs_from_pred(p),
        "label": p["pred_label"],
        "id": p["id"],
    }
    df_gold = pd.DataFrame([serial_gold])
    df_pred = pd.DataFrame([serial_pred])

    out = {}
    # Q-only
    start = time.perf_counter()
    _, [q] = averi.evaluate_questions_only(df_pred, df_gold)
    t = time.perf_counter() - start
    out["Q-only (Hungarian)"] = (q, t)
    times.setdefault("Q-only (Hungarian)", []).append(t)

    # Q+A
    start = time.perf_counter()
    _, [qa] = averi.evaluate_questions_and_answers(df_pred, df_gold)
    t = time.perf_counter() - start
    out["Q+A (Hungarian)"] = (qa, t)
    times.setdefault("Q+A (Hungarian)", []).append(t)

    # end-to-end
    start = time.perf_counter()
    _, [ee] = averi.evaluate_averitec_score(df_pred, df_gold)
    t = time.perf_counter() - start
    out["AVeriTeC end-to-end"] = (ee, t)
    times.setdefault("AVeriTeC end-to-end", []).append(t)

    return out

def check_claim(g, p):
    assert g["claim"] == p["claim"], f"Claim mismatch:\n GOLD: {g['claim']}\n PRED: {p['claim']}"


# Constants 
DATASET_SPLIT = "dev_super_small"
ROOT_PATH = "/cluster/work/projects/ec403/ec-kjetiki/ExamIN5550/"
DATASTORE_PATH = "/cluster/work/projects/ec403/ec-kjetiki/ExamIN5550/data_store"
BASELINE_PREDICTION_PATH = f"{DATASTORE_PATH}/baseline/{DATASET_SPLIT}_veracity_prediction.json"
GOLD_PREDICTION_PATH = f"{DATASTORE_PATH}/averitec/{DATASET_SPLIT}.json"

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


    # assign ids
    for i,(p,g) in enumerate(zip(preds,gold)):
        p["id"]=i; g["id"]=i

    # sample
    idxs = random.sample(range(len(preds)), k=min(args.examples, len(preds)))

    # build subset DataFrames
    sub_preds = [preds[i] for i in idxs]
    sub_gold  = [gold[i]  for i in idxs]
    df_preds = pd.DataFrame([{
        "claim": p["claim"], "evi": serialize_av_pairs_from_pred(p),
        "label": p["pred_label"], "id": p["id"]
    } for p in sub_preds])
    df_gold  = pd.DataFrame([{
        "claim": g["claim"], "evi": serialize_av_pairs_from_gold(g),
        "label": g["label"], "id": g["id"]
    } for g in sub_gold])

    # init scorers
    averi = AVeriTeCEvaluator()
    sys.path.append(os.path.dirname("properties.py"))
    props = importlib.import_module("properties")
    ev2r  = EV2REvaluator(props)

    # batch Ev2R
    print("Batching EV2R Q-only prompts…")
    pred_q, ref_q, pred_qa, ref_qa = ev2r.prepare_dataset(df_preds, df_gold)
    t0 = time.perf_counter()
    q_resps  = ev2r.prompt_api_model(pred_q,  ref_q,  input_type="question")
    t_q_batch = time.perf_counter()-t0

    print("Batching EV2R Q+A prompts…")
    t0 = time.perf_counter()
    qa_resps = ev2r.prompt_api_model(pred_qa, ref_qa, input_type="qa_pair")
    t_qa_batch = time.perf_counter()-t0

    # extract recalls
    _, q_recalls  = ev2r.extract_recall_score(ev2r.calculate_question_scores(q_resps))
    _, qa_recalls = ev2r.extract_recall_score(ev2r.calculate_prediction_scores(qa_resps))

    # track AVeri times
    av_times = {}

    # per‐example reporting
    for j, idx in enumerate(idxs):
        g = gold[idx]; p = preds[idx]
        check_claim(g,p)
        print("\n" + "="*10 + f" Example {idx} " + "="*10)
        print("CLAIM:", g["claim"])
        print(f" Gold QA pairs: {len(g['questions'])}")
        print(f" Pred QA pairs: {len(p['evidence'])}\n")

        av_res = compute_av_metrics_for_example(g,p, averi, av_times)
        for name,(val,t) in av_res.items():
            print(f"  {name:25s} → {val:.4f}   ({t:.3f}s)")

        print(f"  {'EV2R Q-only recall':25s} → {q_recalls[j]:.4f}   (batched in {t_q_batch:.2f}s)")
        print(f"  {'EV2R Q+A recall':25s} → {qa_recalls[j]:.4f}   (batched in {t_qa_batch:.2f}s)")

    # average AVeri times
    print("\nAverage AVeriTeC times:")
    for name,tlist in av_times.items():
        print(f"  {name:25s} : {sum(tlist)/len(tlist):.3f}s over {len(tlist)} examples")

if __name__=="__main__":
    main()
