#!/usr/bin/env python3
import json
import argparse
import random
import time
import sys
import os
import importlib
from semqa import SemQA
import numbers
import numpy as np


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

def compute_av_metrics_for_example(g, p, averi, semqa_scorer, times):
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

    # Output dict for storing results
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

    # SemQA
    gold_qs, gold_as, pred_qs, pred_as = semqa_scorer.prepare_dataset(g, p)
    # print("\n\nGold Qs:", gold_qs)
    # print("Gold As:", gold_as)
    # print("Pred Qs:", pred_qs)
    # print("Pred As:", pred_as, "\n\n")


    print("SemQA scoring...")
    res_dict = semqa_scorer.score(
        gold_qs, pred_qs, gold_as, pred_as
    )

    # Extend out with res_dict
    for k, v in res_dict.items():
        out[k] = v

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

    # Create SemQA scorer
    print("Creating SemQA scorer")
    semqa_scorer = SemQA(cache_dir=DATASTORE_PATH)

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


    # batch Ev2R
    print("Batching EV2R Q-only prompts…")
    pred_q, ref_q, pred_qa, ref_qa = ev2r_scorer.prepare_dataset(df_preds, df_gold)
    print(f"  {len(pred_q)} Q-only pairs")
    print(f"  {len(pred_qa)} Q+A pairs")
    start = time.perf_counter()
    q_resps  = ev2r_scorer.prompt_api_model(pred_q,  ref_q,  input_type="question")
    t_q_batch = time.perf_counter()- start
 
    print("Batching EV2R Q+A prompts…")
    start = time.perf_counter()
    qa_resps = ev2r_scorer.prompt_api_model(pred_qa, ref_qa, input_type="qa_pair")
    t_qa_batch = time.perf_counter() - start
 
    # extract recalls
    _ , q_recalls  = ev2r_scorer.extract_recall_score(ev2r_scorer.calculate_question_scores(q_resps))
    _, qa_recalls = ev2r_scorer.extract_recall_score(ev2r_scorer.calculate_prediction_scores(qa_resps))

    # track AVeri times
    av_times = {}


    semqa_times = {}

    # per‐example reporting
    for j, idx in enumerate(idxs):
        g = gold[idx]; p = preds[idx]
        check_claim(g,p)
        print("\n" + "="*10 + f" Example {idx} " + "="*10)
        print("CLAIM:", g["claim"])
        print(f" Gold QA pairs: {len(g['questions'])}")
        print(f" Pred QA pairs: {len(p['evidence'])}\n")
        print("GOLD:")
        print(g, "\n")
        print("GOLD Serialized:\n")
        print(serialize_av_pairs_from_gold(g), "\n")
        print("PRED:")
        print(p, "\n")
        print("PRED Serialized:\n")
        print(serialize_av_pairs_from_pred(p), "\n")

        av_res = compute_av_metrics_for_example(g, p, averitec_scorer, semqa_scorer,  av_times)
        for k,v in av_res.items():
            # Dont print the elapsed times for semqa
            if k in ["Q_recall_elapsed", "A_entail_elapsed"]:
                continue


            # Value can be dict, float or list depending on the metric
            # Should print the key value pair wihout raising an error 
            if ( isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], numbers.Number)):
                # Data is a 2‐tuple: (score, elapsed)
                raw_score, elapsed = v

                # Scalar numeric score
                if isinstance(raw_score, numbers.Number):
                    score_str = f"{float(raw_score):.4f}"

                # NumPy scalar 
                elif isinstance(raw_score, np.generic):
                    score_str = f"{float(raw_score):.4f}"

                # Everything else (list, dict, etc.) → just str()
                else:
                    score_str = str(raw_score)

                print(f"  {k:25s} => {score_str}   (took {elapsed:.2f}s)")
            elif isinstance(v, float) or isinstance(v, numbers.Number):
                print(f"  {k:25s} => {float(v):.4f}")
            elif isinstance(v, list):
                print(f"  {k:25s} => {v}")
            elif isinstance(v, str):
                print(f"  {k:25s} => {v}")
            else:
                # Should not happen
                print(f"  {k:25s} => {v}")

        # SemQA compute times
        semqa_times.setdefault("Q-recall", []).append(av_res["Q_recall_elapsed"])
        semqa_times.setdefault("A-entail", []).append(av_res["A_entail_elapsed"])

        print(f"  {'Q-recall elapsed':25s} => {av_res['Q_recall_elapsed']:.4f}s")
        print(f"  {'A-entail elapsed':25s} => {av_res['A_entail_elapsed']:.4f}s")
        print(f"  {'EV2R Q-only recall':25s} => {q_recalls[j]:.4f}   (batched in {t_q_batch:.2f}s, time per example: {t_q_batch/len(pred_q):.2f}s)")
        print(f"  {'EV2R Q+A recall':25s} => {qa_recalls[j]:.4f}   (batched in {t_qa_batch:.2f}s, time per example: {t_qa_batch/len(pred_qa):.2f}s)")

    # average AVeri times
    print("\nAverage AVeriTeC times:")
    for name,tlist in av_times.items():
        print(f"  {name:25s} : {sum(tlist)/len(tlist):.3f}s over {len(tlist)} examples")

if __name__=="__main__":
    main()
