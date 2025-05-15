#!/usr/bin/env python3
import json
import argparse
import time
import sys
import os
import importlib
import math
import pandas as pd
from averitec_evaluate import (
    AVeriTeCEvaluator,
    EV2REvaluator,
    download_nltk_data
)

def download_resources():
    download_nltk_data('punkt')
    download_nltk_data('punkt_tab')
    download_nltk_data('wordnet')

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def serialize_av_pairs_from_gold(g):
    return "\t\t\n\n".join(
        f"{q['question']}\t\t\n{q['answers'][0]['answer'] if q['answers'] else 'No answer'}"
        for q in g["questions"]
    )

def serialize_av_pairs_from_pred(p):
    return "\t\t\n\n".join(
        f"{ev['question']}\t\t\n{ev['answer']}"
        for ev in p["evidence"]
    )

def compute_hungarian_metrics(g, p, averi):
    """Compute Q-only, Q+A, and end-to-end metrics (with timings)."""
    # wrap into 1-row DataFrames
    df_gold = pd.DataFrame([{
        "claim": g["claim"],
        "evi": serialize_av_pairs_from_gold(g),
        "label": g["label"],
        "id": g["id"],
    }])
    df_pred = pd.DataFrame([{
        "claim": p["claim"],
        "evi": serialize_av_pairs_from_pred(p),
        "label": p["pred_label"],
        "id": p["id"],
    }])
    out = {}
    # Q-only
    t0 = time.perf_counter()
    _, [q] = averi.evaluate_questions_only(df_pred, df_gold)
    out["Q-only (Hungarian)"] = (q, time.perf_counter() - t0)
    # Q+A
    t0 = time.perf_counter()
    _, [qa] = averi.evaluate_questions_and_answers(df_pred, df_gold)
    out["Q+A (Hungarian)"] = (qa, time.perf_counter() - t0)
    # end-to-end
    t0 = time.perf_counter()
    _, [ee] = averi.evaluate_averitec_score(df_pred, df_gold)
    out["AVeriTeC end-to-end"] = (float(ee[0]), time.perf_counter() - t0)
    return out

def batch_ev2r(sub_preds, sub_gold, ev2r_scorer):
    """Prepare and send the batch of EV2R prompts, return q_recalls, qa_recalls, batch timings."""
    df_preds = pd.DataFrame([{
        "claim": p["claim"], "evi": serialize_av_pairs_from_pred(p),
        "label": p["pred_label"], "id": p["id"]
    } for p in sub_preds])
    df_gold  = pd.DataFrame([{
        "claim": g["claim"], "evi": serialize_av_pairs_from_gold(g),
        "label": g["label"], "id": g["id"]
    } for g in sub_gold])

    pred_q, ref_q, pred_qa, ref_qa = ev2r_scorer.prepare_dataset(df_preds, df_gold)

    t0 = time.perf_counter()
    q_resps = ev2r_scorer.prompt_api_model(pred_q, ref_q, input_type="question")
    q_batch_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    qa_resps = ev2r_scorer.prompt_api_model(pred_qa, ref_qa, input_type="qa_pair")
    qa_batch_time = time.perf_counter() - t0

    _, q_recalls  = ev2r_scorer.extract_recall_score(ev2r_scorer.calculate_question_scores(q_resps))
    _, qa_recalls = ev2r_scorer.extract_recall_score(ev2r_scorer.calculate_prediction_scores(qa_resps))

    return q_recalls, qa_recalls, q_batch_time, qa_batch_time

def build_results(sub_preds, sub_gold, averi, ev2r_scorer, q_recalls, qa_recalls, q_batch, qa_batch):
    """Loop over the N examples and build a DataFrame of all metrics."""
    rows = []
    for j, (g, p) in enumerate(zip(sub_gold, sub_preds)):
        assert g["claim"] == p["claim"], "Claim mismatch!"
        hung = compute_hungarian_metrics(g, p, averi)
        q_hung, t_q_hung = hung["Q-only (Hungarian)"]
        qa_hung, t_qa_hung = hung["Q+A (Hungarian)"]
        ee_score, t_ee = hung["AVeriTeC end-to-end"]

        q_rec  = q_recalls[j]
        qa_rec = qa_recalls[j]

        # debug for zero/NaN
        if q_rec == 0 or math.isnan(q_rec):
            print(f"[DEBUG] Example {g['id']}: Q-only recall is zero or NaN")
        if qa_rec == 0 or math.isnan(qa_rec):
            print(f"[DEBUG] Example {g['id']}: Q+A recall is zero or NaN")

        rows.append({
            "id":                   g["id"],
            "claim":                g["claim"],
            "gold_json":            json.dumps(g),
            "pred_json":            json.dumps(p),
            "gold_label":           g["label"],
            "pred_label":           p["pred_label"],
            "gold_serialized":      serialize_av_pairs_from_gold(g),
            "pred_serialized":      serialize_av_pairs_from_pred(p),
            "Q_only_hungarian":     float(q_hung),
            "Q_only_hungarian_time":t_q_hung,
            "QA_hungarian":         float(qa_hung),
            "QA_hungarian_time":    t_qa_hung,
            "AVeriTeC_end_to_end":  float(ee_score),
            "AVeriTeC_time":        t_ee,
            "Ev2R_Q_only_recall":   float(q_rec),
            "Ev2R_Q_only_batch":    q_batch,
            "Ev2R_Q_only_per_ex_time":   q_batch / len(sub_preds),
            "Ev2R_QA_recall":       float(qa_rec),
            "Ev2R_QA_batch":        qa_batch,
            "Ev2R_QA_per_ex_time":       qa_batch / len(sub_preds),
        })
    return pd.DataFrame(rows)


# Constants 
DATASET_SPLIT = "train_200"
ROOT_PATH = "/cluster/work/projects/ec403/ec-kjetiki/ExamIN5550/"
DATASTORE_PATH = "/cluster/work/projects/ec403/ec-kjetiki/ExamIN5550/data_store"
BASELINE_PREDICTION_PATH = f"{DATASTORE_PATH}/baseline/{DATASET_SPLIT}_veracity_prediction.json"
GOLD_PREDICTION_PATH = f"{DATASTORE_PATH}/averitec/{DATASET_SPLIT}.json"


def main():
    p = argparse.ArgumentParser(description="Per‐example evaluation")
    p.add_argument("--prediction_file", default=BASELINE_PREDICTION_PATH)
    p.add_argument("--gold_file",       default=GOLD_PREDICTION_PATH)
    p.add_argument("--examples", type=int, default=200)
    p.add_argument("--output_csv",      default="train_200.csv")
    args = p.parse_args()

    download_resources()

    print("Loading evaluators…")
    averi     = AVeriTeCEvaluator()
    sys.path.append(os.path.dirname("properties.py"))
    ev2r_scorer = EV2REvaluator(importlib.import_module("properties"))

    print("Reading data…")
    preds = load_json(args.prediction_file)
    gold  = load_json(args.gold_file)
    assert len(preds) == len(gold)

    # assign IDs
    for i, (p_i, g_i) in enumerate(zip(preds, gold)):
        p_i["id"] = g_i["id"] = i

    N = min(args.examples, len(preds))
    sub_preds = preds[:N]
    sub_gold  = gold[:N]

    print(f"Batching Ev2R over first {N} examples…")
    q_recalls, qa_recalls, q_batch, qa_batch = batch_ev2r(sub_preds, sub_gold, ev2r_scorer)

    print(f"Building result DataFrame…")
    df_results = build_results(
        sub_preds, sub_gold,
        averi, ev2r_scorer,
        q_recalls, qa_recalls,
        q_batch, qa_batch
    )

    print(f"Writing out {args.output_csv}")
    df_results.to_csv(args.output_csv, index=False)
    print(df_results.head())

if __name__ == "__main__":
    main()
