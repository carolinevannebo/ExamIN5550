#!/usr/bin/env python3
import json
import time
import itertools
import argparse
import pandas as pd
from semqa import SemQA
from itertools import islice
import random

# the list of other metrics correlate against:
PRE_CALC_METRICS = [
    "Q_only_hungarian",
    "QA_hungarian",
    "AVeriTeC_end_to_end",
    "Ev2R_Q_only_recall",
    "Ev2R_QA_recall",
]

def run_config(df, semqa, alpha, threshold, top_k, variation):
    """
    Given a DataFrame and a SemQA config, compute SemQA scores
    and return a new DataFrame with a 'semqa_score' column.
    """
    scores = []
    for _, row in df.iterrows():
        gold = json.loads(row["gold_json"])
        pred = json.loads(row["pred_json"])

        # prepare and score
        gold_qs, gold_as, pred_qs, pred_as = semqa.prepare_dataset(gold, pred)
        out = semqa.score(
            gold_qs=gold_qs, pred_qs=pred_qs,
            gold_as=gold_as, pred_as=pred_as,
            alpha=alpha, variation=variation,
            threshold=threshold, top_k=top_k
        )

        # get the composite score
        scores.append(out["semqa_composite"])
    df2 = df.copy()
    df2["semqa_score"] = scores
    return df2

def summarize_correlations(df):
    """
    Compute Pearson and Spearman correlations between semqa_score
    and each metric in PRE_CALC_METRICS.
    """
    results = []
    for metric in PRE_CALC_METRICS:
        cov = df["semqa_score"].cov(df[metric])
        pear = df["semqa_score"].corr(df[metric], method="pearson")
        kendall = df["semqa_score"].corr(df[metric], method="kendall")
        results.append({
            "metric": metric,
            "cov": cov,
            "pearson": pear,
            "kendall": kendall,
        })
    return pd.DataFrame(results)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", default="train_200.csv")
    p.add_argument("--alphas",     nargs="+", type=float, default=[0.7, 0.8, 0.9, 1.0])
    p.add_argument("--thresholds", nargs="+", type=float, default=[0.1, 0.2, 0.3])
    p.add_argument("--top_ks",     nargs="+", type=int,   default=[None, 1, 3,  5])
    p.add_argument("--variations", nargs="+", type=str,   default=["hungarian","softmax"])
    p.add_argument("--max_configs", type=int, default=40, help="maximum number of hyperparameter combinations to run")
    args = p.parse_args()
    

    # Read the input CSV file
    print(f"Reading input CSV file: {args.input_csv}")
    df0 = pd.read_csv(args.input_csv, sep="\t")
    print("Loaded DataFrame with", len(df0), "rows")


    # Initialize SemQA
    print("Initializing SemQA")
    semqa = SemQA()

    # Dataset with precomputed metrics
    all_combos = list(itertools.product(
        args.alphas, args.thresholds, args.top_ks, args.variations
    ))
    random.shuffle(all_combos)
    combos_to_run = all_combos[: args.max_configs]

    print("Running SemQA grid seatch with maxum of", args.max_configs, "configs")


    # Run the grid search
    for alpha, thr, k, var in combos_to_run:
        cfg_name = f"alpha={alpha}, theshold={thr}, k={k}, variation={var}"
        print("\n" + "="*len(cfg_name))
        print(cfg_name)
        print("="*len(cfg_name))

        t0 = time.time()
        print("Running SemQA with config:", cfg_name)
        print("  - alpha:", alpha)
        print("  - threshold:", thr)
        print("  - top_k:", k)
        print("  - variation:", var)
        df_scored = run_config(df0, semqa, alpha, thr, k, var)

        print("Computing correlations")
        df_corr   = summarize_correlations(df_scored)
        elapsed   = time.time() - t0

        # print a nice table
        print(df_corr.to_string(index=False, 
            formatters={
                "cov":     "{:0.3f}".format,
                "pearson":  "{:0.3f}".format,
                "kendall":"{:0.3f}".format
            }))


        print(f"done in {elapsed:.1f}s")
        print("=" * 80)

if __name__ == "__main__":
    main()
