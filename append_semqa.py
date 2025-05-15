#!/usr/bin/env python3
import json
import argparse
import sys
import os
import importlib
import math
import pandas as pd
from semqa import SemQA



# Constants 
DATASET_SPLIT = "dev_super_small"
ROOT_PATH = "/cluster/work/projects/ec403/ec-kjetiki/ExamIN5550/"
INPUT_PATH = f"{ROOT_PATH}output.csv"


def main():
    p = argparse.ArgumentParser(description="Per‐example evaluation")
    p.add_argument("--input_csv", default=INPUT_PATH)
    p.add_argument("--alpha", type=float, default=0.8)
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=8)
    p.add_argument("--variation", type=str, default="hungarian")
    args = p.parse_args()

    # Read the input CSV file
    print(f"Reading input CSV file: {args.input_csv}")
    df = pd.read_csv(args.input_csv, sep="\t")
    print("Loaded DataFrame with", len(df), "rows")



    # Print all columns in the DataFrame
    print("Columns in the DataFrame:")
    for col in df.columns:
        print(col)


    # Create SemQA instance
    print("Creating SemQA instance with alpha:", args.alpha)
    semqa_scorer = SemQA()

    # Compute scores for each example in the dataframe 
    print("Computing scores for each example in the DataFrame")
    for index, row in df.iterrows():
        # Log every 20 rows
        if index % 20 == 0:
            print(f"\tRow {index}/{len(df)}")


        # Extract the gold and pred data from the DataFrame
        gold = row['gold']
        pred = row['pred']

        # Convert the JSON strings to dictionaries
        gold_dict = json.loads(gold)
        pred_dict = json.loads(pred)

        # Prepare the dataset
        gold_qs, gold_as, pred_qs, pred_as = semqa_scorer.prepare_dataset(gold_dict, pred_dict)

        # Compute the scores
        output_dict = semqa_scorer.compute_scores(gold_qs, gold_as, pred_qs, pred_as, alpha=args.alpha, threshold=args.threshold, top_k=args.top_k, variation=args.variation)

        # Add each item in the output_dict to the DataFrame
        for key, value in output_dict.items():
            # Do not add the following keys to the DataFrame: 
            if key in ["semqa_variation", "semqa_threshold", "semqa_top_k", "semqa_alpha"]:
                continue

            # Add the key value pait to the dataframe at given index
            if key not in df.columns:
                df[key] = None
            df.at[index, key] = value

    # Log that the results have been computed
    print("Finished processing all rows.")
    

    # Create the output dataframe path 
    output_frame_path = args.input_csv.replace(".csv", f"_semqa.csv")


    # Save the DataFrame to a new CSV file
    print(f"Writing out {output_frame_path}")
    df.to_csv(output_frame_path, index=False, sep="\t")
    print("Output saved successfully.")
    print("Length of results DataFrame:", len(df))
    print("Column names:", df.columns)
    print("First few rows of the DataFrame:")
    print(df.head())

if __name__ == "__main__":
    main()
