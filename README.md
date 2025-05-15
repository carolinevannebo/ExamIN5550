# SemQA



Created a small training dataset with:
```terminal
jq '.[0:200]' data_store/averitec/train.json > data_store/averitec/train_200.json
#  Donwload datastore 
cp -r knowledge_store/train knowledge_store/train_200
```

Generate the metrics for the subsable (without SemQA):
```terminal
# Add huggingface token
HUGGING_FACE_HUB_TOKEN="" sbatch generate.slurm --output_csv output.csv
```

Create a dataframe with the appended SemQA scores (the genearted output csv file will be names `output_semqa_<TIME>.csv`): 
```terminal
python add_semqa.py --input_csv output.csv
```


For finetuning the metric against other metrics: 
```
sbatch finetune_semqa.slurm --top_k 4 --threshold 0.5 --input_csv output.csv 
```