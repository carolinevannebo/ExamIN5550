# SemQA

SemQA is a semantic qustion-answer metric for evaluating the generated evidence against gold evidence. 

This repository is based on the FEVER 2025 workshop. 
Link to the FEVER 2025 workshop repository: https://github.com/Raldir/FEVER-8-Shared-Task 

## Usage

Load the modules: 
```terminal
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/11.7.0
```

Create a python virtual environment. In the environment install the following: 
```terminal
python3 -m pip install torch
python3 -m pip install -r requirements.txt
MAX_JOBS=8 python3 -m pip -v install flash-attn --no-build-isolation
python3 -m pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/nightly/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
python3 -m pip install transformers --upgrade
```


Donload the data with the download script: 
```terminal
./download_data.sh
```

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

For finetuning the metric against other metrics: 
```
sbatch finetune_semqa.slurm --top_k 4 --threshold 0.5 --input_csv output.csv 
```

Create a dataframe with the appended SemQA scores (the genearted output csv file will be names `output_semqa_<TIME>.csv`): 
```terminal
python add_semqa.py --input_csv output.csv
```
