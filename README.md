# SemQA



Created a small training dataset with:
```terminal
jq '.[0:200]' data_store/averitec/train.json > data_store/averitec/train_200.json
# Run the HeRO model on the training dataset with 200 examples 
cp -r knowledge_store/train knowledge_store/train_200
```