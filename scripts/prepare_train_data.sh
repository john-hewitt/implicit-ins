# Downloading same as open-instruct
# check if there is $HF_TOKEN in the environment variables
if [ -z "$HF_TOKEN" ]
then
    echo "Warning: HuggingFace dataset LIMA requires permissive access."
    echo "Warning: Please request the access at https://huggingface.co/datasets/GAIR/lima and set the HF_TOKEN environment variable before running this script."
    exit 1
fi

echo "Downloading Stanford alpaca data..."
wget -P data/raw_train/stanford_alpaca/ https://github.com/tatsu-lab/stanford_alpaca/raw/main/alpaca_data.json


echo "Downloading LIMA dataset..."
wget --header="Authorization: Bearer $HF_TOKEN" -P data/raw_train/lima/ https://huggingface.co/datasets/GAIR/lima/raw/main/train.jsonl

echo "Processing datasets..."
python open_instruct/reformat_datasets.py --raw_data_dir data/raw_train/ --output_dir data/processed/

# Now download and process datasets specific to this repository.
