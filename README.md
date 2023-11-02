# Setup
## Prerequisites
- Ubuntu 20.04
- Python3.8

## Create virtual environment
```shell
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

# Preprocess audio
## Prerequisites
* Install [FFmpeg](https://www.ffmpeg.org/)
* A folder `original_data/private_test` (available [HERE](https://drive.google.com/drive/folders/1NAceqPmkG-M-2NPDbhxbRTD9DKr8hTHu?usp=drive_link))
## Run preprocess
```shell
python preprocess_audio.py --mode private
```
Running this command generates a folder `private_test_16k_1ac_vad`. We also provide this folder [HERE](https://drive.google.com/drive/folders/1NAceqPmkG-M-2NPDbhxbRTD9DKr8hTHu?usp=drive_link) in case you just want to use the proprocessed audios without running the preprocessing yourself.
# Inference
## Prerequisites
* A folder `wav2vec_model`
* A folder `ngram_lm`
* A folder `private_test_16k_1ac_vad`

All is available [HERE](https://drive.google.com/drive/folders/1NAceqPmkG-M-2NPDbhxbRTD9DKr8hTHu?usp=drive_link)

## Run inference
```shell
cd infer
python infer.py
```

Running this command generates a file `results.json`, which is the input for NER and intent classification.