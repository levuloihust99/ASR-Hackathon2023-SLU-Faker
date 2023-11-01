import torch
import torch.nn as nn
import argparse
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from pyctcdecode import build_ctcdecoder
import soundfile as sf
from statistics import mean
import math
import re

def get_decoder_ngram_model(tokenizer, ngram_lm_path, remove_s_in_vocab=True):
    vocab_dict = tokenizer.get_vocab()
    sort_vocab = sorted((value, key) for (key, value) in vocab_dict.items())
    if remove_s_in_vocab: # remove 2 tokens: <s> and </s>
        vocab = [x[1] for x in sort_vocab][:-2]
    else:
        vocab = [x[1] for x in sort_vocab]
    vocab_list = vocab
    vocab_list[tokenizer.word_delimiter_token_id] = " "

    decoder = build_ctcdecoder(
        vocab_list,
        kenlm_model_path=ngram_lm_path,  # either .arpa or .bin file
        alpha=0.5,  # tuned on a val set
        beta=1.0,  # tuned on a val set
    )

    return decoder

class CMCWav2vec(nn.Module):

    def __init__(self, last_checkpoint, model_type='large', device='cpu'):
        super().__init__()
        self.processor = Preprocessor(last_checkpoint,model_type, device)
        self.classifier = Wav2Vec2ForCTC.from_pretrained(last_checkpoint).to(device)
        self.classifier.eval()
        self.device = device
    def forward(self, x):
        
        input_values = self.processor(x)
        
        logits = self.classifier(input_values.to(self.device)).logits
        
        return logits, self.processor.decode(logits)

class Preprocessor(nn.Module):

    def __init__(self, last_checkpoint,model_type='large', device='cpu'):
        super().__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(last_checkpoint)
        self.device = device


    def decode(self, logits):
        argmax_prediction = self.processor.batch_decode(torch.argmax(logits, dim=-1))
        return argmax_prediction[0]

    def forward(self, x):
        input_values = self.processor(x, sampling_rate=16000, return_tensors='pt', padding='longest').input_values.to(self.device)
        return input_values
    

def infer(audio_path,lm_model,pytorch_model):
    audio,sr = sf.read(audio_path)
    end_in_time_step_window = math.floor((audio.shape[0] / sr)/0.02)
    
    logits = pytorch_model(audio)[0]
    
    softmax_pred = torch.nn.functional.softmax(logits, dim=-1)
    max_elements, _ = torch.max(softmax_pred, dim=-1)
    max_elements = max_elements.tolist()
    pred_ids = torch.argmax(logits, dim=-1)
    logits = logits.cpu().detach().numpy()
    
    beam_decode_raw = lm_model.decode_beams(logits[0],
                                            beam_width=500,
                                            hotwords=['rưỡi'], #,'ngắt','khang'],
                                            hotword_weight=10.0
                                            )[0]
    
    wav2vec_results = []
    for word_index, (word, (start, end)) in enumerate(beam_decode_raw[2]):
        end = end if end <= end_in_time_step_window else end_in_time_step_window
        confident_score_list = max_elements[0][start:end]
        confident_score = mean(confident_score_list) if len(confident_score_list) > 0 else 0.0
        output = ""
        if confident_score > 0.2:
            # wav2vec_results.append({'word':word,'start_time':str(datetime.timedelta(seconds=round(start*0.02,2))),'end_time':str(datetime.timedelta(seconds=round(end*0.02,2)))})
            wav2vec_results.append(word)
        output = re.sub(r'\s+', ' ', ' '.join([content for content in wav2vec_results])).strip()
    return output



