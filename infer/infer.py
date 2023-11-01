from model import Preprocessor, get_decoder_ngram_model, CMCWav2vec, infer
import torch
import glob
from decouple import config
from normalize_text import find_all_text_number, word_2_number
import os
import json


def load_all_model(lm_path,last_checkpoint):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocessor = Preprocessor(last_checkpoint,model_type='large', device=device)
    lm_model = get_decoder_ngram_model(preprocessor.processor.tokenizer, ngram_lm_path=lm_path)
    pytorch_model = CMCWav2vec(last_checkpoint, 'large', device=device)
    return lm_model,pytorch_model,device

if __name__ == '__main__':
    lm_model,pytorch_model,device = load_all_model(os.path.join("../",config('LM_PATH')),os.path.join("../",config('WAV2VEC_MODEL')))
    audio_paths = glob.glob(os.path.join("../",f"{config('PRIVATE_TEST_PATH')}/*"))
    with open("results.json",'w',encoding='utf-8') as g:
        results = []
        for audio_path in audio_paths:
            result = {}
            raw = infer(audio_path,lm_model,pytorch_model)
            result['file_name'] = os.path.basename(audio_path)
            result['raw'] = raw
            
            raw = raw.replace("mười mười","mười")
            print(f"raw: {raw}")
            all_numbers = find_all_text_number(raw)
            
            try:
                norm = word_2_number(all_numbers,raw)
                result['norm'] = norm
            except:
                result['norm'] = raw
            print(f"norm: {norm}")
            results.append(result)
        json.dump(results, g, ensure_ascii=False, indent=4)
        
