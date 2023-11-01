from model import Preprocessor, get_decoder_ngram_model, CMCWav2vec, infer
import torch
import glob
from decouple import config
from normalize_text_model import ITN
import os
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration


def load_all_model(lm_path,last_checkpoint):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocessor = Preprocessor(last_checkpoint,model_type='large', device=device)
    lm_model = get_decoder_ngram_model(preprocessor.processor.tokenizer, ngram_lm_path=lm_path)
    pytorch_model = CMCWav2vec(last_checkpoint, 'large', device=device)
    return lm_model,pytorch_model,device

def correct_spelling(correct,tolenizer,s,device):
    input_ids = tokenizer(f"grammar: {s}", return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = correct.generate(input_ids)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == '__main__':
    lm_model,pytorch_model,device = load_all_model(os.path.join("../",config('LM_PATH')),os.path.join("../",config('WAV2VEC_MODEL')))
    itn = ITN(model_root=os.path.join("../",config('W2T_PATH')), gpu_id=0)
    audio_paths = glob.glob(os.path.join("../",f"{config('PRIVATE_TEST_PATH')}/*"))
    device = "cuda:0"
    correct = T5ForConditionalGeneration.from_pretrained(os.path.join("../",config('CORRECT_PATH')))
    
    correct.eval()
    correct.to(device)
    tokenizer = T5Tokenizer.from_pretrained("VietAI/vit5-base")
    with open("results_private_two_norm_model_12000.json",'w',encoding='utf-8') as g:
        results = []
        for audio_path in audio_paths:
            result = {}
            raw = infer(audio_path,lm_model,pytorch_model)
            result['file_name'] = os.path.basename(audio_path)
            result['raw'] = raw
            correct_raw = correct_spelling(correct,tokenizer,raw,device)
            
            
            norm = itn.normalize(correct_raw)
            print(f"norm: {norm}")
            print("---------------------")
            result['norm'] = norm
            results.append(result)
        json.dump(results, g, ensure_ascii=False, indent=4)
        