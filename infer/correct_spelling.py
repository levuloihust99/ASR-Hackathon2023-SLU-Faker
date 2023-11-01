from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

path = "/home/ttmtuoi/projects/soict-slu-2023/misspell/train/checkpoints/v1/"

device = "cuda:0"
model = T5ForConditionalGeneration.from_pretrained(path)
model.eval()
model.to(device)
tokenizer = T5Tokenizer.from_pretrained("VietAI/vit5-base")

def normalize(s: str) -> str:
    input_ids = tokenizer(f"grammar: {s}", return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model.generate(input_ids)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    normalize("dùng van nốp trong nhà vệ sinh")