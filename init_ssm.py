import json
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

from models.mamba.models.mixer_seq_simple import MambaLMHeadModel

tokenize = MistralTokenizer.v3().from_file("./weights/mamba/mamba-codestral-7B-v0.1/tokenizer.model.v3")

with open("./weights/mamba/mamba-codestral-7B-v0.1/params.json") as f:
    config = json.load(f)
    print(config)

model = MambaLMHeadModel(config)

model.from_file("./weights/mamba/mamba-codestral-7B-v0.1/consolidated.safetensors")



