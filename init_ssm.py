import json
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

from models.mamba.models.wrapper_mamba import Mamba
from models.mamba.mistral_inference.args import MambaArgs

import torch

tokenize = MistralTokenizer.v3().from_file("./weights/mamba/mamba-codestral-7B-v0.1/tokenizer.model.v3")

path = "./weights/mamba/mamba-codestral-7B-v0.1/"


with open(path + "params.json", "r") as f:
    model_args = MambaArgs.from_dict(json.load(f))
    print(model_args)

with torch.device("meta"):
    model = Mamba(model_args)

import time
start_time = time.time()
model.from_folder("./weights/mamba/mamba-codestral-7B-v0.1")
print(f"Time to load model: {time.time() - start_time}")

print(model)



