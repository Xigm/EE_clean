from datasets import load_dataset
import torch

# import models.mistral.model as mistral_model
from models.gpt2.model_EE import GPT
from models.gpt2.model import GPTConfig

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.instruct.request import InstructRequest
from mistral_common.protocol.instruct.messages import UserMessage
from models.mistral.model import Transformer, ModelArgs

import json
import tiktoken
# use name="sample-10BT" to use the 10BT sample
fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

model_choice = "gpt2"

if model_choice == "gpt2":
    GPTConfig.vocab_size = 50257
    model = GPT(GPTConfig)
    model.from_pretrained("gpt2")
    
elif model_choice == "mistral":
    path = "./weights/mistral/7b-v0.3"
    with open(path+ "/params.json") as f:
        args = ModelArgs(**dict(json.load(f)))
        args.lora.enable = False
        model = Transformer(args).to(torch.bfloat16).to("cuda")
    model.from_pretrained(path + "/consolidated.safetensors")

if model_choice == "gpt2":
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: torch.tensor(enc.encode(s, allowed_special={"<|endoftext|>"}))[None, :]
    decode = lambda l: enc.decode(l[0,:].tolist())
if model_choice == "mistral":
    enc = MistralTokenizer.v3().instruct_tokenizer
    createMsg = lambda s: InstructRequest(messages = [UserMessage(role = "user", content = s)])
    encode = lambda s: torch.tensor(enc.encode_instruct(createMsg(s)).tokens, device = "cuda")
    decode = lambda l: enc.decode(l.tolist())


model.freeze_backbone()

batch_size = 1
generated_tokens = 50

# check some samples
# for i, ex in enumerate(fw):
#     if i == 3:
#         break
#     print("////// Input:")
#     print(ex["text"])

#     e = encode(ex["text"])

#     if e.size(1) > 1024 - generated_tokens:
#         print("Skipping long input")
#         continue

#     with torch.no_grad():
#         output = model.generate(e, temperature=1, max_new_tokens=generated_tokens, top_k = 10)

#     print("////// Output:")
#     print(decode(output[:0, -generated_tokens:]))
#     print("\n\n")

fw.shuffle()


