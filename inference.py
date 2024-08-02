import torch

# import models.mistral.model as mistral_model
from models.gpt2.model import GPT
from models.gpt2.model import GPTConfig

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.instruct.request import InstructRequest
from mistral_common.protocol.instruct.messages import UserMessage
from models.mistral.model import Transformer, ModelArgs

from models.mamba.models.wrapper_mamba import Mamba
from models.mamba.mistral_inference.args import MambaArgs

import json
import tiktoken

model_choice = "mamba"
tokens_generated = 30
size = "1558" # 124M, 350M, 774M, 1558M
# path = f"./weights/gpt2/gpt2_{size}M_100B_FinewebEdu_hf"
path = f"./weights/gpt2/gpt2_{size}M_final2_hf"
# path = f"./weights/gpt2/gpt2"


if model_choice == "gpt2":

    # open config file
    with open(path + "/config.json") as f:
        config = json.load(f)

    # dump config into GPTConfig
    config_dataclass = GPTConfig(   
                                    block_size = config['n_ctx'],
                                    vocab_size = config['vocab_size'],
                                    n_layer = config['n_layer'],
                                    n_head = config['n_head'],
                                    n_embd = config['n_embd'],
                                    dropout = config['attn_pdrop'],
                                    bias = True,
                                )

    model = GPT(config_dataclass)
    model.from_hf(path + "/model.safetensors")

    model.to("cuda")
    n_layer = model.config.n_layer
    
elif model_choice == "mistral":
    path = "./weights/mistral/7b-v0.3"
    with open(path+ "/params.json") as f:
        args = ModelArgs(**dict(json.load(f)))
        args.lora.enable = False
        model = Transformer(args).to(torch.bfloat16).to("cuda")
    model.from_pretrained(path + "/consolidated.safetensors")

elif model_choice == "mamba":
    path = "./weights/mamba/mamba-codestral-7B-v0.1/"


    with open(path + "params.json", "r") as f:
        model_args = MambaArgs.from_dict(json.load(f))
        print(model_args)

    model = Mamba(model_args)
    model.to(torch.bfloat16).to("cuda")

    import time
    start_time = time.time()
    model.from_folder("./weights/mamba/mamba-codestral-7B-v0.1")
    print(f"Time to load model: {time.time() - start_time}")

    # start_time = time.time()
    # model.to("cuda")
    # print(f"Time to load model to GPU: {time.time() - start_time}")
    n_layer = model_args.n_layers

if model_choice == "gpt2":
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: torch.tensor(enc.encode(s, allowed_special={"<|endoftext|>"}))[None, :]
    decode = lambda l: enc.decode(l[0,:].tolist())
elif model_choice == "mistral" or model_choice == "mamba":
    enc = MistralTokenizer.v3().instruct_tokenizer
    createMsg = lambda s: InstructRequest(messages = [UserMessage(role = "user", content = s)])
    encode = lambda s: torch.tensor(enc.encode_instruct(createMsg(s)).tokens, device = "cuda")
    decode = lambda l: enc.decode(l.tolist())

inputs = "Hello, I'm mamba, a language model,"

with torch.no_grad():
    output = model.generate(encode(inputs).to("cuda"), temperature=0.5, max_new_tokens=tokens_generated, top_k = 10)

print(output)
print(decode(output))