import torch

# import models.mistral.model as mistral_model
from models.gpt2.model_EE import GPT
from models.gpt2.model import GPTConfig

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.instruct.request import InstructRequest
from mistral_common.protocol.instruct.messages import UserMessage
from models.mistral.model_EE import Transformer, ModelArgs

from models.mamba.models.wrapper_mamba_EE import Mamba
from models.mamba.mistral_inference.args import MambaArgs

import json
import tiktoken

from rouge_score import rouge_scorer
import sacrebleu

model_choice = "mistral" # gpt2, mistral, mamba
tokens_generated = 50
size = "350" # 124M, 350M, 774M, 1558M
# path = f"./weights/gpt2/gpt2_{size}M_100B_FinewebEdu_hf"
# path = f"./weights/mamba"
# path_weigths_EE = path + f"/EE_1_layers_middle_2_pos_32_40_48_56"
path = f"./weights/mistral"
path_weigths_EE = path + f"/EE_1_layers_middle_2_pos_16_20_24_28"
plot_intermediate_states = True
th_for_EE = 0.5
ee_pos = [int(p) for p in path_weigths_EE.split("_pos_")[-1].split("_")]

if model_choice == "gpt2":
    
    # open config file
    with open(path + "/config.json") as f:
        config = json.load(f)

    # dump config into GPTConfig
    config_dataclass = GPTConfig(   block_size = config['n_ctx'],
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
        args.ee_pos = ee_pos
        model = Transformer(args).to(torch.bfloat16).to("cuda")
    model.from_pretrained(path + "/consolidated.safetensors")
    n_layer = model.args.n_layers

elif model_choice == "mamba":
    path = "./weights/mamba/mamba-codestral-7B-v0.1/"


    with open(path + "params.json", "r") as f:
        model_args = MambaArgs.from_dict(json.load(f))
        print(model_args)

    model_args.ee_pos = ee_pos
    model_args.block_size = 1024*4

    model = Mamba(model_args)
    # model.to("cuda")

    import time
    start_time = time.time()
    model.from_folder("./weights/mamba/mamba-codestral-7B-v0.1")
    print(f"Time to load model: {time.time() - start_time}")

    start_time = time.time()
    model.to("cuda")
    print(f"Time to load model to GPU: {time.time() - start_time}")
    n_layer = model_args.n_layers

if model_choice == "gpt2":
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: torch.tensor(enc.encode(s, allowed_special={"<|endoftext|>"}))[None, :]
    decode = lambda l: enc.decode(l[0,:].tolist())
if model_choice == "mistral" or model_choice == "mamba":
    enc = MistralTokenizer.v3().instruct_tokenizer
    createMsg = lambda s: InstructRequest(messages = [UserMessage(role = "user", content = s)])
    encode = lambda s: torch.tensor(enc.encode_instruct(createMsg(s)).tokens, device = "cuda")
    decode = lambda l: enc.decode(l.tolist())


model.th = model.th * th_for_EE

if model_choice == "gpt2":
    for i in range(n_layer - 1):
        model.transformer.h[i].ee.load_state_dict(torch.load(f"{path_weigths_EE}/layer_{i}_EE"))
elif model_choice == "mistral":
    for i in range(len(ee_pos)):
        model.ee[i].load_state_dict(torch.load(f"{path_weigths_EE}/layer_{i}_EE"))
elif model_choice == "mamba":
    for i in range(len(ee_pos)):
        model.model.backbone.ee[i].load_state_dict(torch.load(f"{path_weigths_EE}/layer_{i}_EE"))


# open dataset file
with open("datasets/incomplete_sentences/incomplete_random_sentences.csv") as f:
    data = f.readlines()

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


for line in data:
    print("Input:", line)
    input = encode(line)
    output_ee = model.generate(input, tokens_generated)
    output = model.generate(input, tokens_generated, use_EE = False)
    print("Output:", decode(output))

    # compute rogue score
    scores = scorer.score(output, output_ee)
    # compute bleu score
    score = sacrebleu.corpus_bleu([output_ee], [[output]])
