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

model_choice = "mamba" # gpt2, mistral, mamba
tokens_generated = 50
size = "350" # 124M, 350M, 774M, 1558M
# path = f"./weights/gpt2/gpt2_{size}M_100B_FinewebEdu_hf"
path = f"./weights/mamba"
path_weigths_EE = path + f"/EE_1_layers_middle_2_pos_32_40_48_56"
# path = f"./weights/mamba"
# path_weigths_EE = path + f"/EE_1_layers_middle_2_pos_16_20_24_28"
plot_intermediate_states = False
th_for_EE = 0.5
ee_pos = [int(p) for p in path_weigths_EE.split("_pos_")[-1].split("_")]
# ee_pos = None

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


model.th = model.th * th_for_EE if ee_pos is not None else None

if model_choice == "gpt2":
    for i in range(n_layer - 1):
        model.transformer.h[i].ee.load_state_dict(torch.load(f"{path_weigths_EE}/layer_{i}_EE"))
elif model_choice == "mistral":
    for i in range(len(ee_pos)):
        model.ee[i].load_state_dict(torch.load(f"{path_weigths_EE}/layer_{i}_EE"))
elif model_choice == "mamba":
    for i in range(len(ee_pos)):
        model.model.backbone.ee[i].load_state_dict(torch.load(f"{path_weigths_EE}/layer_{i}_EE"))

inputs = "What can you tell me about flowers?"

print("\n")

with torch.no_grad():
    output1 = model.generate(encode(inputs).to("cuda"), temperature=1e-6, max_new_tokens=tokens_generated, top_k = 10, use_EE = False)

if plot_intermediate_states:
    h_states = model.intermediate_states.clone()

print(decode(output1))

if model_choice == "mamba":
    model.inference_params["sequence_length"] = 0
    
with torch.no_grad():
    output2 = model.generate(encode(inputs).to("cuda"), temperature=1e-6, max_new_tokens=tokens_generated, top_k = 10, use_EE = True)

if plot_intermediate_states:
    h_states_EE = model.intermediate_states

output_text = decode(output2)

exits_done = model.exits_done
positions_exit = model.positions_exit

# capitalize words in exits
output_text = " ".join([word.upper() if i in positions_exit else word for i, word in enumerate(output_text.split())])
print(output_text)

# sum 1 if we use last block
if model_choice == "mistral":
    saved = sum([n_layer - e - 1 for e in exits_done])
elif model_choice == "mamba":
    saved = sum([n_layer - e for e in exits_done])

print(f"EEs saved {100*saved/(n_layer*tokens_generated)}% computation")

if plot_intermediate_states:

    # states to cpu
    h_states = h_states.cpu()
    h_states_EE = h_states_EE.cpu()

    # compute the diff between state i and i+1
    norm_dif = h_states[1:] - h_states[:-1]
    norm_dif_EE = h_states_EE[1:] - h_states_EE[:-1]

    # compute the norm of the difference
    norm_dif_norm = torch.norm(norm_dif, dim=-1)
    norm_dif_EE_norm = torch.norm(norm_dif_EE, dim=-1)

    # compute the cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(h_states[1:], h_states[:-1], dim=-1)
    cos_sim_no_EE = torch.nn.functional.cosine_similarity(h_states_EE[1:], h_states_EE[:-1], dim=-1)

    # plot the results for each layer
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axis_x = torch.arange(1, n_layer+1)

    for i in torch.arange(tokens_generated + len(encode(inputs))):
        if max(norm_dif_norm[:,i]) < 100:
            axs[0,0].plot(axis_x, norm_dif_norm[:,i])
        if max(norm_dif_EE_norm[:,i]) < 100:
            axs[1,0].plot(axis_x, norm_dif_EE_norm[:,i])


    axs[0,0].set_title("Norm of the difference between states")
    axs[0,0].plot(axis_x, norm_dif_norm[:,1: tokens_generated + len(encode(inputs))].mean(-1), color = 'black')
    axs[1,0].plot(axis_x, norm_dif_EE_norm[:, 1: tokens_generated + len(encode(inputs))].mean(-1), color = 'black')

    for i in torch.arange(tokens_generated + len(encode(inputs))):
        if not (cos_sim[:,i] >= 0.998).any():
            axs[0,1].plot(axis_x, cos_sim[:,i])
        if not (cos_sim_no_EE[:,i] >= 0.998).any():
            axs[1,1].plot(axis_x, cos_sim_no_EE[:,i])

    axs[0,1].set_title("Cosine similarity between states")
    axs[0,1].plot(axis_x, cos_sim[:, 1: tokens_generated + len(encode(inputs))].mean(-1), color = 'black')
    axs[1,1].plot(axis_x, cos_sim_no_EE[:, 1: tokens_generated + len(encode(inputs))].mean(-1), color = 'black')

    for ax in axs.reshape(-1):
        ax.legend()
        ax.set_xlabel("Layer")
        ax.set_ylabel("Value")

    plt.show()


# for i in range(tokens_generated):
#     print(f"Token {i}:")
#     print("Norm of the difference between states:", norm_dif_norm[:,i].mean().item())
#     print("Norm of the difference between states with EE:", norm_dif_EE_norm[:,i].mean().item())
#     print("Cosine similarity between states:", cos_sim[:,i].mean().item())
#     print("Cosine similarity between states with EE:", cos_sim_no_EE[:,i].mean().item())


print("Play with arrays")

