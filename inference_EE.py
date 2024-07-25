import torch

# import models.mistral.model as mistral_model
from models.gpt2.model_EE import GPT
from models.gpt2.model import GPTConfig

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.instruct.request import InstructRequest
from mistral_common.protocol.instruct.messages import UserMessage
from models.mistral.model_EE import Transformer, ModelArgs

import json
import tiktoken

model_choice = "mistral"
tokens_generated = 100
path_weigths_EE = f"./weights/{model_choice}/EE_1_layers_middle_2"
plot_intermediate_states = True
th_for_EE = 0.95

if model_choice == "gpt2":
    GPTConfig.vocab_size = 50257
    model = GPT(GPTConfig)
    model.from_pretrained("gpt2")
    n_layer = model.config.n_layer
    
elif model_choice == "mistral":
    path = "./weights/mistral/7b-v0.3"
    with open(path+ "/params.json") as f:
        args = ModelArgs(**dict(json.load(f)))
        args.lora.enable = False
        model = Transformer(args).to(torch.bfloat16).to("cuda")
    model.from_pretrained(path + "/consolidated.safetensors")
    n_layer = model.args.n_layers

if model_choice == "gpt2":
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: torch.tensor(enc.encode(s, allowed_special={"<|endoftext|>"}))[None, :]
    decode = lambda l: enc.decode(l[0,:].tolist())
if model_choice == "mistral":
    enc = MistralTokenizer.v3().instruct_tokenizer
    createMsg = lambda s: InstructRequest(messages = [UserMessage(role = "user", content = s)])
    encode = lambda s: torch.tensor(enc.encode_instruct(createMsg(s)).tokens, device = "cuda")
    decode = lambda l: enc.decode(l.tolist())


model.th = model.th * th_for_EE

if model_choice == "gpt2":
    for i in range(n_layer - 1):
        model.transformer.h[i].ee.load_state_dict(torch.load(f"{path_weigths_EE}/layer_{i}_EE"))
elif model_choice == "mistral":
    for i in range(n_layer - 1):
        model.layers[i].ee.load_state_dict(torch.load(f"{path_weigths_EE}/layer_{i}_EE"))

inputs = "Tell me a story about a princess and a dragon."

with torch.no_grad():
    output1 = model.generate(encode(inputs), temperature=0.7, max_new_tokens=tokens_generated, top_k = 10, use_EE = False)

h_states = model.intermediate_states.clone()

print(decode(output1))

with torch.no_grad():
    output2 = model.generate(encode(inputs), temperature=0.7, max_new_tokens=tokens_generated, top_k = 10, use_EE = True)

h_states_EE = model.intermediate_states

print(decode(output2))

exits_done = model.exits_done

# sum 1 if we use last block
saved = sum([n_layer - e - 1 for e in exits_done])

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

