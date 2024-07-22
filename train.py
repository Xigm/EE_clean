from datasets import load_dataset
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

import os

from schedulefree import AdamWScheduleFree

# use name="sample-10BT" to use the 10BT sample
fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

model_choice = "mistral"

if model_choice == "gpt2":
    GPTConfig.vocab_size = 50257
    model = GPT(GPTConfig)
    model.from_pretrained("gpt2")
    model.to("cuda")
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
    encode = lambda s: torch.tensor(enc.encode(s, allowed_special={"<|endoftext|>"}), device = "cuda")[None, :]
    decode = lambda l: enc.decode(l[0,:].tolist())
if model_choice == "mistral":
    enc = MistralTokenizer.v3().instruct_tokenizer
    createMsg = lambda s: InstructRequest(messages = [UserMessage(role = "user", content = s)])
    encode = lambda s: torch.tensor(enc.encode_instruct(createMsg(s)).tokens, device = "cuda")
    decode = lambda l: enc.decode(l.tolist())


model.freeze_backbone()

batch_size = 1

fw = fw.shuffle()

# check number of trainble params
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print("Trainable parameters", trainable)
print(" EEs are a", round(100*trainable/total, 3), "% of the", model_choice, "model")

val_freq = 10

optimizers = []
for i in range(n_layer - 1):
    if model_choice == "gpt2":
        optimizers.append(AdamWScheduleFree(model.transformer.h[i].ee.parameters(), lr=0.005))
    if model_choice == "mistral":
        optimizers.append(AdamWScheduleFree(model.layers[i].ee.parameters(), lr=0.005))

iters = 150
model.k = 10
metrics_val, metrics = torch.zeros((int(iters/val_freq), 5)), torch.zeros((int(iters-iters/val_freq),5))


# create training loop
for i, ex in enumerate(fw):

    if iters == i:
        break

    # if i == 0:
    e = encode(ex["text"])
        # e = encode("bang bang bang bang bang bang bang bang bang bang bang bang bang bang bang bang bang bang bang bang bang bang bang")

    # crop long inputs
    if model_choice == "gpt2":
        if e.size(1) > 1024:
            e = e[:, :1024]
    if model_choice == "mistral":
        if e.size(0) > 1024*4:
            e = e[:1024*4]

    if i % val_freq == 0:
        with torch.no_grad():
            _, loss_val, metrics_val[int(i/val_freq)] = model(e, train_EE = True)
        print(f"Validation loss: {loss_val[0].item():.4f}")

    else:

        _, loss, metrics[i - int(i/val_freq) - 1] = model(e, train_EE = True)

        for i, optimizer in enumerate(optimizers):

            loss[i].backward(retain_graph=True)

            optimizers[i].step()

            optimizers[i].zero_grad()
        
        # losses = [l.item() for l in loss]
        # print(f"Training loss: {losses}")

        # print(f"Training loss: {loss.item():.4f}")

# test run
test_iters = 10
metrics_test = torch.zeros((test_iters, 5))
print("Running test...")
for i, ex in enumerate(fw):

    if i == test_iters:
        break

    e = encode(ex["text"])

    # crop long inputs
    if model_choice == "gpt2":
        if e.size(1) > 1024:
            e = e[:, :1024]
    if model_choice == "mistral":
        if e.size(0) > 1024*4:
            e = e[:1024*4]

    with torch.no_grad():
        _, _, metrics_test[i,:] = model(e, train_EE = True)

print("Test metrics:")
print("Test Accuracy:", metrics_test[:,0].mean().item())
print("Test Recall:", metrics_test[:,1].mean().item())
print("Test Precision:", metrics_test[:,2].mean().item())
print("Test F1:", metrics_test[:,3].mean().item())


import matplotlib.pyplot as plt

# generate x axis for val and train
x_train = torch.arange(0, iters, 1)
x_val = x_train[::val_freq]
x_train = [p.item() for p in x_train if p not in x_val]

print("Mean ratio is:", metrics[:,-1].mean().item())

# make a subplot with 4 graphs
fig, axs = plt.subplots(2, 2)
fig.suptitle('Training and Validation Metrics')
axs[0, 0].plot(x_train, metrics[:,0].numpy(), label="Training")
axs[0, 0].plot(x_val, metrics_val[:,0].numpy(), label="Validation")
axs[0, 0].plot(x_train, metrics[:,-1].numpy(), '--', label="Ratio")
axs[0, 0].set_title('EE Accuracy')
axs[0, 0].legend()
axs[0, 1].plot(x_train, metrics[:,1].numpy(), label="Training")
axs[0, 1].plot(x_val, metrics_val[:,1].numpy(), label="Validation")
axs[0, 1].legend()
axs[0, 1].set_title('EE Recall')
axs[1, 0].plot(x_train, metrics[:,2].numpy(), label="Training")
axs[1, 0].plot(x_val, metrics_val[:,2].numpy(), label="Validation")
axs[1, 0].set_title('EE Precision')
axs[1, 0].legend()
axs[1, 1].plot(x_train, metrics[:,3].numpy(), label="Training")
axs[1, 1].plot(x_val, metrics_val[:,3].numpy(), label="Validation")
axs[1, 1].legend()
axs[1, 1].set_title('EE F1')

plt.show()


# save EE weights
if model_choice == "gpt2":
    i = -1
    for n,m in model.transformer.h[0].ee.named_modules():
        i += 1
    name = f"EE_{i}_layers_middle_{model.transformer.h[0].ee.c_fc.weight.size(0)}"

    # create folder with name
    if not os.path.exists(f"./weights/gpt2/{name}"):
        os.makedirs(f"./weights/gpt2/{name}")

    for i in range(n_layer - 1):
        torch.save(model.transformer.h[i].ee.state_dict(), f"./weights/gpt2/{name}/layer_{i}_EE")


# save EE weights
if model_choice == "mistral":
    i = -1
    for n,m in model.layers[0].ee.named_modules():
        i += 1
    name = f"EE_{i}_layers_middle_{model.layers[0].ee.c_fc.weight.size(0)}"

    # create folder with name
    if not os.path.exists(f"./weights/mistral/{name}"):
        os.makedirs(f"./weights/mistral/{name}")

    for i in range(n_layer - 1):
        torch.save(model.layers[i].ee.state_dict(), f"./weights/mistral/{name}/layer_{i}_EE")

