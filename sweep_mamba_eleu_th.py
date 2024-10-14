import sys
import os
sys.path.append(os.path.join(sys.path[0], './EleutherAI_Eval_harness'))

from lm_eval.models.mamba_models_EE import Mamba_7b
from lm_eval import evaluate, simple_evaluate
from lm_eval.tasks import get_task_dict
from lm_eval.utils import make_table

# os.chdir(os.path.join(sys.path[0], './EE_Clean'))
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from models.mamba.models.wrapper_mamba_EE import Mamba
from models.mamba.mistral_inference.args import MambaArgs
from models.mistral.tokenizer import Tokenizer

import json
import torch

path_weights = "./weights/mamba/mamba-codestral-7B-v0.1"
max_length = 2048*2
max_gen_tokens = 64
device = "cuda"
batch_size = 1
recompute_states = False


print("Loading model...")

path = f"./weights/mamba"
path_weigths_EE = path + f"/EE_1_layers_middle_2_pos_32_40_48_56"
plot_intermediate_states = True
th_for_EE = 0.5
ee_pos = [int(p) for p in path_weigths_EE.split("_pos_")[-1].split("_")]

with open("./weights/mamba/mamba-codestral-7B-v0.1/params.json", "r") as f:
    model_args = MambaArgs.from_dict(json.load(f))
    print(model_args)

model_args.ee_pos = ee_pos
model_args.block_size = max_length

model = Mamba(model_args)

import time
start_time = time.time()
model.from_folder("./weights/mamba/mamba-codestral-7B-v0.1")
print(f"Time to load model: {time.time() - start_time}")


n_layer = model_args.n_layers

model.th = model.th * th_for_EE if ee_pos is not None else None


print("Sending to GPU...")
start_time = time.time()
model.eval()
if device == "cuda":
    model.to(device = device)
    print(f"Time to load model to GPU: {time.time() - start_time}")


if ee_pos is not None:
    for i in range(len(ee_pos)):
        model.model.backbone.ee[i].load_state_dict(torch.load(f"{path_weigths_EE}/layer_{i}_EE"))



print("Loading tokenizer...")
tokenizer = Tokenizer("./weights/mamba/mamba-codestral-7B-v0.1/tokenizer.model.v3")
# tokenizer = MistralTokenizer.v3().instruct_tokenizer

# instantiate an LM subclass that takes your initialized model and can run
# - `Your_LM.loglikelihood()`
# - `Your_LM.loglikelihood_rolling()`
# - `Your_LM.generate_until()`

# lm_obj = Mistral_7b(model=model,
#                     tokenizer = tokenizer,
#                     batch_size=batch_size,
#                     max_length = max_length,
#                     max_gen_tokens = max_gen_tokens,
#                     temperature = 1.0,
#                     top_k = None,
#                     recompute_states = False,
#                     use_EE = True if ee_pos is not None else False,
#                     device = device)

# optional: the task_manager indexes tasks including ones
# specified by the user through `include_path`.
# task_manager = lm_eval.tasks.TaskManager(
#     include_path="/path/to/custom/yaml"
#     )

# To get a task dict for `evaluate`
# task_dict = get_task_dict(
#     ["truthfulqa"]
#     )

range_th = torch.arange(0, 1.0, 0.1)
# range_th = torch.tensor([1, 0.7])
results_list = []
exits_done = []
positions_exited = []
lens_generated = []

for th in range_th:

    lm_obj = Mamba_7b(  model=model,
                    tokenizer = tokenizer,
                    batch_size=batch_size,
                    max_length = max_length,
                    max_gen_tokens = max_gen_tokens,
                    temperature = 1.0,
                    top_k = None,
                    use_EE = True if ee_pos is not None else False,
                    device = device)

    model.th = torch.ones(n_layer - 1) * th

    # for triviaqa in need to generete text
    # also for truthfulqa
    # nq_open

    # triviaqa WITH n fewshots 2!!!!
    # coqa ONLY posible with n_shots = 0
    # truthfulqa_gen n_shots = 1
    # dataset = "truthfulqa_gen"
    dataset = "coqa"

    results = simple_evaluate(
        model = lm_obj,
        tasks = [dataset],
        num_fewshot = 0,
    )

    results_list.append(results)

    if th == 1:
        exits_done.append(0)
        positions_exited.append(None)
    else:
        exits_done.append(lm_obj.model.exits_done)
        lm_obj.model.exits_done = []
        positions_exited.append(lm_obj.model.positions_exit)
        lm_obj.model.positions_exit = []
    lens_generated.append(lm_obj.model.lens_generated)
    lm_obj.model.lens_generated = []

print(make_table(results_list[0]))

# save results list, the exits done and the positions, if it does not exist, create it
if not os.path.exists(path_weigths_EE + f"/results/" + dataset + ("/recompute_states" if recompute_states else "/no_recomp")):
    os.makedirs(path_weigths_EE + f"/results/" + dataset + ("/recompute_states" if recompute_states else "/no_recomp"))

with open(path_weigths_EE + f"/results/"+dataset+ ("/recompute_states" if recompute_states else "/no_recomp") +"/results_list.json", "w") as f:
    json.dump(results_list, f)

with open(path_weigths_EE + f"/results/"+dataset+ ("/recompute_states" if recompute_states else "/no_recomp")+ "/exits_done.json", "w") as f:
    json.dump(exits_done, f)

with open(path_weigths_EE + f"/results/"+dataset+ ("/recompute_states" if recompute_states else "/no_recomp")+"/positions_exited.json", "w") as f:
    json.dump(positions_exited, f)

with open(path_weigths_EE + f"/results/"+dataset+ ("/recompute_states" if recompute_states else "/no_recomp")+"/lens_generated.json", "w") as f: 
    json.dump(lens_generated, f)

# save also the th swept
with open(path_weigths_EE + f"/results/"+dataset+ ("/recompute_states" if recompute_states else "/no_recomp")+"/th_swept.json", "w") as f:
    json.dump(range_th.tolist(), f)
