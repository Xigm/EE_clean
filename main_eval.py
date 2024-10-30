import sys
import os
sys.path.append(sys.path[0] + '/EleutherAI_Eval_harness')
sys.path.append(os.path.join(sys.path[0], '../../'))

from lm_eval.models.mamba_models_EE import Mamba_7b
from lm_eval.models.mistral_models_EE import Mistral_7b
from lm_eval import evaluate, simple_evaluate
from lm_eval.tasks import get_task_dict
from lm_eval.utils import make_table

# os.chdir(os.path.join(sys.path[0], './EE_Clean'))

from models.mistral.model import ModelArgs
from models.mistral.model_EE import Transformer
from models.mistral.tokenizer import Tokenizer
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from models.mamba.models.wrapper_mamba_EE import Mamba
from models.mamba.mistral_inference.args import MambaArgs

import json
import torch

def eval_mistral(task, n_shots = None, max_gen_tokens = 64, temperature = 1.0, topk = None, recomp = False, use_EE = True, th_for_EE = 0.6):

    if n_shots is None:
        n_shots = 0

    path_weights = "./weights/mistral/7B-v0.3"
    max_length = 2048*2
    device = "cuda"
    batch_size = 1

    # create your model (could be running finetuning with some custom modeling code)
    if path_weights is not None:
        with open(path_weights + "/params.json", "r") as f:
            args = ModelArgs.from_dict(json.load(f))

    args.max_batch_size = batch_size

    print("Loading model...")

    path = f"./weights/mistral"
    path_weigths_EE = path + f"/EE_1_layers_middle_2_pos_16_20_24_28"
    plot_intermediate_states = True
    # th_for_EE = 0.6
    ee_pos = [int(p) for p in path_weigths_EE.split("_pos_")[-1].split("_")]
    # ee_pos = None


    path = "./weights/mistral/7b-v0.3"
    with open(path+ "/params.json") as f:
        args = ModelArgs(**dict(json.load(f)))
        args.lora.enable = False
        args.ee_pos = ee_pos
        model = Transformer(args)

    print("Loading weights...")
    model.from_pretrained(path + "/consolidated.safetensors")
    n_layer = model.args.n_layers

    model.th = model.th * th_for_EE

    if ee_pos is not None:
        for i in range(len(ee_pos)):
            model.ee[i].load_state_dict(torch.load(f"{path_weigths_EE}/layer_{i}_EE"))

    print("Sending to GPU...")
    model.eval()
    if device == "cuda":
        model.to(device = device, dtype = torch.bfloat16)

    print("Loading tokenizer...")
    tokenizer = Tokenizer(path_weights + "/tokenizer.model.v3")
    # tokenizer = MistralTokenizer.v3().instruct_tokenizer

    # instantiate an LM subclass that takes your initialized model and can run
    # - `Your_LM.loglikelihood()`
    # - `Your_LM.loglikelihood_rolling()`
    # - `Your_LM.generate_until()`

    lm_obj = Mistral_7b(model=model,
                        tokenizer = tokenizer,
                        batch_size=batch_size,
                        max_length = max_length,
                        max_gen_tokens = max_gen_tokens,
                        temperature = temperature,
                        top_k = topk,
                        recompute_states = recomp,
                        use_EE = use_EE,
                        device = device)

    # optional: the task_manager indexes tasks including ones
    # specified by the user through `include_path`.
    # task_manager = lm_eval.tasks.TaskManager(
    #     include_path="/path/to/custom/yaml"
    #     )

    # To get a task dict for `evaluate`
    # task_dict = get_task_dict(
    #     ["truthfulqa"]
    #     )

    print("Evaluating...")
    # results = evaluate(
    #     lm=lm_obj,
    #     task_dict=task_dict,
    #     # num_fewshot=3,
    # )


    # for triviaqa in need to generete text
    # also for truthfulqa
    # nq_open

    # triviaqa
    # coqa
    # generate_until

    results = simple_evaluate(
        model = lm_obj,
        tasks = [task],
        num_fewshot = n_shots,
    )

    print(make_table(results))

def eval_mamba(dataset, n_shots = None, max_gen_tokens = 64, temperature = 1.0, topk = None, recompute_states = False, use_EE = True, th_for_EE = 0.6):
    

    max_length = 2048*2
    # max_gen_tokens = 64
    device = "cuda"
    batch_size = 1

    # create your model (could be running finetuning with some custom modeling code)
    print("Loading model...")

    path = f"./weights/mamba"
    path_weigths_EE = path + f"/EE_1_layers_middle_2_pos_32_40_48_56"
    plot_intermediate_states = False
    # th_for_EE = 0.5
    ee_pos = [int(p) for p in path_weigths_EE.split("_pos_")[-1].split("_")]
    # ee_pos = None
    # recompute_states = True

    with open("./weights/mamba/mamba-codestral-7B-v0.1/params.json", "r") as f:
        model_args = MambaArgs.from_dict(json.load(f))
        print(model_args)

    model_args.ee_pos = ee_pos
    model_args.block_size = max_length

    model = Mamba(model_args)
    # model.to("cuda")

    import time
    start_time = time.time()
    model.from_folder("./weights/mamba/mamba-codestral-7B-v0.1")
    print(f"Time to load model: {time.time() - start_time}")


    n_layer = model_args.n_layers

    model.th = model.th * th_for_EE if ee_pos is not None else None

    start_time = time.time()
    model.eval()
    model.to(device = device)
    print(f"Time to load model to GPU: {time.time() - start_time}")

    if ee_pos is not None:
        for i in range(len(ee_pos)):
            model.model.backbone.ee[i].load_state_dict(torch.load(f"{path_weigths_EE}/layer_{i}_EE"))

    print("Loading tokenizer...")
    # tokenizer = Tokenizer("./weights/mamba/mamba-codestral-7B-v0.1" + "/tokenizer.model.v3")
    # tokenizer = MistralTokenizer.v3().instruct_tokenizer
    tokenizer = Tokenizer("./weights/mamba/mamba-codestral-7B-v0.1/tokenizer.model.v3")


    # instantiate an LM subclass that takes your initialized model and can run
    # - `Your_LM.loglikelihood()`
    # - `Your_LM.loglikelihood_rolling()`
    # - `Your_LM.generate_until()`
    lm_obj = Mamba_7b(  model=model,
                        tokenizer = tokenizer,
                        batch_size=batch_size,
                        max_length = max_length,
                        max_gen_tokens = max_gen_tokens,
                        temperature = temperature,
                        top_k = topk,
                        use_EE =  use_EE,
                        recompute_states = recompute_states,
                        device = device)

    # optional: the task_manager indexes tasks including ones
    # specified by the user through `include_path`.
    # task_manager = lm_eval.tasks.TaskManager(
    #     include_path="/path/to/custom/yaml"
    #     )

    # To get a task dict for `evaluate`
    # task_dict = get_task_dict(
    #     ["truthfulqa"]
    #     )

    print("Evaluating...")  
    # results = evaluate(
    #     lm=lm_obj,
    #     task_dict=task_dict,
    #     # num_fewshot=3,
    # )


    # triviaqa
    # coqa has to be with n_shots = 0
    # truthfulqa
    results = simple_evaluate(
        model = lm_obj,
        tasks = [dataset],
        num_fewshot = n_shots,
    )

    print(make_table(results))



if __name__ == "__main__":
    # get args from config file 
    with open("./eval_config.json", "r") as f:
        config = json.load(f)
    
    # check all values and if they dont exist assign default values
    task = config.get("task", "triviaqa")
    n_shots = config.get("n_shots", None)
    max_gen_tokens = config.get("max_gen_tokens", 64)
    temperature = config.get("temperature", 1.0)
    topk = config.get("topk", None)
    recomp = config.get("recomp", False)
    use_EE = config.get("use_EE", True)
    th_for_EE = config.get("th_for_EE", 0.6)
    model = config.get("model", "mistral")

    if model == "mistral":
        eval_mistral(task, n_shots, max_gen_tokens, temperature, topk, recomp, use_EE, th_for_EE)
    elif model == "mamba":
        eval_mamba(task, n_shots, max_gen_tokens, temperature, topk, recomp, use_EE, th_for_EE)
    else:
        print("Model not found")    