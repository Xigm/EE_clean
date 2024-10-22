import json
import os
import sys
sys.path.append(os.path.join(sys.path[0], '../../'))
# Define the path to the folder where the files are located
path = f"./weights/mistral"
path_weights_EE_mistral = path + f"/EE_1_layers_middle_2_wsum_pos_15_19_23_27"
dataset = "mmlu"
submetric = "acc" # acc, diff, max
recomputation = True
baseline = True

path = "./weights/mamba"
path_weights_EE_mamba = path + f"/EE_1_layers_middle_2_pos_32_40_48_56"
dataset = "mmlu"
recomputation = False
baseline = True
recomp = "/recompute_states" if recomputation else "/no_recomp" 
penalize = 4/24 if recomputation else 0.0


with open(f"{path_weights_EE_mamba}/results/"+dataset+"/baseline/results_list.json", "r") as f:
    results_list_baseline = json.load(f)

with open(f"{path_weights_EE_mamba}/results/"+dataset+"/baseline/layers_dropped.json", "r") as f:
    layers_dropped_baseline = json.load(f)

with open(f"{path_weights_EE_mistral}/results/"+dataset+"/baseline/results_list.json", "r") as f:
    results_list_baseline = json.load(f)

with open(f"{path_weights_EE_mistral}/results/"+dataset+"/baseline/layers_dropped.json", "r") as f:
    layers_dropped_baseline = json.load(f)

# # Now the variables results_list, exits_done, and positions_exited are loaded and ready to use
# print("Results List:", results_list)
# print("Exits Done:", exits_done)
# print("Positions Exited:", positions_exited)
# print("Lens Generated:", lens_generated)
n_layers_mamba = 64
n_layers_mistral = 32
metrics = ["acc,none"]
metric_values = []
bl_values_mamba = []
bl_values_mistral = []
for i,metric in enumerate(metrics):

    bl_values_mamba.append([r["results"][dataset][metric] for r in results_list_baseline])
    bl_values_mistral.append([r["results"][dataset][metric] for r in results_list_baseline])

import matplotlib.pyplot as plt

plt.plot([r/n_layers_mamba for r in layers_dropped_baseline], bl_values_mamba[0])
plt.plot([r/n_layers_mistral for r in layers_dropped_baseline], bl_values_mistral[0])

plt.savefig(f"{path_weights_EE_mamba}/results/"+dataset+"/baseline"+f"/{metric}_vs_layers_dropped.png")
plt.savefig(f"{path_weights_EE_mistral}/results/"+dataset+"/baseline"+f"/{metric}_vs_layers_dropped.png")
