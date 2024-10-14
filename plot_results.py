import json

# Define the path to the folder where the files are located
# path = f"./weights/mistral"
# path_weights_EE = path + f"/EE_1_layers_middle_2_pos_16_20_24_28"
# dataset = "triviaqa"
# recomputation = False

path = "./weights/mamba"
path_weights_EE = path + f"/EE_1_layers_middle_2_pos_32_40_48_56"
dataset = "coqa"
recomputation = False


recomp = "/recompute_states" if recomputation else "/no_recomp" 
penalize = 4/24 if recomputation else 0.0

# Load the results_list from the JSON file
with open(f"{path_weights_EE}/results/"+dataset+recomp+"/results_list.json", "r") as f:
    results_list = json.load(f)

# Load the exits_done from the JSON file
with open(f"{path_weights_EE}/results/"+dataset+recomp+"/exits_done.json", "r") as f:
    exits_done = json.load(f)

# Load the positions_exited from the JSON file
with open(f"{path_weights_EE}/results/"+dataset+recomp+"/positions_exited.json", "r") as f:
    positions_exited = json.load(f)

with open(f"{path_weights_EE}/results/"+dataset+recomp+"/lens_generated.json", "r") as f:
    lens_generated = json.load(f)


with open(f"{path_weights_EE}/results/"+dataset+recomp+"/th_swept.json", "r") as f:
    range_th = json.load(f)

# # Now the variables results_list, exits_done, and positions_exited are loaded and ready to use
# print("Results List:", results_list)
# print("Exits Done:", exits_done)
# print("Positions Exited:", positions_exited)
# print("Lens Generated:", lens_generated)

# plot a graph of the results 
import matplotlib.pyplot as plt
import torch

n_layers = 32 if "mistral" in path_weights_EE else 64

range_th = torch.tensor(range_th)

# compute stimated speedup
speedups = []
speedup_r = 0
if "mistral" in path_weights_EE:
    for i, th in enumerate(range_th):
        if exits_done[i] == []:
            speedups.append(1)
        else:
            # for ex, pos, len in zip(exits_done, positions_exited, lens_generated):
            # speedup_r += n_layers/torch.tensor(ex, dtype = torch.float).mean()
            if recomp:
                pen = (penalize*(n_layers-torch.tensor(exits_done[i], dtype = torch.float).mean()))
            else:
                pen = 0
            speedups.append(n_layers/(pen+torch.tensor(exits_done[i], dtype = torch.float).mean()))
elif "mamba" in path_weights_EE:
    for i, th in enumerate(range_th):
        if exits_done[i] == []:
            speedups.append(1)
        else:
            if recomp:
                pen = (penalize*(n_layers-torch.tensor(exits_done[i], dtype = torch.float).mean()))
            else:
                pen = 0
            speedups.append(n_layers/(pen+torch.tensor(exits_done[i], dtype = torch.float).mean()))

# Define the y-axis values
if dataset == "triviaqa":
    metrics = "exact_match,remove_whitespace"
elif dataset == "coqa":
    metrics = ["f1,none", "em,none"]
elif dataset == "truthfulqa_gen":
    metrics = ["bleu_max,none","rouge1_max,none","rouge2_max,none","rougeL_max,none"]

metric_values = []
for metric in metrics:
    metric_values.append([r["results"][dataset][metric] for r in results_list])

for j, metric in enumerate(metric_values):
    # plot speedups vs y
    plt.figure(figsize=(10, 5)) 

    # Normalize the colors array for proper color mapping
    norm = plt.Normalize(vmin=min(range_th), vmax=max(range_th))
    cmap = plt.get_cmap('viridis')  # You can use any colormap

    # Plot the line (without color)
    plt.plot(speedups, metric, color='black', zorder=1, label = metrics[j].split(",")[0])

    # Plot the points with colors
    scatter = plt.scatter(speedups, metric, c=range_th, cmap=cmap, norm=norm, marker='o')

    # Add colorbar to the plot
    plt.colorbar(scatter, label='Th Value')

    plt.xlabel('Speedups')
    plt.ylabel('Metric Values')
    plt.legend()
    plt.savefig(f"{path_weights_EE}/results/"+dataset+recomp+"/speedup_vs_"+metrics[j].split(",")[0]+".png")



