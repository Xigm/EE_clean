# Project Title

**Brief description**: A concise summary of what the project does, its goals, or its main functionality.

---

## Table of Contents
- [EE_Clean (in the near future MEEstral)](#project-title)
    - [Installation](#installation)
    - [Usage](#usage)
    <!-- - [Features](#features) -->
    - [Project Structure](#project-structure)
    - [Acknowledgements](#acknowledgements)

---

## Installation
Step-by-step instructions on how to install and set up the project locally. This might include cloning the repository and installing dependencies.

```bash
git clone https://github.com/Xigm/EE_Clean.git
```

Install environment with the environment.yml file. Linux OS is requiered.

```bash
conda env create -f environment.yml
```


---

## Usage

To train your own early exits go to file train.py. Specify where the EE's are placed. Training is perfomed with the dataset FineWeb-Edu.

To test inference of the models go to file inference_EE.py. Select the model and modify the input.

---

<!-- ## Features
Highlight the main features of your project. What makes it special? This is a good place to bullet out the key functionality or purpose of the code.

- Feature 1
- Feature 2
- Feature 3

--- -->

## Project Structure


```
├── datasets/                           # If some local hosted dataset if needed
│   ├── models/                         # ML models
│   ├── utils/                          # Helper functions
├── EleutherAI_Eval_harness/            # Codes from EleutherAI to evaluate LLMs
│   ├── lm_eval/                        
│      ├── models/                      # Wrappers for models to be tested are here
│          ├── mamba_models_EE.py       # Custom wrapper for mamba 
│          ├── mistral_models_EE.py     # Custom wrapper for mistral
│      ├── tasks/                       # Different task available
├── evals/                              # Codes to perform evaluations
│   ├── individual_evals /              # Evaluate a model in a single task
│   ├── sweep_th/                       # Get results for speed up vs performance
│   plot_results.py                     # Compute the graphs for the data obtained
├── models/                             # Main codes for the models
│   ├── mamba/                          # Mamba implementations
│   ├── mistral/                        # Transformer implementations
├── weights/                            # To save the backbones and EE weights
│   ├── mamba/                           
│   │   ├── codestral7b/                # Main backbone weights
│   │   ├── EE_given_config/            # EE weights for a given configuration
│   └── mistral/                        # Transformer implementations
│       ├── mistral7b/                  # Main backbone weights
│       └── EE_given_config/            # EE weights for a given configuration
├── envirnment.yml                      # File to import env to conda
├── inference_EE.py                     # Code to perfor inference of the models
├── train.py                            # Code to train the EEs
├── tasks.txt                           # List of all available tasks in EleutherAi eval harness
└── utils.py                            # Some aux functions
```

---

## Contributing
Instructions for users who want to contribute to the project...

---

## Acknowledgements

Thanks to:
    - EleutherAI
    - Mistral
    - HuggingFace
