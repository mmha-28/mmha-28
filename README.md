## MMHA-28: Human Action Recognition Across RGB, Depth, Thermal, and Event Modalities 

This repository provides the official implementation, dataset, and training scripts for **MMHA-28: Human Action Recognition Across RGB, Depth, Thermal, and Event Modalities** paper. 
It includes a baseline model based on the [VideoMamba](https://arxiv.org/abs/2403.13485) architecture codes.

## Installation
### 1. Clone the Repository
cd mmha-28

### 2. Pull the Docker image to ensure a consistent environment

   ```
   docker pull mmhm28/mmha-28:latest
   ```

### 3. Install the following local modules required by the VideoMamba architecture
   ```
   pip install -e mamba
   pip install -e causal-conv1d
   ```

## Our MMHA-28 Dataset
### 1.Download the MMHA-28 dataset from the official source
   ```
   tbd
   ```
Alternatively, a mini-sample version is available, containing data from one subject in session_1 and session_2, across all human actions. This is option for testing and visualization:

[tomirisss/MV-Mamba](tomirisss/mini-mmha)
   ```
huggingface-cli upload tomirisss/mini-mmha . --repo-type=dataset
   ```


### 2. Visualization
To visualize data from the mini-sample, run the following script with appropriate parameters:
   ```
   python vis.py --path PATH_TO_DATA --session session_1 --exp_num EXP_NUMBER
   ```

## Training

Navigate to the directory of the project:
   ```
cd videomamba/video_sm/
   ```

The video folder paths used during training, validation, and testing are specified in the "data/" directory, within the train.csv, val.csv, and test.csv files.
   ```
   path_to_video_folder  human_action_class
   ```

To begin training on the MMHA-28 dataset, first edit line 23 of the run.py script to set the --nproc_per_node= parameter according to the number of GPUs available on your system. Then, run:

   ```
   python3 run.py
   ```

## Evaluation

To test a pretrained model, first download the final Multimodal VideoMamba checkpoint:

[tomirisss/MV-Mamba](https://huggingface.co/tomirisss/MV-Mamba)

or using this code:

   ```
   huggingface-cli upload tomirisss/MV-Mamba .
   ```

MV-Mamba is the final multimodal model. The filename also indicates the number of frames used during training (e.g., MV-Mamba_f16.pth was trained with --num_frames=16).

Then, run the script, updating the --num_frames parameter and specifying the appropriate paths for MODEL_PATH.

   ```
   python3 run_test.py
   ```









