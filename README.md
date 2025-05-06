## Hi there 👋

run this command to download the docker image needed to run this project or run pip -r requirements.txt
docker pull mmhm28/mmha-28:latest

Plese download the repisotory
Run commands to mount to libraries in the VideoMamba/ directory:

pip3 install -e mamba
pip3 install -e causal-conv1d


Download the MMHA-28 dataset using the link below:

Another option is to download the mini sample of this dataset, which contain 1 subject folder from session_1 and session_2 with all human actions:

http:/dataset

To visulalise the mini sample run the code with indication of path, session number (session_1 or session_2) and experiment number (exp_num):

python vis.py

The path to video folders are available in the train.csv, val.csv, test.csv. You can change it accordingly:
path_to_video_folder human_action_class

To start training on the MMHA-28 dataset run  

cd videomamba/video_sm/
python3 run.py

in run.py line 23 (--nproc_per_node=) you need to specify how many gpus you have for your training and specify --num_frames.

To test the model you need to download our final Multimodal VideoMamba models from hugging face, where MV-Mamba_f16.pth means model trained on num_frames=16: 

🔗 [tomirisss/MV-Mamba](https://huggingface.co/tomirisss/MV-Mamba)

Then run code with changing parameter "--num_frames" and paths LOG_DIR  and MODEL_PATH:








