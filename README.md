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

The path to video folders are available in the train.csv, val.csv, test.csv. You can change it accordingly:
path_to_video_folder human_action_class

To start training on the MMHA-28 dataset run  




