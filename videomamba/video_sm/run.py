import os
import subprocess

# Set environment variables
os.environ["OMP_NUM_THREADS"] = "1"

# Define paths (Change these to your actual paths)
JOB_NAME = "videomamba_middle_mask_f322_skip4_res224"
OUTPUT_DIR = f"exp_7/{JOB_NAME}"  # Replace with absolute path if needed
LOG_DIR = f"exp_7/logs/{JOB_NAME}"
PREFIX = "./"
DATA_PATH = "./"
# Define GPU settings

MODEL_PATH = "exp_7/videomamba_middle_mask_f32_skip4_res224/checkpoint-best.pth"
# Define GPU settings



# Construct the command without torchrun
ARGS = [
    "torchrun",  # Run with standard Python interpreter
    "--nproc_per_node=5",
    "run_class_finetuning.py",  # Main script
    "--model", "videomamba_middle",
    "--finetune", MODEL_PATH,
    "--data_path", DATA_PATH,
    "--prefix", PREFIX,
    "--data_set", "Kinetics_sparse",
    "--split", ",",
    "--nb_classes", "28",
    "--log_dir", OUTPUT_DIR,
    "--output_dir", OUTPUT_DIR,
    "--batch_size", "6",
    "--num_sample", "1",
    "--input_size", "224",
    "--short_side_size", "224",
    "--save_ckpt_freq", "10",
    "--num_frames", "32",
    "--num_workers", "4",  
    "--warmup_epochs", "5",
    "--tubelet_size", "1",
    "--epochs", "250",
    "--lr", "1e-5",
    "--layer_decay", "0.8",
    "--drop_path", "0.8",
    "--opt", "adamw",
    "--opt_betas", "0.9", "0.999",
    "--weight_decay", "0.05",
    "--test_num_segment", "4",
    "--test_num_crop", "3",
    "--dist_eval",
    "--test_best"
]

# Run the command
subprocess.run(ARGS)