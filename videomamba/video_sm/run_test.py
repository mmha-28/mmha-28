import os
import subprocess

# Set environment variables
os.environ["OMP_NUM_THREADS"] = "1"

# Define paths (Change these to your actual paths)
JOB_NAME = "videomamba_middle_mask_f8_res224"
OUTPUT_DIR = f"rgb/{JOB_NAME}"  # Replace with absolute path if needed
LOG_DIR = f"rgb/logs/{JOB_NAME}"
PREFIX = "./"
DATA_PATH = "./"
MODEL_PATH = "rgb/videomamba_middle_mask_f8_res224/Mamba-RGB_only.pth"

# Define GPU settings

# Construct the command without torchrun
ARGS = [
    "python",  # Run with standard Python interpreter
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
    "--check_path", MODEL_PATH,
    "--batch_size", "128",
    "--num_sample", "1",
    "--input_size", "224",
    "--short_side_size", "224",
    "--save_ckpt_freq", "10",
    "--num_frames", "8",
    "--num_workers", "1",  
    "--warmup_epochs", "5",
    "--tubelet_size", "1",
    "--epochs", "80",
    "--lr", "1e-3",
    "--layer_decay", "0.8",
    "--drop_path", "0.4",
    "--opt", "adamw",
    "--opt_betas", "0.9", "0.999",
    "--weight_decay", "0.05",
    "--test_num_segment", "4",
    "--test_num_crop", "3",
    "--dist_eval",
    "--test_best",
    "--smoothing","0",
    "--sampling_rate","4",
    "--test_best",
    "--use_checkpoint",
    "--eval"
]

# Run the command
subprocess.run(ARGS)
