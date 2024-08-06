import argparse
import os
from typing import Dict
from pathlib import Path
import torch
from rich.panel import Panel

from piivot.modeling import Experiment
from piivot.utils.immutable import global_immutable
from piivot.utils.console import console


repo_path = Path(__file__).resolve().parents[0]

RESULTS_DIRNAME = "results"

def main(args: argparse.Namespace) -> None:
    """Main function to run the training or inference.

    Args:
        args (argparse.Namespace): The command line arguments.
    """
    global_immutable.DEBUG = args.debug

    global_immutable.YES = args.yes

    global_immutable.rerun = args.rerun

    global_immutable.DEVICE = (
        "cuda" if not args.use_cpu and torch.cuda.is_available() else "cpu"
    )

    if global_immutable.DEBUG:
        console.print(Panel("Running in DEBUG mode.", style="bold white on red"))
    
    if (args.use_parallelism):
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
    else:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    exp_folder = Path(args.exp_folder).resolve()
    config_filepaths = exp_folder.glob("*.json")
    config_filepaths = list(config_filepaths) # unsure why this is necessary. Calling list(config_filepaths) causes config_filepaths to == []
    console.print(f"Loading {len(config_filepaths)} config(s) from {exp_folder}")

    if args.test_config:
        config_filepaths = [config for config in config_filepaths if config.name == args.test_config]

        if len(config_filepaths) != 1:
            raise Exception(f"{args.test_config} must reference 1 and only 1 config in {args.exp_folder}") 
    
    if args.huggingface_hub and len(config_filepaths) > 1:
        raise Exception("You may only upload 1 trained model to huggingface.")

    for config_filepath in config_filepaths:
        experiment = Experiment(
            repo_path / RESULTS_DIRNAME,
            config_filepath,
            resume_checkpoint_filename=None,
            test=args.test_config != "",
        )
        
        model, tokenizer = experiment.run_train()

        if model and args.huggingface_hub:
            console.print(f"Uploading trained tokenizer and model to {args.huggingface_hub}.")
            tokenizer.push_to_hub(args.huggingface_hub, private=True)
            model.push_to_hub(args.huggingface_hub, private=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_folder", type=str, required=True, help="Path to experiment folder containing configs."
    )

    parser.add_argument(
        "--test_config", 
        type=str, 
        default="",
        help="Train against the test dataset of a previously ran experiment."
    )

    parser.add_argument(
        "--huggingface_hub",
        type=str,
        default=None,
        help="Huggingface repository id to save this model to."
    )

    parser.add_argument(
        "--debug",
        "-d",
        default=False,
        action="store_true",
        help="Run in debug mode.",
    )

    parser.add_argument(
        "--use_cpu",
        default=False,
        action="store_true",
        help="Use the CPU (even if a GPU is available).",
    )

    parser.add_argument(
        "--yes",
        "-y",
        default=False,
        action="store_true",
        help="Use 'Y' for all confirm actions.",
    )

    parser.add_argument(
        "--rerun",
        "-r",
        default=False,
        action="store_true",
        help="Give the option to rerun experiments if there is already a results directory it.",
    )
    parser.add_argument(
        "--use_parallelism",
        default=False,
        action="store_true",
        help="Allow parallelism during training.",
    )

    args = parser.parse_args()

    main(args)