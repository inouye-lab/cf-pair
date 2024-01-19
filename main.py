import argparse
import torch
import torch.cuda
import wandb
from wilds.common.data_loaders import get_eval_loader
from wilds import get_dataset
from src.solver import *
from src.utils import *

from wandb_env import WANDB_ENTITY, WANDB_PROJECT
"""
The main file function:
1. Load the hyperparameter dict.
2. Initialize logger
3. Initialize data (preprocess, data splits, etc.)
4. Initialize clients. 
5. Initialize Server.
6. Register clients at the server.
7. Start the server.
"""
def main(args):
    hparam = vars(args)
    training_angles = [int(item) for item in hparam['training_angles'].split(',')]

    hparam['training_angles'] = training_angles
    wandb_project = WANDB_PROJECT + '_' + hparam['solver']
    # setup WanDB
    if not args.no_wandb:
        wandb.init(project=wandb_project,
                    entity=WANDB_ENTITY,
                    config=hparam)
        wandb.run.log_code()
    hparam['wandb'] = not args.no_wandb
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hparam['device'] = device
    seed = hparam['seed']
    set_seed(seed)
    solver = eval(hparam['solver'])(hparam)
    solver.fit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FedDG Benchmark')
    parser.add_argument('--no_wandb', default=False, action="store_true")
    parser.add_argument('--root', default="/local/scratch/a/bai116/datasets", action="store_true")
    parser.add_argument('--seed', default=1001, type=int)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--solver', default='ERM', choices=["ERM", "Pair_Augmentation"])
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--param', default=100, type=float)
    parser.add_argument('--augmentation', default="cf", choices=["cf", "unpaired"])
    parser.add_argument('--training_angles', type=str)
    parser.add_argument('--test_angle', type=int)
    args = parser.parse_args()
    main(args)

