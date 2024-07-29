import wandb


sweep_configuration = {
    "program": "main.py",
    "method": "grid",
    "name": "Oracle_random_seeds",
    "metric": {
        "goal": "maximize",
        "name": "test_acc"
        },
    "parameters": {
        "solver": {"values":["ERM"]},
        "lr": {"values":[1e-3]},
        "batch_size": {"values":[256]},
        "epochs": {"values":[40]},
        "seed": {"values":[1001]},
        "split_scheme": {"values": ["oracle"]}
     }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="Paired_DG")
print(sweep_id)
wandb.agent(sweep_id)