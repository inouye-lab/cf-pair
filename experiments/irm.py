import wandb


sweep_configuration = {
    "program": "main.py",
    "method": "grid",
    "name": "CMNIST_IRM_hyperparameter_search",
    "metric": {
        "goal": "maximize",
        "name": "test_acc"
        },
    "parameters": {
       "solver": {"values":["IRM"]},
        "param1": {"values":[0.1]},
        "lr": {"values":[1e-4]},
        "batch_size": {"values":[256]},
        "feature_dimension": {"values":[1024]},
        "epochs": {"values":[20]},
        "split_scheme": {"values": ["official"]},
        "seed": {"values": [1001,1002,1003,1004,1005,1006,1007,1008,1009,1010]}
     }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="Paired_DG")
print(sweep_id)
wandb.agent(sweep_id)