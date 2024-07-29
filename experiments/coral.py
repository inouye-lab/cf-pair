import wandb


sweep_configuration = {
    "program": "main.py",
    "method": "grid",
    "name": "CMNIST_Coral_hyperparameter_search",
    "metric": {
        "goal": "maximize",
        "name": "test_acc"
        },
    "parameters": {
        "solver": {"values":["DeepCoral"]},
        "param1": {"values":[0.01]},
        "lr": {"values":[1e-4]},
        "batch_size": {"values":[256]},
        "feature_dimension": {"values":[1024]},
        "epochs": {"values":[20]},
        "seed": {"values":[1001,1002,1003,1004,1005,1006,1007,1008,1009,1010]},
        "split_scheme": {"values": ["official"]}
     }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="Paired_DG")
print(sweep_id)
wandb.agent(sweep_id)