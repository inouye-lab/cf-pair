import wandb


sweep_configuration = {
    "program": "main.py",
    "method": "grid",
    "name": "CMNIST_CF_Coral_hyperparameter_search",
    "metric": {
        "goal": "maximize",
        "name": "test_acc"
        },
    "parameters": {
        "solver": {"values":["CF_DeepCoral"]},
        "param1": {"values":[10]},
        "lr": {"values":[1e-4]},
        "batch_size": {"values":[128]},
        "epochs": {"values":[20]},
        "seed": {"values":[1001,1002,1003,1004,1005,1006,1007,1008,1009,1010]},
        "split_scheme": {"values": ["official"]}
     }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="Paired_DG")
print(sweep_id)
wandb.agent(sweep_id)