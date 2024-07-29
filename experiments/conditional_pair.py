import wandb


sweep_configuration = {
    "program": "main.py",
    "method": "grid",
    "name": "Conditional_Pair_Random_Seeds",
    "metric": {
        "goal": "maximize",
        "name": "test_acc"
        },
    "parameters": {
        "solver": {"values":["Conditional_Pair"]},
        "param1": {"values":[100]},
        "lr": {"values":[2e-4]},
        "feature_dimension": {"values":[512]},
        "batch_size": {"values":[256]},
        "epochs": {"values":[20]},
        "seed": {"values":[1001,1002,1003,1004,1005,1006,1007,1008,1009,1010]},
        "split_scheme": {"values": ["official"]}
     }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="Paired_DG")
print(sweep_id)
wandb.agent(sweep_id)