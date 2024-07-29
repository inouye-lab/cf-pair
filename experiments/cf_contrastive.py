import wandb


sweep_configuration = {
    "program": "main.py",
    "method": "grid",
    "name": "CF_Contrastive",
    "metric": {
        "goal": "maximize",
        "name": "test_acc"
        },
    "parameters": {
        "solver": {"values":["CF_Contrastive"]},
        "param1": {"values":[100]},
        "param2": {"values":[0.01]},
        "param3": {"values":[1000, 10000]},
        "lr": {"values":[2e-4]},
        "feature_dimension": {"values":[512]},
        "batch_size": {"values":[256]},
        "epochs": {"values":[20]},
        "seed": {"values":[1001]},
        "split_scheme": {"values": ["official"]}
     }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="Paired_DG")
print(sweep_id)
wandb.agent(sweep_id)