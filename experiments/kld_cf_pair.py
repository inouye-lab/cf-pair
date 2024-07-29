import wandb


sweep_configuration = {
    "program": "main.py",
    "method": "grid",
    "name": "CMNIST_CF_KLD",
    "metric": {
        "goal": "maximize",
        "name": "test_acc"
        },
    "parameters": {
        "solver": {"values":["KLD_CF_Pair"]},
        "param1": {"values":[0.1, 1, 10, 100]},
        "lr": {"values":[1e-3]},
        "batch_size": {"values":[128]},
        "epochs": {"values":[40]},
        "seed": {"values":[1001]},
        "split_scheme": {"values": ["official"]}
     }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="Paired_DG")
print(sweep_id)
wandb.agent(sweep_id)