import wandb


sweep_configuration = {
    "program": "main.py",
    "method": "grid",
    "name": "CF_Pair_fine_tune",
    "metric": {
        "goal": "maximize",
        "name": "test_acc"
        },
    "parameters": {
        "solver": {"values":["Fewshot"]},
        "param1": {"values":[512]},
        "param3": {"values":[100]},
        "lr": {"values":[5e-4, 1e-4, 5e-3]},
        "feature_dimension": {"values":[64, 128, 256, 512]},
        "batch_size": {"values":[64, 128, 256, 512]},
        "epochs": {"values":[40]},
        "seed": {"values":[8888]},
        "split_scheme": {"values": ["official"]}
     }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="Paired_DG")
print(sweep_id)
wandb.agent(sweep_id)
