import wandb


sweep_configuration = {
    "program": "main.py",
    "method": "grid",
    "name": "Estimated_Contrastive",
    "metric": {
        "goal": "maximize",
        "name": "test_acc"
        },
    "parameters": {
        "solver": {"values":["EstimatedContrastivePair"]},
        "pair_path": {"values": ["marginal_pair.npy"]},
        "param1": {"values":[100]},
        "param2": {"values":[0.01]},
        "param3": {"values":[100]},
        "lr": {"values":[1e-4]},
        "latent_dim": {"values":[16]},
        "batch_size": {"values":[128]},
        "epochs": {"values":[60]},
        "seed": {"values":[1001]},
        "split_scheme": {"values": ["official"]}
     }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="Paired_DG")
print(sweep_id)
wandb.agent(sweep_id)