import wandb


sweep_configuration = {
    "program": "main.py",
    "method": "grid",
    "name": "Estimated_Conditional_Contrastive",
    "metric": {
        "goal": "maximize",
        "name": "test_acc"
        },
    "parameters": {
        "solver": {"values":["EstimatedContrastivePair"]},
        "pair_path": {"values": ["saved_model/conditional_pair.npy"]},
        "param1": {"values":[0.1, 1, 10]},
        "param2": {"values":[1]},
        "lr": {"values":[1e-4]},
        "feature_dimension": {"values":[512]},
        "batch_size": {"values":[128]},
        "epochs": {"values":[20]},
        "seed": {"values":[1001]},
        "split_scheme": {"values": ["official"]}
     }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="Paired_DG")
print(sweep_id)
wandb.agent(sweep_id)