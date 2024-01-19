import wandb


sweep_configuration = {
    "program": "main.py",
    "method": "grid",
    "name": "CMNIST_CF_hyperparameter_search",
    "metric": {
        "goal": "maximize",
        "name": "test_acc"
        },
    "parameters": {
        "solver": {"values":["Pair_Augmentation"]},
        "augmentation": {"values":["unpaired"]},
        "param": {"values":[0,10,1000]},
        "lr": {"values":[1e-3]},
        "batch_size": {"values":[128]},
        "epochs": {"values":[40]},
        "seed": {"values":[1001]},
        "training_angles": {"values":["0,15,30,45,60"]},
        "test_angle": {"values":[75]},
     }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="Paired_DG")
print(sweep_id)
wandb.agent(sweep_id)