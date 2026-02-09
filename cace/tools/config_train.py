__all__ = ['load_default_config']


def load_default_config():
    default_config = {
        "prefix": "CACE_NNP",
        "train_path": "",
        "valid_path": None,
        "valid_fraction": 0.1,
        "energy_key": "energy",
        "forces_key": "forces",
        "cutoff": 4.0,
        "batch_size": 10,
        "valid_batch_size": 20,
        "use_device": "cpu",
        "n_rbf": 6,
        "trainable_rbf": False,
        "cutoff_fn": "PolynomialCutoff",
        "cutoff_fn_p": 5,
        "zs": None,
        "n_atom_basis": 3,
        "n_radial_basis": 8,
        "max_l": 3,
        "max_nu": 3,
        "num_message_passing": 1,
        "embed_receiver_nodes": False,
        "atomwise_layers": 3,
        "atomwise_hidden": [32, 16],
        "atomwise_residual": True,
        "atomwise_batchnorm": False,
        "atomwise_linear_nn": True,
        "lr": 1e-2,
        "scheduler_factor": 0.8,
        "scheduler_patience": 10,
        "max_grad_norm": 10,
        "ema": False,
        "ema_start": 10,
        "warmup_steps": 10,
        "epochs": 200,
        "second_phase_epochs": 100,
        "energy_loss_weight": 1.0,
        "force_loss_weight": 1000.0,
        "num_restart": 5,
        "second_phase_energy_loss_weight": 1000.0
    }
    return default_config


def load_default_les_config():
    les_config = {
        "lr_n_layers": 3,
        "lr_n_hidden": [24, 12],
        "lr_n_out": 1,
        "lr_dl": 2,
        "lr_sigma": 0.1,
    }
    return les_config