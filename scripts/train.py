import argparse
import torch
import logging
import ase.io
import cace
import pickle
import os
import yaml
from cace.representations import Cace
from cace.modules import PolynomialCutoff, BesselRBF, Atomwise, Forces
from cace.models.atomistic import NeuralNetworkPotential
from cace.tasks.train import TrainingTask, GetLoss
from cace.tools import Metrics, init_device, compute_average_E0s, setup_logger, get_unique_atomic_number
from cace.tools import load_default_config, load_default_les_config

def main():
    parser = argparse.ArgumentParser(description='Train a CACE Neural Network Potential (NNP)')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML configuration file')
    args = parser.parse_args()

    config = load_default_config()
    with open(args.config, 'r') as f:
        user_config = yaml.safe_load(f)
    config.update(user_config)

    setup_logger(level='INFO', tag=config['prefix'], directory='./')
    device = init_device(config['use_device'])

    if config['zs'] is None:
        xyz = ase.io.read(config['train_path'], ':', format='extxyz')
        config['zs'] = get_unique_atomic_number(xyz)

    # load the avge0 dict from a file if possible
    if os.path.exists('avge0.pkl'):
        with open('avge0.pkl', 'rb') as f:
            avge0 = pickle.load(f)
    else:
        # Load Dataset
        avge0 = compute_average_E0s(xyz)
        with open('avge0.pkl', 'wb') as f:
            pickle.dump(avge0, f)

    # Prepare Data Loaders
    collection = cace.tasks.get_dataset_from_xyz(
        train_path=config['train_path'],
        valid_path=config['valid_path'],
        valid_fraction=config['valid_fraction'],
        data_key={'energy': config['energy_key'], 'forces': config['forces_key']},
        atomic_energies=avge0,
        cutoff=config['cutoff']
    )
    train_loader = cace.tasks.load_data_loader(
        collection=collection,
        data_type='train',
        batch_size=config['batch_size']
    )

    valid_loader = cace.tasks.load_data_loader(
        collection=collection,
        data_type='valid',
        batch_size=config['valid_batch_size']
    )

    # Configure CACE Representation
    cutoff_fn = PolynomialCutoff(cutoff=config['cutoff'], p=config['cutoff_fn_p'])
    radial_basis = BesselRBF(cutoff=config['cutoff'], n_rbf=config['n_rbf'], trainable=config['trainable_rbf'])
    cace_representation = Cace(
        zs=config['zs'], n_atom_basis=config['n_atom_basis'], embed_receiver_nodes=config['embed_receiver_nodes'],
        cutoff=config['cutoff'], cutoff_fn=cutoff_fn, radial_basis=radial_basis,
        n_radial_basis=config['n_radial_basis'], max_l=config['max_l'], max_nu=config['max_nu'],
        device=device, num_message_passing=config['num_message_passing']
    )
    # Configure Atomwise Module
    # LES if needed
    if 'les' in config and config['les']:
        sr_energy = Atomwise(
            n_layers=config['atomwise_layers'],
            n_hidden=config['atomwise_hidden'],
            residual=config['atomwise_residual'],
            use_batchnorm=config['atomwise_batchnorm'],
            add_linear_nn=config['atomwise_linear_nn'],
            output_key='SR_energy'
        )
        les_config = load_default_les_config()
        les_config.update(config['les'])
        lr_charge = Atomwise(
            n_layers=les_config['lr_n_layers'],
            n_hidden=les_config['lr_n_hidden'],
            n_out=les_config['lr_n_out'],
            per_atom_output_key='q',
            output_key='ewald_potential',
            residual=False, add_linear_nn=False, use_batchnorm=False
        )
        ewald_pot = cace.modules.EwaldPotential(
            dl=les_config['lr_dl'],
            sigma=les_config['lr_sigma'],
            output_key='LR_energy'
        )
        tot_energy = cace.modules.FeatureAdd(
            feature_keys=['SR_energy', 'LR_energy'], output_key='CACE_energy'
        )
        output_modules = [sr_energy, lr_charge, ewald_pot, tot_energy]
    else:
        energy = Atomwise(
            n_layers=config['atomwise_layers'], n_hidden=config['atomwise_hidden'], residual=config['atomwise_residual'],
            use_batchnorm=config['atomwise_batchnorm'], add_linear_nn=config['atomwise_linear_nn'],
            output_key='CACE_energy'
        )
        output_modules = [energy]

    # Configure Forces Module
    forces = Forces(energy_key='CACE_energy', forces_key='CACE_forces')
    output_modules.append(forces)

    # Assemble Neural Network Potential
    cace_nnp = NeuralNetworkPotential(
        representation=cace_representation,
        output_modules=output_modules
    ).to(device)

    # Phase 1 Training Configuration
    optimizer_args = {'lr': float(config['lr'])}
    scheduler_args = {'mode': 'min', 'factor': config['scheduler_factor'], 'patience': config['scheduler_patience']}
    energy_loss = GetLoss(
        target_name='energy',
        predict_name='CACE_energy',
        loss_fn=torch.nn.MSELoss(),
        loss_weight=config['energy_loss_weight']
    )
    force_loss = GetLoss(
        target_name='forces', 
        predict_name='CACE_forces', 
        loss_fn=torch.nn.MSELoss(), 
        loss_weight=config['force_loss_weight']
    )


    e_metric = Metrics(
        target_name='energy',
        predict_name='CACE_energy',
        name='e/atom',
        per_atom=True
    )

    f_metric = Metrics(
        target_name='forces',
        predict_name='CACE_forces',
        name='f'
    )

    for _ in range(config["num_restart"]): 
        # Initialize and Fit Training Task for Phase 1
        task = TrainingTask(
            model=cace_nnp, losses=[energy_loss, force_loss], metrics=[e_metric, f_metric],
            device=device, optimizer_args=optimizer_args, scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
            scheduler_args=scheduler_args, max_grad_norm=config["max_grad_norm"], ema=config["ema"],
            ema_start=config["ema_start"], warmup_steps=config["warmup_steps"]
        )
        task.fit(train_loader, valid_loader, epochs=int(config["epochs"]/config["num_restart"]), print_stride=0)
    task.save_model(config["prefix"]+'_phase_1.pth')
    # Phase 2 Training Adjustment
    energy_loss_2 = GetLoss('energy', 'CACE_energy', torch.nn.MSELoss(), config["second_phase_energy_loss_weight"])
    task.update_loss([energy_loss_2, force_loss])

    # Fit Training Task for Phase 2
    task.fit(train_loader, valid_loader, epochs=config["second_phase_epochs"])
    task.save_model(config["prefix"]+'_phase_2.pth')

if __name__ == '__main__':
    main()

