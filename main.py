"""
Maintainer: Mateus Oliveira (mateus.oliveira@icomp.ufam.edu.br)
        Gabriel Dias (g172441@dac.unicamp.br)
        Marcio Almeida (m240781@dac.unicamp.br)
"""
import argparse
import gc
import os
import numpy as np
import wandb
from tqdm import trange
import torch
from constants import *
from metrics import calculate_metrics
from utils import read_yaml


def get_dataset(dataset_configs, configs, train=True):
    try:
        transform_type = list(dataset_configs.values())[0]["transform"]
        if transform_type:
            transformations_configs = configs["transformation"]
            transfomations = FACTORY_DICT["transformation"][transform_type](
                **transformations_configs[transform_type]
            )
            list(dataset_configs.values())[0]["transform"] = transfomations.get_transformations(train)
    except:
        pass

    dataset = FACTORY_DICT["dataset"][list(dataset_configs)[0]](
        **dataset_configs[list(dataset_configs.keys())[0]]
    )

    return dataset


def experiment_factory(configs):
    train_dataset_configs = configs["train_dataset"]
    validation_dataset_configs = configs["valid_dataset"]
    model_configs = configs["model"]
    optimizer_configs = configs["optimizer"]
    criterion_configs = configs["loss"]

    # Construct the dataloaders with any given transformations (if any)
    train_dataset = get_dataset(train_dataset_configs, configs, True)
    validation_dataset = get_dataset(validation_dataset_configs, configs, False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=configs["train"]["batch_size"], shuffle=True, num_workers=16
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=configs["valid"]["batch_size"], shuffle=False, num_workers=16
    )

    # Build model
    if type(model_configs) == dict:
        model = FACTORY_DICT["model"][list(model_configs.keys())[0]](
            **model_configs[list(model_configs.keys())[0]]
        )
    else:
        model = FACTORY_DICT["model"][model_configs]()

    optimizer = FACTORY_DICT["optimizer"][list(optimizer_configs.keys())[0]](
        model.parameters(), **optimizer_configs[list(optimizer_configs.keys())[0]]
    )
    criterion = FACTORY_DICT["loss"][list(criterion_configs.keys())[0]]

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min'
    )

    return model, train_loader, validation_loader, optimizer, \
        criterion, scheduler


def run_train_epoch(model, optimizer, criterion, loader,
                    monitoring_metrics, epoch, scheduler, run, step_update=1000):
    model.to(DEVICE)
    model.train()

    epoch_loss = 0

    with trange(len(loader), desc='Train Loop') as progress_bar:
        for batch_idx, sample_batch in zip(progress_bar, loader):
            optimizer.zero_grad()

            inputs, labels, _ = sample_batch[0], sample_batch[1], sample_batch[2]

            if type(inputs) != list:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

            pred_labels = model(inputs)
            loss = criterion(pred_labels, labels)
            epoch_loss += loss.item()

            progress_bar.set_postfix(
                desc=f'[epoch: {epoch + 1:d}], iteration: {batch_idx:d}/{len(train_loader):d}, loss: {loss.item()}'
            )

            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % step_update == 0:
                scheduler.step(loss.item())

        epoch_loss = (epoch_loss / len(loader))

    return epoch_loss


def run_validation(model, optimizer, criterion, loader,
                   epoch, configs, epsilon=1e-5):
    with torch.no_grad():
        torch.cuda.empty_cache()
        gc.collect()

        model.to(DEVICE)
        model.eval()
        running_loss = 0
        mean_mae = 0
        epoch_loss = 0

        with trange(len(loader), desc='Validation Loop') as progress_bar:
            for batch_idx, sample_batch in zip(progress_bar, loader):
                inputs, labels, _ = sample_batch[0], sample_batch[1], sample_batch[2]

                if type(inputs) != list:
                    inputs = inputs.to(DEVICE)
                    labels = labels.to(DEVICE)

                pred_labels = model(inputs)

                loss = criterion(pred_labels, labels)
                epoch_loss += loss.item()

                pred_labels = pred_labels.cpu().numpy()
                labels = labels.cpu().numpy()

                result = calculate_metrics(pred_labels, labels)

                running_loss += loss.cpu()

                mean_mae += np.array(result['mae']).mean()

                progress_bar.set_postfix(
                    desc=f"[Epoch {epoch + 1}] Loss: {running_loss / (batch_idx + 1):.3f} | mae:{np.array(result['mae']).mean():.7f}"
                )

    epoch_loss = (running_loss / len(loader)).detach().numpy()

    name_model = f"{configs['path_save_model']}{configs['network']}_{configs['reload_model']['data']}.pt"

    loader_mean_mae = mean_mae / len(loader)

    print(f"validation mean_mae: {mean_mae}")

    save_best_model(loader_mean_mae,
                    batch_idx,
                    model, optimizer, criterion, name_model, run)

    return epoch_loss


def get_params_lr_scheduler(configs):
    scheduler_kwargs = configs["lr_scheduler"]["info"]
    scheduler_type = configs["lr_scheduler"]["scheduler_type"]
    return scheduler_type, scheduler_kwargs


def calculate_parameters(model):
    qtd_model = sum(p.numel() for p in model.parameters())
    print(f"quantidade de parametros: {qtd_model}")
    return


def run_training_experiment(model, train_loader, validation_loader, optimizer,
                            criterion, scheduler, configs, run
                            ):
    os.makedirs(configs["path_save_model"], exist_ok=True)

    monitoring_metrics = {
        "loss": {"train": [], "validation": []},
        "accuracy": {"train": [], "validation": []}
    }

    calculate_parameters(model)

    for epoch in range(0, configs["epochs"] + 1):
        train_loss = run_train_epoch(
            model, optimizer, criterion, train_loader, monitoring_metrics,
            epoch, scheduler, run
        )

        valid_loss = run_validation(
            model, optimizer, criterion, validation_loader,
            epoch, configs
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="unsupervised main WileC")

    parser.add_argument(
        "config_file", type=str, help="Path to YAML configuration file"
    )

    args = parser.parse_args()

    configs = read_yaml(args.config_file)

    model, train_loader, validation_loader, \
        optimizer, criterion, scheduler = experiment_factory(configs)

    if configs['reload_model']['type']:
        name_model = f"{configs['path_to_save_model']}/{configs['network']}_{configs['reload_model']['data']}.pt"

        load_dict = torch.load(name_model)

        model.load_state_dict(load_dict['model_state_dict'])

    run = None

    run_training_experiment(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=None,
        configs=configs,
        run=run
    )
    # model, train_loader, validation_loader, optimizer,
    # criterion, scheduler, configs, run
    torch.cuda.empty_cache()
