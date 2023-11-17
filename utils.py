"""
Maintainer: Mateus Oliveira (mateus.oliveira@icomp.ufam.edu.br)
        Gabriel Dias (g172441@dac.unicamp.br)
        Marcio Almeida (m240781@dac.unicamp.br)
"""

import random
import torch
import yaml
from tqdm import tqdm
import numpy as np
from typing import List
from sklearn.metrics import confusion_matrix
import h5py
from scipy import signal
import os
import itertools


def set_device():
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    print('Using {}'.format(device))

    return device


def read_yaml(file: str) -> yaml.loader.FullLoader:
    with open(file, "r") as yaml_file:
        configurations = yaml.load(yaml_file, Loader=yaml.FullLoader)

    return configurations


class ReadDatasets:
    @staticmethod
    def read_h5_pred_fit(filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with h5py.File(filename) as hf:
            input_spec = hf["input_spec"][()]
            ground = hf["ground"][()]
            pred = hf["pred"][()]
            ppm = hf["ppm"][()]

        return input_spec, ground, pred, ppm
    @staticmethod
    def write_h5_pred_fit(save_file_path: str,
                                    input_spec: np.ndarray,
                                    ground: np.ndarray,
                                    pred: np.ndarray,
                                    ppm: np.ndarray) -> None:

        with h5py.File(save_file_path, 'w') as hf:
            hf.create_dataset('input_spec', data=input_spec)
            hf.create_dataset('ground', data=ground)
            hf.create_dataset('pred', data=pred)
            hf.create_dataset('ppm', data=ppm)

    @staticmethod
    def read_h5(filename: str) -> List[np.ndarray]:
        with h5py.File(filename, "r") as f:
            a_group_key = list(f.keys())[0]

            data = list(f[a_group_key])

            return data

    @staticmethod
    def read_h5_spectogram(filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with h5py.File(filename) as hf:
            gt_fids = hf["ground_truth_fids"][()]
            ppm = hf["ppm"][()]
            t = hf["t"][()]

        return gt_fids, ppm, t

    def read_h5_spectogram_track_2(filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with h5py.File(filename) as hf:
            ppm = hf["ppm"][()]
            t = hf["t"][()]
            target_spectra = hf["target_spectra"][()]
            transient_fids = hf["transient_fids"][()]

        return ppm, t, target_spectra, transient_fids

    def read_h5_spectogram_track_3(filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with h5py.File(filename) as hf:
            ppm_down = hf['data_2048']["ppm"][()]
            t_down = hf['data_2048']["t"][()]
            target_spectra_down = hf['data_2048']["target_spectra"][()]
            transient_fids_down = hf['data_2048']["transient_fids"][()]

            ppm_up = hf['data_4096']["ppm"][()]
            t_up = hf['data_4096']["t"][()]
            target_spectra_up = hf['data_4096']["target_spectra"][()]
            transient_fids_up = hf['data_4096']["transient_fids"][()]

        return ppm_down, t_down, target_spectra_down, transient_fids_down, ppm_up, t_up, target_spectra_up, transient_fids_up

    def read_h5_spectrogram_track_3(filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with h5py.File(filename) as hf:
            ppm_down = hf['data_2048']["ppm"][()]
            t_down = hf['data_2048']["t"][()]
            target_spectra_down = hf['data_2048']["target_spectra"][()]
            transient_fids_down = hf['data_2048']["transient_fids"][()]

            ppm_up = hf['data_4096']["ppm"][()]
            t_up = hf['data_4096']["t"][()]
            target_spectra_up = hf['data_4096']["target_spectra"][()]
            transient_fids_up = hf['data_4096']["transient_fids"][()]

        return ppm_down, t_down, target_spectra_down, transient_fids_down, \
            ppm_up, t_up, target_spectra_up, transient_fids_up

    def read_h5_spectrogram_test_sample_track_3(filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with h5py.File(filename) as hf:
            input_ppm_down = hf['data_2048']["ppm"][()]
            input_t_down = hf['data_2048']["t"][()]
            input_transients_down = hf['data_2048']["transient_fids"][()]

            input_ppm_up = hf['data_4096']["ppm"][()]
            input_t_up = hf['data_4096']["t"][()]
            input_transients_up = hf['data_4096']["transient_fids"][()]

        return input_transients_down, input_ppm_down, input_t_down, \
            input_transients_up, input_ppm_up, input_t_up

    @staticmethod
    def read_h5_spectogram_sample(filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with h5py.File(filename) as hf:
            fid_truth = hf["FID_truth"][()]
            # normalizing the fid_truth
            if len(fid_truth.shape) == 2:
                fid_truth = (fid_truth) / (np.max(np.abs(fid_truth), axis=1, keepdims=True))
            fid_noise = hf["FID_noise"][()]
            ppm = hf["ppm"][()]
            t = hf["t"][()]

        return fid_truth, fid_noise, ppm, t

    def read_h5_spectrogram_sample_track_2(filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with h5py.File(filename) as hf:
            ppm = hf["ppm"][()]
            t = hf["t"][()]
            target_spectra = hf["target_spectra"][()]
            transient_fids = hf["transient_fids"][()]

        return target_spectra, transient_fids, ppm, t

    @staticmethod
    def read_h5_spectrogram_sample_track_1(filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with h5py.File(filename) as hf:
            ppm = hf["ppm"][()]
            t = hf["t"][()]
            transients = hf["transients"][()]

        return transients, ppm, t

    @staticmethod
    def read_h5_spectrogram_test_sample_track_2(filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with h5py.File(filename) as hf:
            ppm = hf["ppm"][()]
            t = hf["t"][()]
            transients = hf['transient_fids'][()]

        return transients, ppm, t

    @staticmethod
    def read_h5_spectrogram_predict_sample(filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with h5py.File(filename) as hf:
            spectrogram_predict = hf["spectrogram_predict"][()].flatten()
            spectrogram_truth = hf["spectrogram_truth"][()].flatten()
            ppm = hf["ppm"][()].flatten()

        return spectrogram_predict, spectrogram_truth, ppm

    @staticmethod
    def write_h5_spectogram(filename: str,
                            fid_truth: np.ndarray,
                            fid_noise: np.ndarray,
                            ppm: np.ndarray,
                            t: np.ndarray) -> None:
        with h5py.File(filename, 'w') as hf:
            hf.create_dataset('FID_truth', data=fid_truth)
            hf.create_dataset('FID_noise', data=fid_noise)
            hf.create_dataset('ppm', data=ppm)
            hf.create_dataset('t', data=t)

    @staticmethod
    def write_h5_simulated_predict(filename: str,
                                   spectrogram_input: torch.Tensor,
                                   spectrogram_predict: torch.Tensor,
                                   spectrogram_truth: torch.Tensor,
                                   ppm: torch.Tensor,
                                   t: np.ndarray) -> None:
        with h5py.File(filename, 'w') as hf:
            hf.create_dataset('spectrogram_input', data=spectrogram_input.detach().numpy())
            hf.create_dataset('spectrogram_predict', data=spectrogram_predict.detach().numpy())
            hf.create_dataset('spectrogram_truth', data=spectrogram_truth.detach().numpy())
            hf.create_dataset('ppm', data=ppm.detach().numpy())
            hf.create_dataset('t', data=t)

    @staticmethod
    def write_h5_track1_predict_submission(filename: str,
                                           spectrogram_predict: np.ndarray,
                                           ppm: np.ndarray):
        """
        Save the results in the submission format
        Parameters:
            - results_spectra (np.array): Resulting predictions from test in format scan x spectral points
            - ppm (np.array): ppm values associataed with results, in same format
            - filename (str): name of the file to save results in, should end in .h5

        """
        with h5py.File(filename, "w") as hf:
            hf.create_dataset("result_spectra", spectrogram_predict.shape, dtype=float, data=spectrogram_predict)
            hf.create_dataset("ppm", ppm.shape, dtype=float, data=ppm)

    @staticmethod
    def write_h5_track2_predict_submission(filename: str,
                                           spectrogram_predict: np.ndarray,
                                           ppm: np.ndarray):
        """
        Save the results in the submission format
        Parameters:
            - results_spectra (np.array): Resulting predictions from test in format scan x spectral points
            - ppm (np.array): ppm values associataed with results, in same format
            - filename (str): name of the file to save results in, should end in .h5

        """
        with h5py.File(filename, "w") as hf:
            hf.create_dataset("result_spectra", spectrogram_predict.shape, dtype=float, data=spectrogram_predict)
            hf.create_dataset("ppm", ppm.shape, dtype=float, data=ppm)

    @staticmethod
    def write_h5_track3_predict_submission(filename: str,
                                           spectrogram_predict_down: np.ndarray,
                                           ppm_down: np.ndarray,
                                           spectrogram_predict_up: np.ndarray,
                                           ppm_up: np.ndarray):
        with h5py.File(filename, "w") as hf:
            hf.create_dataset("result_spectra_2048", spectrogram_predict_down.shape, dtype=float,
                              data=spectrogram_predict_down)
            hf.create_dataset("ppm_2048", ppm_down.shape, dtype=float, data=ppm_down)
            hf.create_dataset("result_spectra_4096", spectrogram_predict_up.shape, dtype=float,
                              data=spectrogram_predict_up)
            hf.create_dataset("ppm_4096", ppm_up.shape, dtype=float, data=ppm_up)

    @staticmethod
    def write_h5_invivo_predict(filename: str,
                                spectrogram_input: torch.Tensor,
                                spectrogram_predict: torch.Tensor,
                                ppm: torch.Tensor,
                                t: np.ndarray) -> None:
        with h5py.File(filename, 'w') as hf:
            hf.create_dataset('spectrogram_input', data=spectrogram_input.detach().numpy())
            hf.create_dataset('spectrogram_predict', data=spectrogram_predict.detach().numpy())
            hf.create_dataset('ppm', data=ppm.detach().numpy())
            hf.create_dataset('t', data=t)

    @staticmethod
    def read_h5_test_in_vivo(filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with h5py.File(filename) as hf:
            try:
                transients = hf["transients"][()]
            except:
                transients = hf["transient_fids"][()]
            ppm = hf["ppm"][()]
            t = hf["t"][()]

        return transients, ppm, t


def evaluate_matrix_confusion(model, loader, context_size, device: str):
    y_list = []
    y_hat_list = []

    with torch.no_grad():
        total = 0
        acc = 0
        for x, y in tqdm(loader):
            y_hat = model(x.to(device))
            y_hat = torch.argmax(y_hat, dim=1)

            y = y.to(device)
            y_hat = y_hat.to(device)

            y = np.squeeze(y.numpy())
            y_hat = np.squeeze(y_hat.numpy())

            y_list.append(y)
            y_hat_list.append(y_hat)

            acc += y[y == y_hat].shape[0]
            total += 1 * x.shape[0]

        mean_accuracy = acc / total

        y_array = np.concatenate(y_list)
        y_hat_array = np.concatenate(y_hat_list)

        cm = confusion_matrix(y_array, y_hat_array)
        return cm / context_size, mean_accuracy


def evaluate_batch_next(model, valid_loader, criterion, device):
    with torch.no_grad():
        total = 0
        acc = 0
        x, y = next(iter(valid_loader))
        y_hat = model(x.to(device))

        y = y.to(device)
        loss = criterion(y_hat, y[:, 0])

        # y_hat = torch.argmax(y_hat, dim=1)

        # acc += y[y.squeeze(1) == y_hat].shape[0]
        # total += 1 * x.shape[0]

        # mean_accuracy = acc / total
    return loss


def evaluate_batch(model, valid_loader, criterion, device):
    with torch.no_grad():
        total = 0
        acc = 0
        x, y = next(iter(valid_loader))
        y_hat = model(x.to(device))

        y = y.to(device)
        loss = criterion(y_hat, y[:, 0])

        y_hat = torch.argmax(y_hat, dim=1)

        acc += y[y.squeeze(1) == y_hat].shape[0]
        total += 1 * x.shape[0]

        mean_accuracy = acc / total

    return mean_accuracy, loss


def evaluate_batch(model, valid_loader, criterion, device):
    with torch.no_grad():
        total = 0
        acc = 0
        x, y = next(iter(valid_loader))
        y_hat = model(x.to(device))

        y = y.to(device)
        loss = criterion(y_hat, y[:, 0])

        y_hat = torch.argmax(y_hat, dim=1)

        acc += y[y.squeeze(1) == y_hat].shape[0]
        total += 1 * x.shape[0]

        mean_accuracy = acc / total

    return mean_accuracy, loss


def evaluate(model, loader, device: str) -> float:
    with torch.no_grad():
        total = 0
        acc = 0
        for x, y in tqdm(loader):
            y_hat = model(x.to(device))
            y_hat = torch.argmax(y_hat, dim=1)

            y = y.to(device)
            y_hat = y_hat.to(device)

            acc += y[y.squeeze(1) == y_hat].shape[0]
            total += 1 * x.shape[0]

        mean_accuracy = int(acc / total)

        return mean_accuracy


def train(model, train_dataloader, valid_dataloader, optimizer, criterion, lr,
          batch_size_train, name_models, num_epochs, save_scores, device):
    train_loss = 0
    list_loss_valid = []
    accuracy_list_valid = []
    list_loss_train = []

    for epoch in range(num_epochs):
        for x, y in tqdm(train_dataloader):
            optimizer.zero_grad()

            x_ids = x[0].to(device)
            attention_mask = x[1].to(device)
            labels = y.to(device)
            logits = model(x_ids, attention_mask=attention_mask).logits
            loss = criterion(logits, labels)

            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        list_loss_train.append(train_loss / len(train_dataloader))

        num_examples = len(valid_dataloader.dataset)

        accuracy_valid, loss_valid = evaluate(model, num_examples, valid_dataloader, criterion, device)

        last_loss = loss_valid

        save_scores.save_loss_in_file(loss_valid,
                                      f"valid_lr_{lr}_batch_size_{batch_size_train}_name_{name_models}.txt".replace("/",
                                                                                                                    ""))
        save_scores.save_loss_in_file(accuracy_valid,
                                      f"acc_lr_{lr}_batch_size_{batch_size_train}_name_{name_models}.txt".replace("/",
                                                                                                                  ""))

        save_scores.save_loss_in_file((train_loss / len(train_dataloader)),
                                      f"train_lr_{lr}_batch_size_{batch_size_train}_name_{name_models}.txt".replace("/",
                                                                                                                    ""))

        print(
            f"Epoch: {epoch} | accuracy {accuracy_valid}| valid loss {loss_valid} | exp_loss {torch.Tensor([last_loss])}")

        accuracy_list_valid.append(accuracy_valid)
        list_loss_valid.append(last_loss)

    return list_loss_train, accuracy_list_valid, list_loss_valid


def normalize_spectogram(X: np.ndarray) -> np.ndarray:
    X_norm = (X - X.min()) / (X.max() - X.min())
    return X_norm


def min_max_norm(Zxx: np.ndarray) -> np.ndarray:
    Zxx = np.abs(Zxx)
    Zxx = (Zxx - Zxx.min()) / (Zxx.max() - Zxx.min())
    return Zxx


def z_norm(Zxx: np.ndarray) -> np.ndarray:
    Zxx = np.abs(Zxx)
    Zxx_norm = (Zxx - Zxx.mean()) / Zxx.std()
    return Zxx_norm


def db_norm(Zxx: np.ndarray) -> np.ndarray:
    Zxx = np.abs(Zxx) ** 2
    Zxx_dcb = 20 * np.log(Zxx / np.amax(Zxx))
    return Zxx_dcb


def get_fid_params(params: dict) -> np.ndarray:
    params.pop("A_Ace", None)
    params_array = []
    for key in params.keys():
        params_array.append(params[key])

    return np.asarray(params_array)


def transform_spectrogram_2D(FID, t, window_size=256, hope_size=64, window='hann', nfft=None):
    # calculating the overlap between the windows
    noverlap = window_size - hope_size

    # checking for the NOLA criterion
    if not signal.check_NOLA(window, window_size, noverlap):
        raise ValueError("signal windowing fails Non-zero Overlap Add (NOLA) criterion; "
                         "STFT not invertible")
    # fs = 1 / (t[1] - t[0])
    fs = 1 / t
    # computing the STFT
    _, _, Zxx = signal.stft(np.real(FID), fs=fs, nperseg=window_size, noverlap=noverlap,
                            return_onesided=False, nfft=nfft)

    # Zxx = min_max_norm(Zxx)
    # Zxx = z_norm(Zxx)
    # Zxx = db_norm(Zxx)
    # Zxx = np.abs(Zxx)
    # Zxx = np.abs(Zxx) / (np.max(np.abs(Zxx)))
    return Zxx


def transform_spectrogram_2D_complete(FID, t, window_size=256, hope_size=64, window='hann', nfft=None):
    # calculating the overlap between the windows
    noverlap = window_size - hope_size

    # checking for the NOLA criterion
    if not signal.check_NOLA(window, window_size, noverlap):
        raise ValueError("signal windowing fails Non-zero Overlap Add (NOLA) criterion; "
                         "STFT not invertible")
    fs = 1 / t
    # computing the STFT
    _, _, Zxx = signal.stft(FID, fs=fs, nperseg=window_size, noverlap=noverlap,
                            return_onesided=False, nfft=nfft)
    # Zxx = np.concatenate([np.split(Zxx, 2)[1], np.split(Zxx, 2)[0]])
    return Zxx


def open_txt_samples(dir_data: str) -> List[str]:
    with open(dir_data, 'r') as f:
        read_content = f.read()

    list_generate_samples = read_content.split('\n')
    return list_generate_samples


def write_txt_samples(samples: list, dir_data: str) -> None:
    with open(dir_data, 'w') as f:
        f.write('\n'.join(samples))
    return


def split_any_datasets(dataset_dir, seed_split, split_train):
    random.seed(seed_split)

    elements_in_dataset = os.listdir(dataset_dir)

    len_elements_in_dataset = len(elements_in_dataset)

    n_rands = random.sample(range(0, len_elements_in_dataset), len_elements_in_dataset)

    select_split = int(split_train * len_elements_in_dataset)

    idx_samples_train = n_rands[:select_split]
    idx_samples_valid = n_rands[select_split:]

    samples_train = [f"{dataset_dir}/{elements_in_dataset[idx]}" for idx in idx_samples_train]
    samples_valid = [f"{dataset_dir}/{elements_in_dataset[idx]}" for idx in idx_samples_valid]

    print(f"length train: {len(samples_train)}")
    print(f"length validation: {len(samples_valid)}")

    return samples_train, samples_valid

def random_dataset_split_track_3(**kargs: dict):
    dataset_dir_train_2048 = kargs['path_2048']
    dataset_dir_train_4096 = kargs['path_4096']
    split_train = kargs['split_train']
    seed_split = kargs['seed_split']

    samples_train_2048, samples_valid_2048 = split_any_datasets(dataset_dir_train_2048,
                                                                split_train,
                                                                seed_split)

    samples_train_4096, samples_valid_4096 = split_any_datasets(dataset_dir_train_4096,
                                                                split_train,
                                                                seed_split)

    write_txt_samples(samples_train_2048, 'split_train_txt/train_2048.txt')
    write_txt_samples(samples_valid_2048, 'split_train_txt/valid_2048.txt')

    write_txt_samples(samples_train_4096, 'split_train_txt/train_4096.txt')
    write_txt_samples(samples_valid_4096, 'split_train_txt/valid_4096.txt')

    return samples_train_2048, samples_valid_2048, samples_train_4096, samples_valid_4096


def random_split_dataset(**kargs: dict) -> tuple[List[str], List[str]]:
    dataset_dir = kargs['path']
    split_train = kargs['split_train']
    seed_split = kargs['seed_split']

    elements_in_dataset = os.listdir(dataset_dir)

    if "sorted_data" in kargs:
        sorted_data = kargs['sorted_data']

        select_split = split_train * 120

        elements_in_dataset_sorted = sorted(elements_in_dataset)
        # list_dataset, seed_split, pacients, train_elements
        if "k_fold" in kargs:
            pacients_all = kargs["k_fold"]['pacients_all']
            qtd_k_fold_valid = kargs["k_fold"]['qtd_k_fold_valid']

            samples_train, samples_valid = KFold.k_fold_split(elements_in_dataset_sorted,
                                                              seed_split,
                                                              pacients_all,
                                                              qtd_k_fold_valid)

        else:
            samples_train = elements_in_dataset_sorted[:select_split]
            samples_valid = elements_in_dataset_sorted[select_split:]

        samples_train = [f"{dataset_dir}/{elem}" for elem in samples_train]
        samples_valid = [f"{dataset_dir}/{elem}" for elem in samples_valid]

        print(f"length train: {len(samples_train)}")
        print(f"length validation: {len(samples_valid)}")

        write_txt_samples(samples_train, 'split_train_txt/train.txt')
        write_txt_samples(samples_valid, 'split_train_txt/valid.txt')

        return samples_train, samples_valid

    random.seed(seed_split)

    len_elements_in_dataset = len(elements_in_dataset)

    n_rands = random.sample(range(0, len_elements_in_dataset), len_elements_in_dataset)

    select_split = int(split_train * len_elements_in_dataset)

    idx_samples_train = n_rands[:select_split]
    idx_samples_valid = n_rands[select_split:]

    samples_train = [f"{dataset_dir}/{elements_in_dataset[idx]}" for idx in idx_samples_train]
    samples_valid = [f"{dataset_dir}/{elements_in_dataset[idx]}" for idx in idx_samples_valid]

    print(f"length train: {len(samples_train)}")
    print(f"length validation: {len(samples_valid)}")

    write_txt_samples(samples_train, 'split_train_txt/train.txt')
    write_txt_samples(samples_valid, 'split_train_txt/valid.txt')

    return samples_train, samples_valid


def plot_spectrums_variation(reconstructed_signals):
    # each row of the reconstructed_signals correspond to the individual reconstructed signal
    # Calculate the mean and standard deviation of the signals
    mean = reconstructed_signals.mean(axis=0)
    std = reconstructed_signals.std(axis=0)
    # Defining Region of interest and indexes
    min_ppm = 2.5
    max_ppm = 4
    max_ind = np.amax(np.where(ppm >= min_ppm))
    min_ind = np.amin(np.where(ppm <= max_ppm))
    # cropping to region of interest
    mean_crop = mean[min_ind:max_ind]
    # mean_crop = (mean_crop-mean_crop.min())/(mean_crop.max()-mean_crop.min())
    std_crop = std[min_ind:max_ind]
    # std_crop = (std_crop -std_crop.min())/(std_crop.max()-std_crop.min())
    ppm_crop = ppm[min_ind:max_ind]

    # plotting x and y spectra to and highlighting region of difference
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))

    ax[0].plot(ppm, mean, label='Mean')
    ax[0].invert_xaxis()
    ax[0].set_xlabel("ppm")
    ax[0].set_yticks([])
    ax[0].set_title("Spectrums' Variation")
    ax[0].fill_between(ppm, mean - std, mean + std, alpha=0.2, label='Standard Deviation')
    ax[0].legend(loc='upper right')

    ax[1].plot(ppm_crop, mean_crop, label='Mean')
    ax[1].fill_between(ppm_crop, mean_crop - std_crop, mean_crop + std_crop, alpha=0.2, label='Standard Deviation')
    ax[1].invert_xaxis()
    ax[1].set_xlabel("ppm")
    ax[1].set_yticks([])
    ax[1].set_title("Zoom - Region of GABA (2.5ppm-4ppm)")

    plt.legend()
    plt.show()


def calculate_ppm_shift_range(ppm):
    # find the size of the spectra
    len_spectra = ppm.shape[0]
    # find the left corner of the spectra:
    left_corner = np.amin(np.where(ppm <= 10.8))
    # find the right corner of the spectra:
    right_corner = np.amax(np.where(ppm >= 0))
    # define maximum left shift
    maximum_left_shift = left_corner - 1
    # define maximum right shift
    maximum_right_shift = len_spectra - right_corner
    # calculate the shift range
    shift_range = np.arange(-maximum_left_shift, maximum_right_shift + 1)
    return shift_range


def padd_zeros_spectrogram(X_dcb, output_shape=(224, 224)):
    matrix = X_dcb
    pad_width = ((0, output_shape[0] - matrix.shape[0]), (0, output_shape[1] - matrix.shape[1]))
    padded_matrix = np.pad(matrix, pad_width, mode="constant")
    return padded_matrix


def combine_samples_for_triplet(filenames: list, filter_int: None):
    #    filenames = sorted([filename for filename in filenames if filename.startswith("sample_")])

    if filter_int:
        filenames = filenames[:filter_int]

    permutacoes = itertools.permutations(filenames, 2)
    permutacoes_distintas = filter(lambda x: len(set(x)) == len(x), permutacoes)

    # triplets = [(f"{directory}/{samples[0]}",
    #              f"{directory}/{samples[1]}",
    #              f"{directory}/{samples[2]}") for samples in permutacoes_distintas]

    triplets = permutacoes_distintas

    n = len(filenames)

    n_triplets = n ** 2 - n

    return triplets, n_triplets

def calculate_fqn(spec, residual, ppm):

    dt_max_ind, dt_min_ind = np.amax(np.where(ppm >= 9.8)), np.amin(np.where(ppm <= 10.8))
    noise_var = np.var(spec[dt_min_ind:dt_max_ind])
    residual_var = np.var(residual)

    fqn = residual_var/noise_var
    return fqn

def track_validation_split(track_data, validation_prop):
    val_size = int(validation_prop * len(track_data))
    np.random.shuffle(track_data)
    track_val = track_data[:val_size]
    track_train = track_data[val_size:]
    return track_train, track_val

def reset_datasets():

    track1_folder = "../tests/dataset_1000"
    track2_folder = "../tests/dataset_5000"
    track3_folder = "../tests/dataset_10000"
    test_folder = "../tests/test"

    track_folder = [track1_folder, track2_folder, track3_folder]

    for folder in track_folder:
        for subfolder in os.listdir(folder):
            for file in os.listdir(os.path.join(folder,subfolder)):
                os.remove(os.path.join(folder, subfolder, file))

    for file in os.listdir(test_folder):
        os.remove(os.path.join(test_folder, file))

class NormalizeData:
    def normalize(self, arr, method):
        if method == "min-max":
            return self.min_max_normalize(arr)
        elif method == "z_norm":
            return self.z_score_normalize(arr)

    def min_max_normalize(self, arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        normalized_arr = (arr - min_val) / (max_val - min_val)
        return normalized_arr

    def z_score_normalize(self, arr):
        mean = np.mean(arr)
        std_dev = np.std(arr)
        normalized_arr = (arr - mean) / std_dev
        return normalized_arr


def calculate_r_squared(predicted, ground_truth):
    mean_ground_truth = np.mean(ground_truth)
    ss_total = np.sum((ground_truth - mean_ground_truth) ** 2)
    ss_residual = np.sum((ground_truth - predicted) ** 2)

    r_squared = 1 - (ss_residual / ss_total)
    return r_squared