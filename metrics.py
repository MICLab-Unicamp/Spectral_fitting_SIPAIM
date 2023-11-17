"""
Maintainer: Mateus Oliveira (mateus.oliveira@icomp.ufam.edu.br)
        Gabriel Dias (g172441@dac.unicamp.br)
        Marcio Almeida (m240781@dac.unicamp.br)
"""

import os
import numpy as np
import wandb
from utils import ReadDatasets
import pandas as pd
from tqdm import tqdm


class GenerateMetricsPandas:
    def __init__(self):
        self.data = []
        self.table = wandb.Table(columns=["mse", "snr", "linewidth", "shape_score"])

    def __call__(self, spectrogram_predict, spectrogram_truth, ppm):
        metrics = calculate_metrics(np.expand_dims(spectrogram_predict, axis=0),
                                    np.expand_dims(spectrogram_truth, axis=0),
                                    np.expand_dims(ppm, axis=0))

        metrics["mse"] = metrics["mse"][0]
        metrics["snr"] = metrics["snr"][0]
        metrics["linewidth"] = metrics["linewidth"][0]
        metrics["shape_score"] = metrics["shape_score"][0]

        self.data.append(metrics)

    def wandb_upload_table(self, spectrogram_predict, spectrogram_truth, ppm, run):
        """
        "mse": mse,
        "snr": snr,
        "linewidth": linewidth,
        "shape_score": shape_score
        """

        metrics = calculate_metrics(np.expand_dims(spectrogram_predict, axis=0),
                                    np.expand_dims(spectrogram_truth, axis=0),
                                    np.expand_dims(ppm, axis=0))
        # "mse", "snr", "linewidth", "shape_score"
        self.table.add_data(metrics["mse"],
                            metrics["snr"],
                            metrics["linewidth"],
                            metrics["shape_score"])

    def save_log_upload(self, run):
        run.log({"predictions_table": self.table}, commit=False)

    def save_dataframe(self, list_samples, dir_dataframe="artifacts/metrics_dataframe.csv"):
        df = pd.DataFrame.from_dict(self.data)
        df["file"] = list_samples
        df = df[["file", "mse", "snr", "linewidth", "shape_score"]]
        # saving df
        df.to_csv(dir_dataframe, sep=";")

    def _upload(self, df, run, name):
        run.log({f"{name}_mean": df["mean"]})
        run.log({f"{name}_std": df["std"]})
        run.log({f"{name}_count": df["count"]})

    def relevant_metrics(self, run):
        df = pd.read_csv("artifacts/metrics_dataframe.csv", delimiter=";")
        self._upload(df["mse"].describe(), run, "mse")
        self._upload(df["snr"].describe(), run, "snr")
        self._upload(df["linewidth"].describe(), run, "linewidth")
        self._upload(df["shape_score"].describe(), run, "shape_score")


def generate_metrics_dataframe_2(list_samples):
    data = []
    for filename in tqdm(list_samples):
        ### load data
        spectrogram_predict, spectrogram_truth, ppm = ReadDatasets.read_h5_spectrogram_predict_sample(filename)
        # calculate metrics
        metrics = calculate_metrics(np.expand_dims(spectrogram_predict, axis=0),
                                    np.expand_dims(spectrogram_truth, axis=0),
                                    np.expand_dims(ppm, axis=0))

        metrics = {k: v[0] for k, v in metrics.items()}
        data.append(metrics)
    df = pd.DataFrame.from_dict(data)
    df["file"] = list_samples
    df = df[["file", "mse", "snr", "linewidth", "shape_score"]]
    # saving df
    df.to_csv("artifacts/metrics_dataframe.csv", sep=";")


def generate_metrics_dataframe(list_samples):
    data = []
    for filename in tqdm(list_samples):
        ### load data
        spectrogram_predict, spectrogram_truth, ppm = ReadDatasets.read_h5_spectrogram_predict_sample(filename)
        # calculate metrics
        metrics = calculate_metrics(np.expand_dims(spectrogram_predict, axis=0),
                                    np.expand_dims(spectrogram_truth, axis=0),
                                    np.expand_dims(ppm, axis=0))

        metrics = {k: v[0] for k, v in metrics.items()}
        data.append(metrics)
    df = pd.DataFrame.from_dict(data)
    df["file"] = list_samples
    df = df[["file", "mse", "snr", "linewidth", "shape_score"]]
    # saving df
    df.to_csv("artifacts/metrics_dataframe.csv", sep=";")


def calculate_metrics(x, y):
    x = np.real(x)
    y = np.real(y)

    # Creating the lists that are going to be returned in the output dictionary
    mse = []
    mae = []
    snr = []
    linewidth = []
    shape_score = []

# Iterating over the rows (samples) of the input arrays and calculating their respective metrics
    for i in range(x.shape[0]):
        # calculating the mse
        mse.append(calculate_mse(x[i, :], y[i, :]))
        mae.append(calculate_mae(x[i, :], y[i, :]))

    output = {
        "mse": mse,
        "mae": mae
    }

    return output

def calculate_mae(x, y):

    mse = np.abs(x-y).mean()

    return mse

def calculate_mse(x, y):

    mse = np.square(x-y).mean()

    return mse


def calculate_snr(x, ppm):
    """
    This function calculates the GABA SNR metric for ONE sample.
    ----------
    x : numpy array
        Testing reconstructed spectra sample.
    ppm : numpy array
        ppm array sample.

    Returns
    -------
    snr : float
        GABA SNR.

    """

    # selecting indexes of regions of interest
    # ppm = ppm.flatten()
    gaba_max_ind, gaba_min_ind = np.amax(np.where(ppm >= 2.8)), np.amin(np.where(ppm <= 3.2))
    dt_max_ind, dt_min_ind = np.amax(np.where(ppm >= 9.8)), np.amin(np.where(ppm <= 10.8))

    # extracting region peak
    max_peak = x[gaba_min_ind:gaba_max_ind].max()

    # calculating fitted standard deviation of noise region
    dt = np.polyfit(ppm[dt_min_ind:dt_max_ind], x[dt_min_ind:dt_max_ind], 2)
    sizeFreq = ppm[dt_min_ind:dt_max_ind].shape[0]
    stdev_Man = np.sqrt(
        np.sum(np.square(np.real(x[dt_min_ind:dt_max_ind] - np.polyval(dt, ppm[dt_min_ind:dt_max_ind])))) / (
                sizeFreq - 1))

    # calculate snr
    snr = np.real(max_peak) / (2 * stdev_Man)

    # return snr
    return snr


def calculate_linewidth(x, ppm):
    """
    This function calculates the Linewidth for ONE sample.
    
    Linewidth, or full width at half maximum (FWHM) is defined as the difference between the ppm value of the 
    farthest points at either side of the peak which are greater than half of the peak's height.

    Only the region of the GABA peak is considered and, in order to determine the height, a min-max 
    normalization is performed.

    Parameters
    ----------
    x : numpy array
        Testing reconstructed spectra sample.
    ppm : numpy array
        ppm array sample.

    Returns
    -------
    linewidth : float
        The linewidth metric

    """

    # selecting indexes of gaba peak region
    gaba_max_ind, gaba_min_ind = np.amax(np.where(ppm >= 2.8)), np.amin(np.where(ppm <= 3.2))

    # cropping spectrum for only gaba region and performin min-max normalization
    spec = x[gaba_min_ind:gaba_max_ind]

    ##normalizing spec
    spec = (spec - spec.min()) / (spec.max() - spec.min())

    # selecting max point index
    max_peak = spec.max()
    ind_max_peak = np.argmax(spec)

    # selecting highest ppm value with point above half the peak value
    try:
        left_side = spec[:ind_max_peak]
        left_ind = np.amin(np.where(left_side > max_peak / 2)) + gaba_min_ind
        left_ppm = ppm[left_ind]
    except:
        left_side = spec[:ind_max_peak + 1] # possible solution
        left_ind = np.amin(np.where(left_side > max_peak / 2)) + gaba_min_ind
        left_ppm = ppm[left_ind]

    # selecting lowest ppm value with point above half the peak value
    right_side = spec[ind_max_peak:]
    right_ind = np.amax(np.where(right_side > max_peak / 2)) + gaba_min_ind + ind_max_peak
    right_ppm = ppm[right_ind]

    # calculating the linewidth
    linewidth = left_ppm - right_ppm

    return linewidth


def calculate_shape_score(x, y, ppm):
    """
    This function calculates the shape score for ONE sample.
    
    The shape score is a metric to determine if the shape of the reconstructed peak is according to the expected. 
    This is especially interesting when considering the GLX double peak at 3.75ppm.

    The definition of the shape score is the weighted average of the correlation between the ground-truth 
    spectrum and the model reconstruction, using min-max normalization and only considering the region close to 
    each peak.

    The regions were determined as from 2.8ppm to 3.2ppm for GABA, and from 3.6ppm to 3.9ppm for GLX, and 
    their weights were 0.6 and 0.4, respectively.

    Parameters
    ----------
    x : numpy array
        Testing reconstructed spectra sample.
    y : numpy array
        Reference reconstructed spectra sample.
    ppm : numpy array
        ppm array sample.

    Returns
    -------
    shape score : float
        The shape score metric
    """
    # selecting indexes of region of interest
    gaba_max_ind, gaba_min_ind = np.amax(np.where(ppm >= 2.8)), np.amin(np.where(ppm <= 3.2))
    glx_max_ind, glx_min_ind = np.amax(np.where(ppm >= 3.6)), np.amin(np.where(ppm <= 3.9))

    # cropping reconstruction to gaba region and performing min-max normalization
    gaba_spec_x = x[gaba_min_ind:gaba_max_ind]
    gaba_spec_x = (gaba_spec_x - gaba_spec_x.min()) / (gaba_spec_x.max() - gaba_spec_x.min())

    # cropping ground-truth spectrum to gaba region and performing min-max normalization
    gaba_spec_y = y[gaba_min_ind:gaba_max_ind]
    gaba_spec_y = (gaba_spec_y - gaba_spec_y.min()) / (gaba_spec_y.max() - gaba_spec_y.min())

    # cropping reconstruction to GLX region and performing min-max normalization
    glx_spec_x = x[glx_min_ind:glx_max_ind]
    glx_spec_x = (glx_spec_x - glx_spec_x.min()) / (glx_spec_x.max() - glx_spec_x.min())

    # cropping ground-truth spectrum to GLX region and performing min-max normalization
    glx_spec_y = y[glx_min_ind:glx_max_ind]
    glx_spec_y = (glx_spec_y - glx_spec_y.min()) / (glx_spec_y.max() - glx_spec_y.min())

    # calculating correlations
    gaba_corr = np.corrcoef(gaba_spec_x, gaba_spec_y)[0, 1]
    glx_corr = np.corrcoef(glx_spec_x, glx_spec_y)[0, 1]

    # Shape Score = 0.6*GABA_corr + 0.4 GLX_corr
    return (0.6 * gaba_corr + 0.4 * glx_corr)
