[![MICLab](https://img.shields.io/badge/MICLab-Spectral%20Fitting%20SIPAIM-blue)](https://github.com/MICLab-Unicamp/Spectral_fitting_SIPAIM)

## Project Overview

Magnetic Resonance Spectroscopy (MRS) is an efficient non-invasive solution for exploration of the biochemical composition of brain tissue, with potential for verification and follow-up of a variety of diseases visible or not through neuroimaging. This repository contains the code for the paper [Magnetic Resonance Spectroscopy Fitting through Spectrogram and Vision Transformers presented at the 2023 19th International Symposium on Medical Information Processing and Analysis (SIPAIM)](https://ieeexplore.ieee.org/abstract/document/10373415?casa_token=e2zW2Gf-J6cAAAAA:2OXdSVBDcxuJ1Tqk0DJy4DPev5gS-V5fleX9p-BbQ8nxcxQEPl49B8A1TTeeQcgX3h1vxrItszc). 

This innovative approach combines spectrogram representations of MRS data with the Vision Transformer (ViT) model to perform linear combination fitting for metabolite quantification in MRS. Leveraging the image recognition capabilities of the ViT model and the dual use of time and frequency domains, this method promises to enhance MRS techniques, demonstrating promising results for the MRS fitting task. The model receives a spectrogram as input, consisting of three channels: real, imaginary, and magnitude. It outputs amplitudes for 20 metabolites and one macromolecule, as well as values for damping, frequency, and phase shifts.

![Residual Model Output](https://github.com/MICLab-Unicamp/Spectral_fitting_SIPAIM/assets/91618118/348e2b63-d9b4-42d0-856b-564b822b9c81)
*Figure 1: Model architecture.*


## Credits

This project has adapted the robust and flexible Spectro-ViT training framework [![GitHub](https://img.shields.io/badge/MICLab-Spectro_ViT-red)](https://github.com/MICLab-Unicamp/Spectro-ViT)
, which has been instrumental in the development of this research and repository.

#### Features of the Spectro-ViT Training Framework:
- **Training Pipeline**: Provides a complete training pipeline and a YAML configuration file.
- **PyTorch Implementation**: Built using the widely-used PyTorch library.
- **Weights & Biases Integration**: Supports integration with [Weights & Biases](https://wandb.ai/site) for real-time training monitoring and results logging.
- **On-the-Fly Validation**: Features real-time validation with visualization during training.
- **Customizable Framework**: Allows users to modify various aspects of the model and training process to meet specific research needs and objectives through an easy-to-use YAML configuration file.

We also encourage you to explore the Spectro-ViT GitHub repository and utilize this framework for your MRS research and development needs.

## Training Configuration File Details

The model's training and evaluation behaviors are fully configurable through a YAML configuration file. Below, you will find detailed explanations of key sections within this file:

### Weights & Biases (wandb) Configuration

- `activate`: Enables or disables integration with Weights & Biases. Set to `True` for tracking and visualizing metrics during training.
- `project`: Specifies the project name under which the runs should be logged in Weights & Biases.
- `entity`: Specifies the user or team name under Weights & Biases where the project is located.

### Saving/Reloading Configuration

- **Current Model**
  - `save_model`: Enable or disable the saving of model weights.
  - `model_dir`: Directory to save the model weights.
  - `model_name`: Name under which to save the model weights.

- **Reload from Existing Model**
  - `activate`: Enable reloading weights from a previously saved model to continue training.
  - `model_dir`: Directory from where the model weights should be loaded.
  - `model_name`: Name of the model weights file to be loaded.

### Model Configuration
- **Model**
  - `Model Class Name`: Specifies the model class. Example: `TimmModelSpectrogram`.
    - `Instantiation parameters of the model class`.

### Training Parameters

- `epochs`: Number of training epochs.
- `optimizer`: Configuration for the optimizer. Example:
  - `Adam`: Specifies using the Adam optimizer.
    - `lr`: Learning rate for the optimizer.

- `loss`: Specifies the loss function used for training. Example: `MAELoss`.

- `lr_scheduler`: Configuration for the learning rate scheduler. Example:
  - `activate`: Enable or disable the learning rate scheduler.
  - `scheduler_type`: Type of scheduler, e.g., `cosineannealinglr`.
  - `info`: Parameters for the scheduler, such as `T_max` (maximum number of iterations) and `eta_min` (minimum learning rate).

### Dataset Configuration

The following configuration parameters are designed to instantiate the Dataset class:

- **Training Dataset**
  - `Dataset Class Name`: Specifies the class used for managing the training dataset. Example: `DatasetBasisSetOpenTransformer3ChNormalize`
    - `path_data`: Directory containing the training data.
    - `norm`: Normalization to be applied to the spectrogram channels.

- **Validation Dataset**
  - `Dataset Class Name`: Specifies the class used for managing the validation dataset. Example: `DatasetBasisSetOpenTransformer3ChNormalize`
    - `path_data`: Directory containing the validation data.
    - `norm`: Normalization to be applied to the spectrogram channels.

- **Valid on the Fly**
  - `activate`: Enables saving plots that help analyze the model's performance on validation data during training.
  - `save_dir_path`: Directory to save these plots.

- **Test Dataset**
  - `Dataset Class Name`: Specifies the class used for managing the test dataset. Example: `DatasetBasisSetOpenTransformer3ChNormalize`
    - `path_data`: Directory containing the validation data.
    - `norm`: Normalization to be applied to the spectrogram channels.

## Model Evaluation

The `evaluate.py` script evaluates the performance of the trained MRS fitting model. It parses command-line arguments, reads test dataset settings from the YAML file, and loads model weights. The script then processes the test dataset, generating predictions for MRS data and calculating fitting metrics such as MSE, MAE, MAPE, R2, and FQN for each sample. It compares the predicted spectrum (generated with the predicted fitting parameters and basis set) to the actual spectrum, calculates residuals, and visualizes the results in plots. Additionally, the script computes average fitting metrics across all samples and outputs these along with coefficients' metrics to CSV files for further analysis.

![Fitting Evaluation](https://github.com/MICLab-Unicamp/Spectral_fitting_SIPAIM/assets/91618118/0633399a-5a45-416d-a7b5-76afd49ec8c5)

*Figure 2: Example of one of the plots generated by the evaluation script.*

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/MICLab-Unicamp/Spectral_fitting_SIPAIM.git
   ```
2. Navigate to the project directory:

   ```bash
   cd Spectral_fitting_SIPAIM
   ```
3. Check the Python version in requirements.txt and install the required dependencies:

    ```bash
   pip install -r requirements.txt
   ```

## Training Example 

Here's an example of how you might train the model using the provided configuration file:

```bash
python main.py configs/config_model.yaml
```

## Evaluation Example

Here's an example of how you might evaluate the trained model:

```bash
python evaluate.py configs/config_model.yaml models_h5/TimmModelSpectrogram.pt
```
## Developers

- [Gabriel Dias](https://github.com/gsantosmdias)
- [Mateus Oliveira](https://github.com/oliveiraMats2)
- [Márcio Almeida](https://github.com/marciovinialmeida)
  
## Citation

If you use our model and code in your research please cite:

    @inproceedings{almeida2023magnetic,
      title={Magnetic Resonance Spectroscopy Fitting through Spectrogram and Vision Transformers},
      author={Almeida, M{\'a}rcio Vin{\'\i}cius De Jesus and Dias, Gabriel Santos Martins and Da Silva, Mateus Oliveira and Rittner, Let{\'\i}cia},
      booktitle={2023 19th International Symposium on Medical Information Processing and Analysis (SIPAIM)},
      pages={1--5},
      year={2023},
      organization={IEEE}
    }
