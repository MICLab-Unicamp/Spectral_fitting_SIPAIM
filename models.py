import torch.nn as nn
import timm
import torch
import torch.nn.functional as F
from utils import set_device
from einops import rearrange

DEVICE = set_device()


def get_n_out_features(encoder, img_size, nchannels):
    out_feature = encoder(torch.randn(1, nchannels, img_size[0], img_size[1]))
    n_out = 1
    for dim in out_feature[-1].shape:
        n_out *= dim
    return n_out


class SpectrogramModelPreview(nn.Module):
    def __init__(self, **kwargs):
        super(SpectrogramModelPreview, self).__init__()

        self.conv_1 = nn.Conv2d(
            in_channels=kwargs["in_channels"],
            out_channels=kwargs["out_channels"],
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )

        self.batch_normalize = nn.BatchNorm2d(kwargs["out_channels"])
        self.relu = nn.ReLU()
        self.linear = nn.Linear(42570, 2048)

    def forward(self, spectrum):
        x = self.conv_1(spectrum)
        x = self.batch_normalize(x)
        x = self.relu(x)

        embedding = x.view((-1, x.shape[1] * x.shape[2] * x.shape[3]))

        x = torch.relu(self.linear(embedding))
        return x


def convBlock(ni, no):
    return nn.Sequential(
        nn.Conv2d(ni, no, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(no),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )


class SpectrogramModel(nn.Module):
    def __init__(self, **kwargs):
        super(SpectrogramModel, self).__init__()

        self.features = nn.Sequential(
            convBlock(kwargs["in_channels"], kwargs["out_channels"]),
            convBlock(10, 128),
            convBlock(128, 200),
            # convBlock(32, 64)
        )
        self.linears = nn.Sequential(nn.Flatten(),
                                     nn.Linear(17000, 5000),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(5000, 2048)
                                     )

    def forward(self, spectrum):
        x = self.features(spectrum)

        x = self.linears(x)
        return x


class ModelConv2deep:
    def __init__(self, **kwargs):
        super(ModelConv2deep, self).__init__()

        self.conv_1 = nn.Conv2d(
            in_channels=kwargs["in_channels"],
            out_channels=kwargs["out_channels"],
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )

        self.batch_normalize = nn.BatchNorm2d(kwargs["out_channels"])
        self.relu = nn.ReLU()
        self.linear = nn.Linear(42570, 2048)

    def forward(self, spectrum):
        x = self.conv_1(spectrum)
        x = self.batch_normalize(x)
        x = self.relu(x)

        embedding = x.view((-1, x.shape[1] * x.shape[2] * x.shape[3]))

        x = self.linear(embedding)
        return x


class TimmSimpleCNN(nn.Module):
    def __init__(self, network: str,
                 image_size: int,
                 nchannels: int,
                 transformers: bool = False,
                 pretrained: bool = False,
                 num_classes: int = 0,
                 features_only: bool = True):

        super().__init__()
        if transformers:
            model_creator = {'model_name': network,
                             "pretrained": pretrained,
                             "num_classes": num_classes}
        else:
            model_creator = {'model_name': network,
                             "pretrained": pretrained,
                             "features_only": features_only}

        self.encoder = timm.create_model(**model_creator)

        self.dimensionality_reductor = None

        for param in self.encoder.parameters():
            param.requires_grad = True

        n_out = get_n_out_features(self.encoder, image_size, nchannels)

        if transformers:
            self.dimensionality_up_sampling = nn.Sequential(
                nn.Linear(n_out, 512), nn.ReLU(inplace=True),
                nn.Linear(512, 256), nn.ReLU(inplace=True),
                nn.Linear(256, 128), nn.ReLU(inplace=True),
                nn.Linear(128, 24)
                #nn.Linear(512, 1024), nn.ReLU(inplace=True),
                #nn.Linear(1024, 2048)
            )
        else:
            self.dimensionality_up_sampling = nn.Sequential(
                nn.Flatten(),
                nn.Linear(n_out, 512), nn.ReLU(inplace=True),
                nn.Linear(512, 256), nn.ReLU(inplace=True),
                nn.Linear(256, 128), nn.ReLU(inplace=True),
                nn.Linear(128, 24)
            )

    def forward(self, signal_input):
        #output = self.encoder(signal_input)[-1]
        output = self.encoder(signal_input)
        output = self.dimensionality_up_sampling(output)

        return output


class TimmTripletModel(nn.Module):
    def __init__(self,
                 network: str,
                 image_size: int,
                 nchannels: int,
                 transformers: bool = False,
                 pretrained: bool = False,
                 num_classes: int = 0,
                 features_only: bool = True):
        super(TimmTripletModel, self).__init__()

        self.backbone = TimmSimpleCNN(network,
                                      image_size,
                                      nchannels,
                                      transformers,
                                      pretrained,
                                      num_classes,
                                      features_only)

    def forward(self, inputs):
        input_anchor, input_pos, input_neg = inputs

        input_anchor = input_anchor.to(DEVICE)
        input_pos = input_pos.to(DEVICE)
        input_neg = input_neg.to(DEVICE)

        output_anchor = self.backbone.encoder(input_anchor)
        output_pos = self.backbone.encoder(input_pos)
        output_neg = self.backbone.encoder(input_neg)

        output = output_anchor, output_pos, output_neg

        return output


class TimmSimpleCNNTrack3(nn.Module):
    def __init__(self, network: str,
                 image_size: int,
                 nchannels: int,
                 transformers: bool = False,
                 pretrained: bool = False,
                 num_classes: int = 0,
                 features_only: bool = True):

        super().__init__()
        if transformers:
            model_creator = {'model_name': network,
                             "pretrained": pretrained,
                             "num_classes": num_classes}
        else:
            model_creator = {'model_name': network,
                             "pretrained": pretrained,
                             "features_only": features_only}

        self.encoder = timm.create_model(**model_creator)

        self.dimensionality_reductor = None

        for param in self.encoder.parameters():
            param.requires_grad = True

        n_out = get_n_out_features(self.encoder, image_size, nchannels)

        if transformers:
            self.dimensionality_up_sampling = nn.Sequential(
                nn.Linear(n_out, 512), nn.ReLU(inplace=True),
                nn.Linear(512, 1024), nn.ReLU(inplace=True),
                nn.Linear(1024, 2048), nn.ReLU(inplace=True),
                nn.Linear(2048, 4096)
            )
        else:
            self.dimensionality_up_sampling = nn.Sequential(
                nn.Flatten(),
                nn.Linear(n_out, 512), nn.ReLU(inplace=True),
                nn.Linear(512, 1024), nn.ReLU(inplace=True),
                nn.Linear(1024, 2048), nn.ReLU(inplace=True),
                nn.Linear(2048, 4096)
            )

    def forward(self, signal_input):
        # output = self.encoder(signal_input)[-1]
        output = self.encoder(signal_input)
        output = self.dimensionality_up_sampling(output)

        return output

class TimmSimpleCNNgelu(nn.Module):
    def __init__(self, network: str,
                 image_size: int,
                 nchannels: int,
                 transformers: bool = False,
                 pretrained: bool = False,
                 num_classes: int = 0,
                 features_only: bool = True):

        super().__init__()
        if transformers:
            model_creator = {'model_name': network,
                             "pretrained": pretrained,
                             "num_classes": num_classes}
        else:
            model_creator = {'model_name': network,
                             "pretrained": pretrained,
                             "features_only": features_only}

        self.encoder = timm.create_model(**model_creator)

        self.dimensionality_reductor = None

        for param in self.encoder.parameters():
            param.requires_grad = True

        n_out = get_n_out_features(self.encoder, image_size, nchannels)

        if transformers:
            self.dimensionality_up_sampling = nn.Sequential(
                nn.Linear(n_out, 512), nn.GELU(),
                nn.Dropout(p=0.2),
                nn.Linear(512, 256), nn.GELU(),
                nn.Linear(256, 128), nn.GELU(),
                nn.Linear(128, 24)
                #nn.Linear(512, 1024), nn.ReLU(inplace=True),
                #nn.Linear(1024, 2048)
            )

            self.linear_1 = nn.Linear(n_out, 512)
            self.gelu = nn.GELU()
            self.linear_2 = nn.Linear(512, 256)
            self.linear_3 = nn.Linear(n_out, 128)
            self.linear_4 = nn.Linear(128, 24)
            self.relu = nn.ReLU()

        else:
            self.dimensionality_up_sampling = nn.Sequential(
                nn.Flatten(),
                nn.Linear(n_out, 512), nn.GELU(),
                nn.Linear(512, 256), nn.GELU(),
                nn.Linear(256, 128), nn.GELU(),
                nn.Linear(128, 24)
            )

    def forward(self, signal_input):
        #descomentar essa linha para rodar cnn
        #output = self.encoder(signal_input)[-1]
        #descomentar essa linha para rodar vit
        x = self.encoder(signal_input)

        out1 = self.linear_1(x)
        out1_gelu = self.gelu(out1)

        out2 = self.linear_2(out1_gelu)
        out1_downsampled = F.avg_pool1d(out1_gelu, kernel_size=2, stride=2)
        out2_residue = out2 + out1_downsampled

        out3 = self.linear_3(x)

        out4 = self.linear_4(out3)

        out4_residue = out4

        return out4_residue


class TimmSimpleTrack3Transfer(nn.Module):
    def __init__(self, network: str,
                 image_size: int,
                 nchannels: int,
                 transformers: bool = False,
                 pretrained: bool = False,
                 num_classes: int = 0,
                 features_only: bool = True):

        super().__init__()

        self.timm_cnn = TimmSimpleCNN(network,
                                      image_size,
                                      nchannels,
                                      transformers,
                                      pretrained,
                                      num_classes,
                                      features_only)

        self.linear = nn.Linear(2048, 4096)

    def forward(self, signal_input):
        # output = self.encoder(signal_input)[-1]
        output = self.timm_cnn(signal_input)
        output = self.linear(output)

        return output


class TimmCNN(nn.Module):
    def __init__(self, network: str,
                 image_size: int,
                 nchannels: int,
                 transformers: bool = False):

        super().__init__()
        if transformers:
            model_creator = {'model_name': network,
                             "pretrained": False,
                             "num_classes": 0}
        else:
            model_creator = {'model_name': network,
                             "pretrained": False,
                             "features_only": True}

        self.encoder = timm.create_model(**model_creator)

        self.dimensionality_reductor = None

        for param in self.encoder.parameters():
            param.requires_grad = True

        n_out = get_n_out_features(self.encoder, image_size, nchannels)

        if transformers:
            self.dimensionality_up_sampling = nn.Sequential(
                nn.Linear(n_out, 512), nn.ReLU(inplace=True),
                nn.Linear(512, 1024), nn.ReLU(inplace=True),
                nn.Linear(1024, 2048)
            )
        else:
            self.dimensionality_up_sampling = nn.Sequential(
                nn.Flatten(),
                nn.Linear(n_out, 512), nn.ReLU(inplace=True),
                nn.Linear(512, 1024), nn.ReLU(inplace=True),
                nn.Linear(1024, 2048)
            )

    def forward(self, signal_input, ppm):
        output = self.encoder(signal_input)[-1]
        output = self.dimensionality_up_sampling(output)

        min_ind = torch.argmax(ppm[ppm <= 10.8])
        max_ind = torch.argmin(ppm[ppm >= 9.8])

        # selecting part of arrays pertaining to region of interest
        noise_region = output[:, min_ind:max_ind].clone()
        output[:, min_ind:max_ind] = noise_region.mean(dim=1).view(noise_region.shape[0], 1).repeat(1,
                                                                                                    noise_region.shape[
                                                                                                        1])
        return output


class TimmMultiResolution(nn.Module):
    def __init__(self, network: str,
                 image_size: int,
                 nchannels: int,
                 transformers: bool = False,
                 pretrained: bool = False,
                 num_classes: int = 0,
                 features_only: bool = True):

        super().__init__()
        if transformers:
            model_creator = {'model_name': network,
                             "pretrained": pretrained,
                             "num_classes": num_classes}
        else:
            model_creator = {'model_name': network,
                             "pretrained": pretrained,
                             "features_only": features_only}

        self.encoder = timm.create_model(**model_creator)

        self.dimensionality_reductor = None

        for param in self.encoder.parameters():
            param.requires_grad = True

        n_out = get_n_out_features(self.encoder, image_size, nchannels)

        if transformers:
            self.dimensionality_up_sampling = nn.Sequential(
                nn.Linear(n_out, 512), nn.ReLU(inplace=True),
                nn.Linear(512, 1024), nn.ReLU(inplace=True),
                nn.Linear(1024, 2048),
                nn.Linear(2048, 4096)
            )
        else:
            self.dimensionality_sampling_2048 = nn.Sequential(
                nn.Flatten(),
                nn.Linear(30720, 512), nn.ReLU(inplace=True),
                nn.Linear(512, 1024), nn.ReLU(inplace=True),
                nn.Linear(1024, 2048)
            )

            self.dimensionality_sampling_residual = nn.Sequential(
                nn.Linear(2048, 2048)
            )

    def forward(self, signal_input, ppm):
        output = self.encoder(signal_input)[-1]

        output_2048 = self.dimensionality_sampling_2048(output)

        output_4096 = self.dimensionality_sampling_residual(output_2048)

        output_part_2_track_3 = rearrange([output_2048, output_4096], 't h w -> h (w t)')

        return output_2048, output_part_2_track_3


class UNETBaseline(nn.Module):

    # initializing the weights for the convolution layers
    def __init__(self, transient_count):
        super(UNETBaseline, self).__init__()

        self.down_conv_1_1 = nn.Conv2d(1, 16, kernel_size=(5, 1), padding="same")
        self.down_conv_1_2 = nn.Conv2d(16, 16, kernel_size=(3, 3), padding="same")

        self.down_conv_2_1 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding="same")
        self.down_conv_2_2 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding="same")

        self.down_conv_3_1 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding="same")
        self.down_conv_3_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding="same")

        self.up_conv_1_1 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding="same")
        self.up_conv_1_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding="same")

        self.up_conv_2_1 = nn.Conv2d(192, 64, kernel_size=(3, 3), padding="same")
        self.up_conv_2_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding="same")

        self.up_conv_3_1 = nn.Conv2d(96, 32, kernel_size=(3, 3), padding="same")
        self.up_conv_3_2 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding="same")

        self.end_conv_1_1 = nn.Conv2d(48, 128, kernel_size=(1, transient_count))
        self.end_conv_1_2 = nn.Conv2d(128, 1, kernel_size=(5, 5), padding="same")

    # defining forward pass
    def forward(self, x):
        # changing order of dimensions, as in torch the filters come first
        y = x.transpose(1, 3)
        y = y.transpose(2, 3)

        y = F.relu(self.down_conv_1_1(y))
        y_skip1 = F.relu(self.down_conv_1_2(y))

        y = F.max_pool2d(y_skip1, (2, 1))

        y = F.relu(self.down_conv_2_1(y))
        y_skip2 = F.relu(self.down_conv_2_2(y))

        y = F.max_pool2d(y_skip2, (2, 1))

        y = F.relu(self.down_conv_3_1(y))
        y_skip3 = F.relu(self.down_conv_3_2(y))

        y = F.max_pool2d(y_skip3, (2, 1))

        y = F.relu(self.up_conv_1_1(y))
        y = F.relu(self.up_conv_1_2(y))

        y = F.upsample(y, scale_factor=(2, 1))

        y = torch.concat([y, y_skip3], axis=1)

        y = F.relu(self.up_conv_2_1(y))
        y = F.relu(self.up_conv_2_2(y))

        y = F.upsample(y, scale_factor=(2, 1))

        y = torch.concat([y, y_skip2], axis=1)

        y = F.relu(self.up_conv_3_1(y))
        y = F.relu(self.up_conv_3_2(y))

        y = F.upsample(y, scale_factor=(2, 1))

        y = torch.concat([y, y_skip1], axis=1)

        y = F.relu(self.end_conv_1_1(y))
        y = self.end_conv_1_2(y)

        # converting the order of layers back to the original format

        y = y.transpose(1, 3)
        y = y.transpose(1, 2)

        # flattening result to only have 2 dimensions
        return y.view(y.shape[0], -1)
