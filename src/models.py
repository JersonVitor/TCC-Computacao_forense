import torch
import torch.nn as nn
import torchvision.models as models

def inicializar_resnet18(weights=None):
    """
    Inicializa ResNet-18 e ajusta a camada totalmente conectada
    para classificação binária (Humano vs Spoof).
    """
    if weights == 'default':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    else:
        model = models.resnet18(weights=None)
    
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    return model

class BlocoResidual1D(nn.Module):
    """
    Bloco residual de convolução 1D usado no processamento temporal da RawNet2.
    """
    def __init__(self, canais):
        super(BlocoResidual1D, self).__init__()
        self.conv1 = nn.Conv1d(canais, canais, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(canais)
        self.leaky = nn.LeakyReLU(0.3)
        self.conv2 = nn.Conv1d(canais, canais, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(canais)

    def forward(self, x):
        residual = x
        out = self.leaky(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.leaky(out)

class RawNet2Forense(nn.Module):
    """
    Arquitetura RawNet2 Forense adaptada para processamento
    de ondas sonoras 1D brutas e detecção de voz sintetizada.
    """
    def __init__(self):
        super(RawNet2Forense, self).__init__()
        self.conv_frontal = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=251, stride=10, padding=125),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.3),
            nn.MaxPool1d(kernel_size=3)
        )
        self.res_block1 = BlocoResidual1D(32)
        self.pooling1 = nn.MaxPool1d(kernel_size=3)

        self.conv_meio = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.res_block2 = BlocoResidual1D(64)
        self.pooling2 = nn.MaxPool1d(kernel_size=3)

        self.gru = nn.GRU(input_size=64, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)

        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.4),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        out = self.conv_frontal(x)
        out = self.pooling1(self.res_block1(out))
        out = self.conv_meio(out)
        out = self.pooling2(self.res_block2(out))
        out = out.transpose(1, 2)
        self.gru.flatten_parameters()
        out, _ = self.gru(out)
        out = out[:, -1, :]
        return self.fc(out)
