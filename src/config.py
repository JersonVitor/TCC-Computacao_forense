import os
import torch

# Diretórios Base do Projeto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Certifica-se de que os diretórios principais existem
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Caminhos padrão para bases de áudio e metadados
AUDIO_DIR = os.path.join(DATA_DIR, "audios")
METADATA_PATH = os.path.join(DATA_DIR, "trial_metadata.txt")

# Caminhos padrão de saída para características acústicas extraídas
PASTA_SAIDA_SPECTROGRAM = os.path.join(DATA_DIR, "features_ResNet-18")
PASTA_SAIDA_RAW = os.path.join(DATA_DIR, "features_rawNet2")

# Caminhos padrão para carregar características durante o treino
PASTA_FEATURES_SPECTROGRAM = PASTA_SAIDA_SPECTROGRAM
PASTA_FEATURES_RAW = PASTA_SAIDA_RAW

# Caminhos dos modelos preditivos
PATH_PESOS_RESNET = os.path.join(MODELS_DIR, "resnet18_asvspoof_robusto_melhor.pth")
PATH_PESOS_RAWNET = os.path.join(MODELS_DIR, "rawnet2_asvspoof_robusto_melhor.pth")

# Configurações de Treinamento Geral
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4

# Parâmetros de Áudio
SAMPLE_RATE = 16000
TAMANHO_FIXO_TEMPO = 400      # Espectrograma Mel
TAMANHO_FIXO_AMOSTRAS = 64000  # Raw Waveform (4 segundos a 16kHz)
