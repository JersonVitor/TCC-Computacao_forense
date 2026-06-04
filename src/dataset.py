import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class EspectrogramaBlocksDataset(Dataset):
    def __init__(self, pasta_features, usar_treino=True, split_ratio=0.8):
        """
        Versão Otimizada para o TCC: Não carrega as matrizes na RAM.
        Mapeia os índices dinamicamente entre os blocos robustos salvos.
        """
        self.x_files = sorted(glob.glob(os.path.join(pasta_features, "x_train_robust_block_*.npy")))
        self.y_files = sorted(glob.glob(os.path.join(pasta_features, "y_train_robust_block_*.npy")))

        if len(self.x_files) == 0:
            # Tenta também procurar por blocos não-robustos normais caso existam (fallback)
            self.x_files = sorted(glob.glob(os.path.join(pasta_features, "x_train_block_*.npy")))
            self.y_files = sorted(glob.glob(os.path.join(pasta_features, "y_train_block_*.npy")))

        if len(self.x_files) == 0:
            raise ValueError(f"Nenhum arquivo .npy encontrado em {pasta_features}. Verifique o pipeline de extração.")

        # Descobre o tamanho total mapeando o cabeçalho dos arquivos de rótulos (y)
        self.tamanhos_blocos = [len(np.load(f, mmap_mode='r')) for f in self.y_files]
        self.total_amostras = sum(self.tamanhos_blocos)

        # Divisão determinística Treino/Validação (80/20) baseada em índices virtuais
        np.random.seed(42)
        indices_globais = np.arange(self.total_amostras)
        np.random.shuffle(indices_globais)

        limiar = int(self.total_amostras * split_ratio)

        if usar_treino:
            self.indices_finais = indices_globais[:limiar]
            print(f"[DATASET] Conjunto de TREINO instanciado com {len(self.indices_finais)} amostras de forma virtual.")
        else:
            self.indices_finais = indices_globais[limiar:]
            print(f"[DATASET] Conjunto de VALIDAÇÃO instanciado com {len(self.indices_finais)} amostras de forma virtual.")

    def __len__(self):
        return len(self.indices_finais)

    def __getitem__(self, idx):
        # Converte o índice do subconjunto para o índice global do dataset
        idx_global = self.indices_finais[idx]

        # Descobre em qual bloco físico de arquivo .npy este índice está guardado
        bloco_idx = 0
        acumulado = 0
        for i, tam in enumerate(self.tamanhos_blocos):
            if idx_global < acumulado + tam:
                bloco_idx = i
                idx_local = idx_global - acumulado
                break
            acumulado += tam

        # CARREGAMENTO DINÂMICO USANDO MMAP_MODE (Lê direto do disco gastando ZERO de RAM)
        x_block = np.load(self.x_files[bloco_idx], mmap_mode='r')
        y_block = np.load(self.y_files[bloco_idx], mmap_mode='r')

        # Extrai apenas a linha exata requisitada pelo Batch
        x_sample = np.array(x_block[idx_local], dtype=np.float32)
        y_sample = int(y_block[idx_local])

        # 1. Normalização Min-Max
        min_val = x_sample.min()
        max_val = x_sample.max()
        if max_val - min_val > 0:
            x_sample = (x_sample - min_val) / (max_val - min_val)

        # 2. Redimensionamento dinâmico para 224x224 (ResNet input)
        tensor_x = torch.tensor(x_sample).unsqueeze(0).unsqueeze(0) # Shape: (1, 1, 128, 400)
        tensor_x = nn.functional.interpolate(tensor_x, size=(224, 224), mode='bilinear', align_corners=False)
        tensor_x = tensor_x.squeeze(0) # Shape: (1, 224, 224)

        # 3. Replicar para 3 canais (RGB simulado para Transfer Learning)
        tensor_x = tensor_x.repeat(3, 1, 1) # Shape final: (3, 224, 224)

        return tensor_x, torch.tensor(y_sample, dtype=torch.long)


class RawAudioBlocksDataset(Dataset):
    def __init__(self, pasta_features, usar_treino=True, split_ratio=0.8):
        """
        Dataset para ondas brutas (Raw Waveforms 1D) com Lazy Loading.
        """
        self.x_files = sorted(glob.glob(os.path.join(pasta_features, "x_train_raw_block_*.npy")))
        self.y_files = sorted(glob.glob(os.path.join(pasta_features, "y_train_raw_block_*.npy")))

        if len(self.x_files) == 0:
            raise ValueError(f"Nenhum arquivo .npy de audio 1D encontrado em {pasta_features}. Verifique o pipeline de extração.")

        self.tamanhos_blocos = [len(np.load(f, mmap_mode='r')) for f in self.y_files]
        self.total_amostras = sum(self.tamanhos_blocos)

        # Divisão determinística Treino/Validação (80/20) baseada em índices virtuais
        np.random.seed(42)
        indices_globais = np.arange(self.total_amostras)
        np.random.shuffle(indices_globais)

        limiar = int(self.total_amostras * split_ratio)

        if usar_treino:
            self.indices_finais = indices_globais[:limiar]
            print(f"[DATASET 1D] Conjunto de TREINO instanciado com {len(self.indices_finais)} amostras de forma virtual.")
        else:
            self.indices_finais = indices_globais[limiar:]
            print(f"[DATASET 1D] Conjunto de VALIDAÇÃO instanciado com {len(self.indices_finais)} amostras de forma virtual.")

    def __len__(self):
        return len(self.indices_finais)

    def __getitem__(self, idx):
        idx_global = self.indices_finais[idx]
        bloco_idx = 0
        acumulado = 0
        for i, tam in enumerate(self.tamanhos_blocos):
            if idx_global < acumulado + tam:
                bloco_idx = i
                idx_local = idx_global - acumulado
                break
            acumulado += tam

        x_block = np.load(self.x_files[bloco_idx], mmap_mode='r')
        y_block = np.load(self.y_files[bloco_idx], mmap_mode='r')

        x_sample = np.array(x_block[idx_local], dtype=np.float32)
        y_sample = int(y_block[idx_local])

        # Adiciona dimensão extra de canal: shape (1, num_amostras)
        tensor_x = torch.tensor(x_sample).unsqueeze(0)
        return tensor_x, torch.tensor(y_sample, dtype=torch.long)
