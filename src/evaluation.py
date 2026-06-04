import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve
import src.config as config
from src.dataset import EspectrogramaBlocksDataset, RawAudioBlocksDataset
from src.models import inicializar_resnet18, RawNet2Forense

def calcular_eer_forense(y_true, y_scores):
    """
    Calcula o Equal Error Rate (EER) e o limiar ótimo de decisão.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    idx_otimo = np.nanargmin(np.absolute((fpr - fnr)))
    return fpr[idx_otimo] * 100, thresholds[idx_otimo]

def calcular_cllr_forense(y_true, y_scores):
    """
    Calcula o Cllr (Log-Likelihood-Ratio Cost), métrica forense de custo de decisão.
    """
    scores = np.clip(y_scores, 1e-7, 1 - 1e-7)
    scores_bonafide = scores[y_true == 1]
    scores_spoof = scores[y_true == 0]

    llr_bonafide = np.log2(scores_bonafide / (1 - scores_bonafide))
    llr_spoof = np.log2(scores_spoof / (1 - scores_spoof))

    c_llr = 0.5 * (np.mean(np.log2(1 + 2**(-llr_bonafide))) + np.mean(np.log2(1 + 2**(llr_spoof))))
    return c_llr

def avaliar_resnet18(pasta_features=config.PASTA_FEATURES_SPECTROGRAM, pesos_path=config.PATH_PESOS_RESNET):
    """
    Avalia a rede neural ResNet-18 carregando o checkpoint salvo e calculando EER e Cllr.
    """
    if not os.path.exists(pesos_path):
        print(f"[ERRO] O arquivo de pesos em {pesos_path} não foi localizado.")
        return

    print(f"[*] Carregando partição oculta de validação 2D da pasta local...")
    dataset_teste = EspectrogramaBlocksDataset(pasta_features, usar_treino=False)
    loader_teste = DataLoader(dataset_teste, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # Reconstrução da topologia
    model = inicializar_resnet18(weights=None).to(config.DEVICE)
    model.load_state_dict(torch.load(pesos_path, map_location=config.DEVICE))
    model.eval()
    print("[INFO] Checkpoint da ResNet-18 injetado com sucesso.")

    y_true_acumulado = []
    y_scores_acumulado = []

    print("[*] Processando predições e extraindo scores estatísticos espectrais...")
    with torch.no_grad():
        for audios, labels in loader_teste:
            audios = audios.to(config.DEVICE)
            outputs = model(audios)

            # Softmax para obter probabilidade contínua da classe 1 (Bonafide)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()

            y_scores_acumulado.extend(probs)
            y_true_acumulado.extend(labels.numpy())

    y_true = np.array(y_true_acumulado)
    y_scores = np.array(y_scores_acumulado)

    # Métricas
    eer, limiar_otimo = calcular_eer_forense(y_true, y_scores)
    cllr = calcular_cllr_forense(y_true, y_scores)

    print("\n" + "="*50)
    print("        MÉTRICAS FORENSES EXTRAÍDAS (RESNET-18)        ")
    print("="*50)
    print(f"[*] Equal Error Rate (EER):           {eer:.2f}%")
    print(f"[*] Limiar Ótimo de Decisão:          {limiar_otimo:.4f}")
    print(f"[*] Log-Likelihood-Ratio Cost (Cllr): {cllr:.4f}")
    print("="*50)

def avaliar_rawnet2(pasta_features=config.PASTA_FEATURES_RAW, pesos_path=config.PATH_PESOS_RAWNET):
    """
    Avalia a rede neural RawNet2 carregando o checkpoint salvo e calculando EER e Cllr.
    """
    if not os.path.exists(pesos_path):
        print(f"[ERRO] O arquivo de pesos em {pesos_path} não foi localizado.")
        return

    print(f"[*] Carregando partição oculta de validação 1D da pasta local...")
    dataset_teste = RawAudioBlocksDataset(pasta_features, usar_treino=False)
    loader_teste = DataLoader(dataset_teste, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # Reconstrução da topologia
    model = RawNet2Forense().to(config.DEVICE)
    model.load_state_dict(torch.load(pesos_path, map_location=config.DEVICE))
    model.eval()
    print("[INFO] Checkpoint da RawNet2 injetado com sucesso.")

    y_true_acumulado = []
    y_scores_acumulado = []

    print("[*] Processando predições e extraindo scores estatísticos temporais...")
    with torch.no_grad():
        for audios, labels in loader_teste:
            audios = audios.to(config.DEVICE)
            outputs = model(audios)

            # Softmax para obter probabilidade contínua da classe 1 (Bonafide)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()

            y_scores_acumulado.extend(probs)
            y_true_acumulado.extend(labels.numpy())

    y_true = np.array(y_true_acumulado)
    y_scores = np.array(y_scores_acumulado)

    # Métricas
    eer, limiar_otimo = calcular_eer_forense(y_true, y_scores)
    cllr = calcular_cllr_forense(y_true, y_scores)

    print("\n" + "="*50)
    print("        MÉTRICAS FORENSES EXTRAÍDAS (RAWNET2)        ")
    print("="*50)
    print(f"[*] Equal Error Rate (EER):           {eer:.2f}%")
    print(f"[*] Limiar Ótimo de Decisão:          {limiar_otimo:.4f}")
    print(f"[*] Log-Likelihood-Ratio Cost (Cllr): {cllr:.4f}")
    print("="*50)
