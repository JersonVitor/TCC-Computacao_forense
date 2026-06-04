import os
import time
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import hashlib
import matplotlib.pyplot as plt
import src.config as config
from src.models import inicializar_resnet18, RawNet2Forense
from src.praat_metrics import extrair_biometria_fonetica_praat, decodificar_para_wav_temp
from src.report import construir_pdf_laudo_pericial_completo

def calcular_sha256(caminho_arquivo):
    """Gera a assinatura digital do arquivo para cadeia de custódia forense."""
    sha256_hash = hashlib.sha256()
    with open(caminho_arquivo, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest().upper()

def executar_pericia_forense_hibrida(caminho_audio, caminho_saida_pdf=None):
    """
    Executa a análise pericial híbrida em um arquivo de áudio:
    - Triagem neural (ResNet18 Mel-2D + RawNet2 Waveform-1D)
    - Extração biométrica vocal (F0, Jitter, Shimmer, HNR, F1, F2 via Praat)
    - Emissão de Laudo Técnico Formal com cadeia de custódia em PDF.
    """
    if not os.path.exists(caminho_audio):
        print(f"[ERRO] O áudio {caminho_audio} não foi localizado.")
        return

    nome_arquivo = os.path.basename(caminho_audio)
    nome_base = os.path.splitext(nome_arquivo)[0]
    print(f"\n[+] Iniciando Perícia Forense Híbrida para: {nome_arquivo}")

    # Diretórios de saída
    temp_dir = os.path.join(config.OUTPUT_DIR, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    if caminho_saida_pdf is None:
        caminho_saida_pdf = os.path.join(config.OUTPUT_DIR, f"Laudo_Pericial_Robustecido_{nome_base}.pdf")

    # 1. CADEIA DE CUSTÓDIA
    hash_original = calcular_sha256(caminho_audio)
    waveform, sr = torchaudio.load(caminho_audio)
    duracao = waveform.shape[1] / sr

    # Conversão e Normalização (16kHz PCM Mono)
    CAMINHO_WAV_NORMALIZADO = os.path.join(temp_dir, f"{nome_base}_normalizado.wav")
    decodificar_para_wav_temp(caminho_audio, CAMINHO_WAV_NORMALIZADO)
    hash_trabalho = calcular_sha256(CAMINHO_WAV_NORMALIZADO)

    dados_audio = {
        "nome": nome_arquivo,
        "duracao": duracao,
        "sr": sr,
        "hash_original": hash_original,
        "hash_trabalho": hash_trabalho
    }

    # Recarrega a onda normalizada para processamento das IAs
    waveform_16k, sr_16k = torchaudio.load(CAMINHO_WAV_NORMALIZADO)

    # 2. GERAÇÃO DE GRÁFICO: Waveform
    PATH_IMG_WF = os.path.join(temp_dir, "temp_waveform.png")
    plt.figure(figsize=(10, 2))
    plt.plot(np.linspace(0, duracao, num=waveform_16k.shape[1]), waveform_16k[0].numpy(), color='#1A365D', linewidth=0.4)
    plt.title("Forma de Onda (Waveform) - Sinal de Trabalho", fontsize=8, fontweight='bold', color='#1A365D')
    plt.xlabel("Tempo (s)", fontsize=7)
    plt.ylabel("Amplitude", fontsize=7)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(PATH_IMG_WF, dpi=200)
    plt.close()

    # 3. TRIAGEM POR INTELIGÊNCIA ARTIFICIAL (FASE 1)
    tamanho_chunk = config.TAMANHO_FIXO_AMOSTRAS # 64000
    total_amostras = waveform_16k.shape[1]
    melhor_chunk = None
    maior_rms = -1.0

    # Busca o trecho (chunk) de maior energia (RMS) no áudio
    if total_amostras <= tamanho_chunk:
        melhor_chunk = torch.nn.functional.pad(waveform_16k, (0, tamanho_chunk - total_amostras))
    else:
        for i in range(0, total_amostras, tamanho_chunk):
            chunk_atual = waveform_16k[:, i:i+tamanho_chunk]
            if chunk_atual.shape[1] < tamanho_chunk:
                chunk_atual = torch.nn.functional.pad(chunk_atual, (0, tamanho_chunk - chunk_atual.shape[1]))
            rms_atual = torch.sqrt(torch.mean(chunk_atual ** 2)).item()
            if rms_atual > maior_rms:
                maior_rms = rms_atual
                melhor_chunk = chunk_atual

    # Extrai espectrograma Mel
    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=512, n_mels=128)
    amp_to_db = torchaudio.transforms.AmplitudeToDB()
    mel_spec = amp_to_db(mel_transform(melhor_chunk))

    # GERAÇÃO DE GRÁFICO: Espectrograma
    PATH_IMG_SPEC = os.path.join(temp_dir, "temp_spectrogram.png")
    plt.figure(figsize=(10, 2))
    plt.imshow(mel_spec[0].numpy(), aspect='auto', origin='lower', cmap='viridis', extent=[0, 4, 0, 8000])
    plt.title("Espectrograma Digital (Análise de Frequência Temporal)", fontsize=8, fontweight='bold', color='#1A365D')
    plt.xlabel("Tempo Chunk (s)", fontsize=7)
    plt.ylabel("Frequência (Hz)", fontsize=7)
    plt.tight_layout()
    plt.savefig(PATH_IMG_SPEC, dpi=200)
    plt.close()

    # Prepara entrada para a ResNet (224x224, 3 canais)
    if mel_spec.shape[2] > config.TAMANHO_FIXO_TEMPO:
        mel_spec = mel_spec[:, :, :config.TAMANHO_FIXO_TEMPO]
    else:
        mel_spec = torch.nn.functional.pad(mel_spec, (0, config.TAMANHO_FIXO_TEMPO - mel_spec.shape[2]))
    
    # Adiciona dimensão de batch e replica para 3 canais
    tensor_resnet = mel_spec.unsqueeze(0).repeat(1, 3, 1, 1).to(config.DEVICE)

    # Inicia modelos neurais
    resnet = inicializar_resnet18(weights=None).to(config.DEVICE)
    if os.path.exists(config.PATH_PESOS_RESNET):
        resnet.load_state_dict(torch.load(config.PATH_PESOS_RESNET, map_location=config.DEVICE))
    else:
        print(f"[AVISO] Pesos da ResNet-18 em {config.PATH_PESOS_RESNET} não encontrados. Usando modelo não treinado.")

    rawnet = RawNet2Forense().to(config.DEVICE)
    if os.path.exists(config.PATH_PESOS_RAWNET):
        rawnet.load_state_dict(torch.load(config.PATH_PESOS_RAWNET, map_location=config.DEVICE))
    else:
        print(f"[AVISO] Pesos da RawNet2 em {config.PATH_PESOS_RAWNET} não encontrados. Usando modelo não treinado.")

    resnet.eval()
    rawnet.eval()

    # Inferências
    with torch.no_grad():
        out_resnet = resnet(tensor_resnet)
        logit_r = out_resnet.cpu().numpy()[0][1]
        pred_r = "SPOOF (IA)" if np.argmax(out_resnet.cpu().numpy()[0]) == 1 else "BONAFIDE (HUMANO)"

    # Prepara entrada para RawNet2
    tensor_rawnet = melhor_chunk.unsqueeze(0).to(config.DEVICE)
    with torch.no_grad():
        out_rawnet = rawnet(tensor_rawnet)
        logit_raw = out_rawnet.cpu().numpy()[0][1]
        pred_raw = "SPOOF (IA)" if np.argmax(out_rawnet.cpu().numpy()[0]) == 1 else "BONAFIDE (HUMANO)"

    dados_ia = {
        "logits_resnet": logit_r,
        "pred_resnet": pred_r,
        "logits_rawnet": logit_raw,
        "pred_rawnet": pred_raw
    }

    # 4. EXTRAÇÃO FONÉTICA PRAAT (FASE 2)
    print("[*] Extraindo invariantes biológicas de laringe via Parselmouth C++ Engine...")
    dados_praat = extrair_biometria_fonetica_praat(CAMINHO_WAV_NORMALIZADO)

    # 5. COMPILAÇÃO DO RELATÓRIO PDF
    construir_pdf_laudo_pericial_completo(
        caminho_saida_pdf, dados_audio, dados_ia, dados_praat, PATH_IMG_WF, PATH_IMG_SPEC
    )

    # Limpeza dos arquivos temporários
    for temp_file in [CAMINHO_WAV_NORMALIZADO, PATH_IMG_WF, PATH_IMG_SPEC]:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
    # Tenta remover a pasta temporária se estiver vazia
    try:
        os.rmdir(temp_dir)
    except Exception:
        pass
