import os
import pandas as pd
import torch
import torchaudio
import numpy as np
from src.augment import AdaptadorAcusticoForense
import src.config as config

def load_asvspoof_protocol(metadata_path):
    records = []
    with open(metadata_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            # O protocolo ASVspoof contem:
            # partes[1] = file_id, partes[4] = 'bonafide'/'spoof'
            if len(parts) >= 5:
                label = 0 if parts[4] == 'bonafide' else 1
                records.append((parts[1], label))
    return pd.DataFrame(records, columns=['file_id', 'label'])

def extrair_espectrogramas(metadata_path, audio_dir, output_dir, samples_per_class=15000, prob_aug=0.7, block_size=5000):
    """
    Esteira de Processamento de Espectrogramas Mel 2D com Blindagem de Canal.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Carregando metadados de {metadata_path}...")
    df_all = load_asvspoof_protocol(metadata_path)
    
    df_reais = df_all[df_all['label'] == 0]
    df_spoof_ia = df_all[df_all['label'] == 1]
    
    print(f"[INFO] Disponíveis - Humanos (Reais): {len(df_reais)} | IA (Spoof): {len(df_spoof_ia)}")
    
    # Valida quantidades solicitadas
    n_reais = min(samples_per_class, len(df_reais))
    n_ia = min(samples_per_class, len(df_spoof_ia))
    
    print(f"[INFO] Selecionando {n_reais} Reais e {n_ia} IA para extração...")
    df_reais_amostra = df_reais.sample(n=n_reais, random_state=42)
    df_ia_amostra = df_spoof_ia.sample(n=n_ia, random_state=42)
    
    df_sample = pd.concat([df_reais_amostra, df_ia_amostra], axis=0)
    df_sample = df_sample.sample(frac=1, random_state=42).reset_index(drop=True)
    
    total_requisitado = len(df_sample)
    print(f"[EVIDÊNCIA] Iniciando extração de MelSpectrograms para {total_requisitado} arquivos...")

    augmentor = AdaptadorAcusticoForense(sample_rate=config.SAMPLE_RATE, probabilidade=prob_aug)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=128
    )
    amp_to_db = torchaudio.transforms.AmplitudeToDB()

    all_features = []
    all_labels = []
    contador = 0
    bloco_num = 0

    for idx, row in df_sample.iterrows():
        audio_path = os.path.join(audio_dir, f"{row['file_id']}.flac")
        if not os.path.exists(audio_path):
            continue

        try:
            waveform, sr = torchaudio.load(audio_path)
            if sr != config.SAMPLE_RATE:
                waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=config.SAMPLE_RATE)(waveform)
            
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            waveform_degradada = augmentor.aplicar_degradacao(waveform)
            mel_spec = amp_to_db(mel_transform(waveform_degradada))

            # Padronização temporal
            if mel_spec.shape[2] > config.TAMANHO_FIXO_TEMPO:
                mel_spec = mel_spec[:, :, :config.TAMANHO_FIXO_TEMPO]
            else:
                pad_size = config.TAMANHO_FIXO_TEMPO - mel_spec.shape[2]
                mel_spec = torch.nn.functional.pad(mel_spec, (0, pad_size))

            all_features.append(mel_spec.squeeze(0).numpy().astype(np.float32))
            all_labels.append(row['label'])
            contador += 1

        except Exception as e:
            continue

        if contador % 500 == 0:
            print(f"[PROGRESSO] {contador}/{total_requisitado} espectrogramas processados...")

        if contador % block_size == 0 and len(all_features) > 0:
            bloco_num += 1
            np.save(os.path.join(output_dir, f"x_train_robust_block_{bloco_num}.npy"), np.array(all_features))
            np.save(os.path.join(output_dir, f"y_train_robust_block_{bloco_num}.npy"), np.array(all_labels))
            all_features = []
            all_labels = []
            print(f"[DISCO] Bloco robusto {bloco_num} gravado com sucesso em: {output_dir}")

    # Salva o resíduo restante
    if len(all_features) > 0:
        bloco_num += 1
        np.save(os.path.join(output_dir, f"x_train_robust_block_{bloco_num}.npy"), np.array(all_features))
        np.save(os.path.join(output_dir, f"y_train_robust_block_{bloco_num}.npy"), np.array(all_labels))
        print(f"[DISCO] Bloco final robusto {bloco_num} gravado com sucesso em: {output_dir}")

    print(f"\n[PROCESSO CONCLUÍDO] Extração Concluída. {contador} espectrogramas salvos em {output_dir}")

def extrair_raw_waveforms(metadata_path, audio_dir, output_dir, samples_per_class=15000, prob_aug=0.7, block_size=5000):
    """
    Esteira de Processamento de Ondas Brutas 1D (Raw Audio) com Blindagem de Canal.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Carregando metadados de {metadata_path}...")
    df_all = load_asvspoof_protocol(metadata_path)
    
    # Embaralhamento total para buscar arquivos de forma aleatória em toda a base
    df_shuffled = df_all.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"[INFO] Universo de {len(df_shuffled)} chaves pronto para varredura.")

    augmentor = AdaptadorAcusticoForense(sample_rate=config.SAMPLE_RATE, probabilidade=prob_aug)
    
    contador_reais = 0
    contador_ia = 0
    all_features_raw = []
    all_labels = []
    contador = 0
    bloco_num = 0

    print(f"[INFO] Iniciando busca direta e extração 1D até atingir {samples_per_class * 2} arquivos...")

    for idx, row in df_shuffled.iterrows():
        if contador_reais >= samples_per_class and contador_ia >= samples_per_class:
            break

        label = row['label']
        # Ignora se cota da classe já foi preenchida
        if label == 0 and contador_reais >= samples_per_class:
            continue
        if label == 1 and contador_ia >= samples_per_class:
            continue

        audio_path = os.path.join(audio_dir, f"{row['file_id']}.flac")
        if not os.path.exists(audio_path):
            continue

        try:
            waveform, sr = torchaudio.load(audio_path)
            if sr != config.SAMPLE_RATE:
                waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=config.SAMPLE_RATE)(waveform)

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            waveform_degradada = augmentor.aplicar_degradacao(waveform)

            # Padroniza tamanho da onda
            if waveform_degradada.shape[1] > config.TAMANHO_FIXO_AMOSTRAS:
                waveform_final = waveform_degradada[:, :config.TAMANHO_FIXO_AMOSTRAS]
            else:
                pad_size = config.TAMANHO_FIXO_AMOSTRAS - waveform_degradada.shape[1]
                waveform_final = torch.nn.functional.pad(waveform_degradada, (0, pad_size))

            vetor_1d = waveform_final.squeeze(0).numpy().astype(np.float32)

            all_features_raw.append(vetor_1d)
            all_labels.append(label)

            if label == 0:
                contador_reais += 1
            else:
                contador_ia += 1
            contador += 1

            if contador % 500 == 0:
                print(f"[PROGRESSO] {contador}/{samples_per_class * 2} processados (Humanos: {contador_reais}/{samples_per_class} | IA: {contador_ia}/{samples_per_class})")

            if contador % block_size == 0 and len(all_features_raw) > 0:
                bloco_num += 1
                np.save(os.path.join(output_dir, f"x_train_raw_block_{bloco_num}.npy"), np.array(all_features_raw))
                np.save(os.path.join(output_dir, f"y_train_raw_block_{bloco_num}.npy"), np.array(all_labels))
                all_features_raw = []
                all_labels = []
                print(f"[DISCO] Bloco 1D robusto {bloco_num} gravado com sucesso em: {output_dir}")

        except Exception as e:
            continue

    if len(all_features_raw) > 0:
        bloco_num += 1
        np.save(os.path.join(output_dir, f"x_train_raw_block_{bloco_num}.npy"), np.array(all_features_raw))
        np.save(os.path.join(output_dir, f"y_train_raw_block_{bloco_num}.npy"), np.array(all_labels))
        print(f"[DISCO] Bloco 1D final robusto {bloco_num} gravado com sucesso em: {output_dir}")

    print(f"\n[PROCESSO CONCLUÍDO] Extração Raw Waveform concluída. {contador} arquivos processados em {output_dir}")
