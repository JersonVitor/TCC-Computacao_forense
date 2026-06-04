import os
import glob
import subprocess
import numpy as np
import pandas as pd
import librosa
import parselmouth
from parselmouth.praat import call
import src.config as config

def decodificar_para_wav_temp(caminho_entrada, caminho_wav):
    """
    Decodifica um arquivo de áudio (por exemplo, .m4a) para .wav (16kHz, mono, PCM) usando ffmpeg.
    Se a extensão já for .wav ou .flac, pode ser usado diretamente, mas ffmpeg garante a normalização de sample rate.
    """
    comando = f'ffmpeg -i "{caminho_entrada}" -acodec pcm_s16le -ac 1 -ar 16000 "{caminho_wav}" -y -loglevel quiet'
    try:
        subprocess.run(comando, shell=True, check=True)
    except Exception as e:
        # Tenta copiar o arquivo caso ffmpeg falhe ou não esteja no PATH (fallback se já for compatível)
        import shutil
        if caminho_entrada.lower().endswith(('.wav', '.flac')):
            shutil.copy(caminho_entrada, caminho_wav)
        else:
            raise RuntimeError("ffmpeg falhou ou não está instalado para decodificar arquivos comprimidos.")

def extrair_biometria_fonetica_praat(caminho_wav):
    """
    Extrai parâmetros acústicos e micro-flutuações da laringe via Praat e Parselmouth C++.
    """
    sound = parselmouth.Sound(caminho_wav)
    
    # 1. Pitch (F0)
    pitch = call(sound, "To Pitch (ac)", 0.0, 75.0, 15.0, "no", 0.03, 0.45, 0.01, 0.35, 0.14, 600.0)
    mean_f0 = call(pitch, "Get mean", 0.0, 0.0, "Hertz")
    min_f0 = call(pitch, "Get minimum", 0.0, 0.0, "Hertz", "Parabolic")
    max_f0 = call(pitch, "Get maximum", 0.0, 0.0, "Hertz", "Parabolic")

    # 2. Perturbações Glóticas (Jitter e Shimmer)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", 75.0, 600.0)
    local_jitter = call(pointProcess, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)
    local_shimmer = call([sound, pointProcess], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)

    # 3. Relação Harmônico-Ruído (HNR)
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75.0, 0.1, 4.5)
    mean_hnr = call(harmonicity, "Get mean", 0.0, 0.0)

    # 4. Formantes (Burg)
    formant = call(sound, "To Formant (burg)", 0.0, 5.0, 5500.0, 0.025, 50.0)
    mean_f1 = call(formant, "Get mean", 1, 0.0, 0.0, "Hertz")
    mean_f2 = call(formant, "Get mean", 2, 0.0, 0.0, "Hertz")

    return {
        "mean_f0": mean_f0 if not np.isnan(mean_f0) else 0.0,
        "min_f0": min_f0 if not np.isnan(min_f0) else 0.0,
        "max_f0": max_f0 if not np.isnan(max_f0) else 0.0,
        "jitter": local_jitter if not np.isnan(local_jitter) else 0.0,
        "shimmer": local_shimmer if not np.isnan(local_shimmer) else 0.0,
        "hnr": mean_hnr if not np.isnan(mean_hnr) else 0.0,
        "f1": mean_f1 if not np.isnan(mean_f1) else 0.0,
        "f2": mean_f2 if not np.isnan(mean_f2) else 0.0,
    }

def extrair_metricas_librosa_e_pausas(caminho_wav):
    """
    Mapeia silêncios e pausas no fluxo articulatório utilizando librosa.
    """
    y, sr = librosa.load(caminho_wav, sr=16000)
    intervalos_fala = librosa.effects.split(y, top_db=50)

    duracao_total = (len(y) / sr) * 1000  # em ms
    tempo_fala = sum([(fim - inc) / sr * 1000 for inc, fim in intervalos_fala])
    tempo_silencio = duracao_total - tempo_fala
    qtd_pausas = len(intervalos_fala) - 1 if len(intervalos_fala) > 1 else 0

    return {
        "duracao_total": duracao_total,
        "tempo_fala": tempo_fala,
        "tempo_silencio": tempo_silencio,
        "qtd_pausas": qtd_pausas
    }

def processar_lote_pasta_teste(pasta_entrada, pasta_wav_temp):
    """
    Executa a extração em lote de áudios comprimidos (.m4a ou outros)
    e calcula as médias separando por "Verdade" ou "Mentira" conforme nomenclatura do arquivo.
    """
    os.makedirs(pasta_wav_temp, exist_ok=True)
    arquivos_m4a = glob.glob(os.path.join(pasta_entrada, "*.m4a"))
    arquivos_wav = glob.glob(os.path.join(pasta_entrada, "*.wav"))
    arquivos = arquivos_m4a + arquivos_wav
    
    resultados = []
    print(f"[*] Processando {len(arquivos)} arquivos de áudio para estatísticas de laringe...")

    for arq in arquivos:
        nome_arquivo = os.path.basename(arq)
        partes = nome_arquivo.split('_')
        
        # Tenta detectar classe 'Verdade'/'Mentira'
        veracidade = "Indefinido"
        for p in partes:
            if p.lower() in ['verdade', 'mentira']:
                veracidade = p.capitalize()
                break

        nome_limpo = os.path.splitext(nome_arquivo)[0]
        caminho_wav = os.path.join(pasta_wav_temp, f"{nome_limpo}_temp.wav")
        
        try:
            decodificar_para_wav_temp(arq, caminho_wav)
            
            p_metrics = extrair_biometria_fonetica_praat(caminho_wav)
            l_metrics = extrair_metricas_librosa_e_pausas(caminho_wav)
            
            # Agrega todas as métricas
            metricas_comp = {
                "F0 (Hz)": p_metrics["mean_f0"],
                "Jitter (%)": p_metrics["jitter"] * 100,
                "Shimmer (%)": p_metrics["shimmer"] * 100,
                "HNR (dB)": p_metrics["hnr"],
                "Pausas (Qtd)": l_metrics["qtd_pausas"],
                "Silencio (ms)": l_metrics["tempo_silencio"],
                "Veracidade": veracidade
            }
            resultados.append(metricas_comp)
        except Exception as e:
            print(f"[AVISO] Falha ao processar {nome_arquivo}: {e}")
        finally:
            if os.path.exists(caminho_wav):
                os.remove(caminho_wav)

    if len(resultados) == 0:
        print("[AVISO] Nenhum áudio pôde ser processado com sucesso.")
        return pd.DataFrame()

    df = pd.DataFrame(resultados)
    df_medias = df.groupby('Veracidade').mean().round(3)
    return df_medias
