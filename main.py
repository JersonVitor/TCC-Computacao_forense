import sys
import argparse
import os
import src.config as config
from src.feature_extraction import extrair_espectrogramas, extrair_raw_waveforms
from src.training import treinar_resnet18, treinar_rawnet2
from src.evaluation import avaliar_resnet18, avaliar_rawnet2
from src.praat_metrics import processar_lote_pasta_teste
from src.inference import executar_pericia_forense_hibrida

def command_extract_features(args):
    """Subcomando para extração de características acústicas com canal degradado."""
    metadata_path = args.metadata or config.METADATA_PATH
    audio_dir = args.audio_dir or config.AUDIO_DIR
    
    if args.type == 'spectrogram':
        output_dir = args.output_dir or config.PASTA_SAIDA_SPECTROGRAM
        extrair_espectrogramas(
            metadata_path=metadata_path,
            audio_dir=audio_dir,
            output_dir=output_dir,
            samples_per_class=args.samples_per_class,
            prob_aug=args.prob_aug,
            block_size=args.block_size
        )
    elif args.type == 'raw':
        output_dir = args.output_dir or config.PASTA_SAIDA_RAW
        extrair_raw_waveforms(
            metadata_path=metadata_path,
            audio_dir=audio_dir,
            output_dir=output_dir,
            samples_per_class=args.samples_per_class,
            prob_aug=args.prob_aug,
            block_size=args.block_size
        )
    else:
        print("[ERRO] Tipo de características desconhecido.")

def command_train(args):
    """Subcomando para treinar modelos neurais forenses."""
    if args.model == 'resnet18':
        features_dir = args.features_dir or config.PASTA_FEATURES_SPECTROGRAM
        output_dir = args.output_dir or config.MODELS_DIR
        epochs = args.epochs if args.epochs is not None else 5
        lr = args.lr if args.lr is not None else 1e-4
        
        treinar_resnet18(
            pasta_features=features_dir,
            pasta_saida=output_dir,
            epochs=epochs,
            lr=lr,
            batch_size=args.batch_size
        )
    elif args.model == 'rawnet2':
        features_dir = args.features_dir or config.PASTA_FEATURES_RAW
        output_dir = args.output_dir or config.MODELS_DIR
        epochs = args.epochs if args.epochs is not None else 20
        lr = args.lr if args.lr is not None else 5e-4
        
        treinar_rawnet2(
            pasta_features=features_dir,
            pasta_saida=output_dir,
            epochs=epochs,
            lr=lr,
            batch_size=args.batch_size
        )
    else:
        print("[ERRO] Modelo desconhecido para treinamento.")

def command_evaluate(args):
    """Subcomando para avaliação forense de EER e Cllr."""
    if args.model == 'resnet18':
        features_dir = args.features_dir or config.PASTA_FEATURES_SPECTROGRAM
        weights = args.weights or config.PATH_PESOS_RESNET
        avaliar_resnet18(pasta_features=features_dir, pesos_path=weights)
    elif args.model == 'rawnet2':
        features_dir = args.features_dir or config.PASTA_FEATURES_RAW
        weights = args.weights or config.PATH_PESOS_RAWNET
        avaliar_rawnet2(pasta_features=features_dir, pesos_path=weights)
    else:
        print("[ERRO] Modelo desconhecido para avaliação.")

def command_pericia(args):
    """Subcomando para realizar perícia em um único arquivo e gerar laudo PDF."""
    if not args.audio:
        print("[ERRO] Caminho do áudio obrigatório. Use --audio <caminho>.")
        return
    executar_pericia_forense_hibrida(args.audio, args.output_pdf)

def command_batch_stats(args):
    """Subcomando para extrair estatísticas vocais por veracidade a partir de uma pasta de m4a/wav."""
    if not args.dir:
        print("[ERRO] Caminho da pasta contendo os áudios é obrigatório. Use --dir <caminho>.")
        return
    temp_dir = args.temp_wav_dir or os.path.join(config.DATA_DIR, "temp_wav_forense")
    
    df_medias = processar_lote_pasta_teste(args.dir, temp_dir)
    
    if not df_medias.empty:
        print("\n=======================================================")
        print("  MÉDIAS EXTRAÍDAS PARA A TABELA DO TCC (FASE 2)       ")
        print("=======================================================\n")
        print(df_medias.T)
        
        # Salva o resultado em CSV para uso do aluno
        csv_path = os.path.join(config.OUTPUT_DIR, "medias_tcc_fase2.csv")
        df_medias.to_csv(csv_path)
        print(f"\n[SUCESSO] Tabela exportada em: {csv_path}")
    else:
        print("[ERRO] Nenhuma estatística pôde ser extraída dos arquivos na pasta especificada.")

def main():
    parser = argparse.ArgumentParser(
        description="PUC Minas - Framework Forense Híbrido Autônomo para Detecção de Spoofing e Clonagem de Voz"
    )
    subparsers = parser.add_subparsers(dest="command", help="Comandos disponíveis", required=True)

    # 1. Subcomando: extract-features
    p_extract = subparsers.add_parser("extract-features", help="Extrai características acústicas com canal degradado")
    p_extract.add_argument("--type", choices=["spectrogram", "raw"], required=True, help="Tipo de característica a ser extraída")
    p_extract.add_argument("--metadata", help="Caminho do arquivo de metadados trial_metadata.txt")
    p_extract.add_argument("--audio-dir", help="Caminho do diretório contendo áudios FLAC")
    p_extract.add_argument("--output-dir", help="Diretório de saída para os blocos .npy")
    p_extract.add_argument("--samples-per-class", type=int, default=15000, help="Quantidade de amostras por classe (Bonafide/Spoof)")
    p_extract.add_argument("--prob-aug", type=float, default=0.7, help="Probabilidade de aplicar degradação acústica forense")
    p_extract.add_argument("--block-size", type=int, default=5000, help="Tamanho do lote de arquivos salvos em cada bloco físico")
    p_extract.set_defaults(func=command_extract_features)

    # 2. Subcomando: train
    p_train = subparsers.add_parser("train", help="Treina modelos de rede profunda (ResNet-18 ou RawNet2)")
    p_train.add_argument("--model", choices=["resnet18", "rawnet2"], required=True, help="Modelo a ser treinado")
    p_train.add_argument("--features-dir", help="Caminho do diretório contendo as características .npy extraídas")
    p_train.add_argument("--output-dir", help="Diretório onde os pesos salvos serão salvos")
    p_train.add_argument("--epochs", type=int, help="Número de épocas de treinamento")
    p_train.add_argument("--lr", type=float, help="Taxa de aprendizado (Learning Rate)")
    p_train.add_argument("--batch-size", type=int, default=64, help="Tamanho do lote de entrada (Batch Size)")
    p_train.set_defaults(func=command_train)

    # 3. Subcomando: evaluate
    p_eval = subparsers.add_parser("evaluate", help="Avalia modelos treinados na partição de validação")
    p_eval.add_argument("--model", choices=["resnet18", "rawnet2"], required=True, help="Modelo deep learning a ser avaliado")
    p_eval.add_argument("--features-dir", help="Caminho do diretório contendo as características .npy")
    p_eval.add_argument("--weights", help="Caminho do arquivo de checkpoint contendo os pesos treinados (.pth)")
    p_eval.set_defaults(func=command_evaluate)

    # 4. Subcomando: pericia
    p_pericia = subparsers.add_parser("pericia", help="Executa perícia em arquivo individual de voz e gera laudo em PDF")
    p_pericia.add_argument("--audio", required=True, help="Caminho para o arquivo de áudio de teste (.wav, .m4a, .flac)")
    p_pericia.add_argument("--output-pdf", help="Caminho para o PDF de laudo a ser gerado")
    p_pericia.set_defaults(func=command_pericia)

    # 5. Subcomando: batch-stats
    p_batch = subparsers.add_parser("batch-stats", help="Extrai médias acústicas em lote agrupando por Verdade/Mentira")
    p_batch.add_argument("--dir", required=True, help="Diretório contendo áudios comprimidos (.m4a ou .wav) para lote")
    p_batch.add_argument("--temp-wav-dir", help="Caminho do diretório temporário para decodificação WAV")
    p_batch.set_defaults(func=command_batch_stats)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
