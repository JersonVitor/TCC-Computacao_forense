import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import src.config as config
from src.dataset import EspectrogramaBlocksDataset, RawAudioBlocksDataset
from src.models import inicializar_resnet18, RawNet2Forense

def treinar_resnet18(pasta_features=config.PASTA_FEATURES_SPECTROGRAM, 
                     pasta_saida=config.MODELS_DIR, 
                     epochs=config.NUM_EPOCHS, 
                     lr=config.LEARNING_RATE, 
                     batch_size=config.BATCH_SIZE):
    """
    Treina a rede neural ResNet-18 utilizando características MelSpectrogram 2D.
    """
    os.makedirs(pasta_saida, exist_ok=True)
    print(f"[INFO] Utilizando dispositivo: {config.DEVICE}")

    # Instanciação dos Datasets e Dataloaders
    train_dataset = EspectrogramaBlocksDataset(pasta_features, usar_treino=True)
    val_dataset = EspectrogramaBlocksDataset(pasta_features, usar_treino=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Inicialização do Modelo
    print("[INFO] Inicializando ResNet-18 com pesos pré-treinados...")
    model = inicializar_resnet18(weights='default').to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    melhor_perda_val = float('inf')

    print(f"[INFO] Iniciando ciclo de {epochs} épocas de treinamento com Lazy Loading...")

    for epoch in range(epochs):
        # --- FASE DE TREINO ---
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        print(f"\n--- Época {epoch+1}/{epochs} ---")
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += targets.size(0)
            correct_train += predicted.eq(targets).sum().item()

            if batch_idx % 50 == 0:
                acc_parcial = 100. * correct_train / total_train
                print(f"Treino | Batch {batch_idx}/{len(train_loader)} | Perda: {loss.item():.4f} | Acurácia: {acc_parcial:.2f}%")

        # --- FASE DE VALIDAÇÃO ---
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                running_val_loss += loss.item()
                _, predicted = outputs.max(1)
                total_val += targets.size(0)
                correct_val += predicted.eq(targets).sum().item()

        epoch_train_loss = running_train_loss / len(train_loader)
        epoch_train_acc = 100. * correct_train / total_train
        epoch_val_loss = running_val_loss / len(val_loader)
        epoch_val_acc = 100. * correct_val / total_val

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        print("-" * 50)
        print(f"[DIAGNÓSTICO ÉPOCA {epoch+1}]")
        print(f"-> TREINO    | Perda Média: {epoch_train_loss:.4f} | Acurácia: {epoch_train_acc:.2f}%")
        print(f"-> VALIDAÇÃO | Perda Média: {epoch_val_loss:.4f} | Acurácia: {epoch_val_acc:.2f}%")
        print("-" * 50)

        # Salva o melhor checkpoint com base na perda de validação robusta
        if epoch_val_loss < melhor_perda_val:
            melhor_perda_val = epoch_val_loss
            caminho_melhor_modelo = os.path.join(pasta_saida, "resnet18_asvspoof_robusto_melhor.pth")
            torch.save(model.state_dict(), caminho_melhor_modelo)
            print(f"[SALVO] Melhor checkpoint robusto persistido.")

    # Salva pesos finais consolidados e histórico
    caminho_modelo_final = os.path.join(pasta_saida, "resnet18_asvspoof_robusto_final.pth")
    torch.save(model.state_dict(), caminho_modelo_final)

    with open(os.path.join(pasta_saida, "historico_treino_robusto.json"), "w") as f:
        json.dump(history, f, indent=4)

    print(f"\n[FIM] Treinamento concluído com estabilidade de memória!")
    gerar_graficos_treinamento(history, epochs, pasta_saida, "resnet18")

def treinar_rawnet2(pasta_features=config.PASTA_FEATURES_RAW, 
                    pasta_saida=config.MODELS_DIR, 
                    epochs=20, 
                    lr=5e-4, 
                    batch_size=config.BATCH_SIZE):
    """
    Treina a rede neural RawNet2 utilizando ondas sonoras brutas 1D.
    """
    os.makedirs(pasta_saida, exist_ok=True)
    print(f"[INFO] Utilizando dispositivo: {config.DEVICE}")

    # Instanciação dos Datasets e Dataloaders
    train_dataset = RawAudioBlocksDataset(pasta_features, usar_treino=True)
    val_dataset = RawAudioBlocksDataset(pasta_features, usar_treino=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Inicialização do Modelo
    print("[INFO] Inicializando RawNet2 Forense...")
    model = RawNet2Forense().to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    melhor_perda_val = float('inf')

    print(f"[INFO] Iniciando regime de {epochs} épocas de treinamento da RawNet2...")

    for epoch in range(epochs):
        # --- FASE DE TREINO ---
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        print(f"\n--- Época {epoch+1}/{epochs} ---")
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Gradient Clipping para estabilização das camadas GRU
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_train_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += targets.size(0)
            correct_train += predicted.eq(targets).sum().item()

            if batch_idx % 50 == 0:
                acc_parcial = 100. * correct_train / total_train
                print(f"RawNet2 | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {acc_parcial:.2f}%")

        # --- FASE DE VALIDAÇÃO ---
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                running_val_loss += loss.item()
                _, predicted = outputs.max(1)
                total_val += targets.size(0)
                correct_val += predicted.eq(targets).sum().item()

        epoch_train_loss = running_train_loss / len(train_loader)
        epoch_train_acc = 100. * correct_train / total_train
        epoch_val_loss = running_val_loss / len(val_loader)
        epoch_val_acc = 100. * correct_val / total_val

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        # Atualiza a taxa de aprendizado dinamicamente
        scheduler.step(epoch_val_loss)

        print("-" * 50)
        print(f"[DIAGNÓSTICO RAWNET2 ÉPOCA {epoch+1}]")
        print(f"-> TREINO    | Perda Média: {epoch_train_loss:.4f} | Acurácia: {epoch_train_acc:.2f}%")
        print(f"-> VALIDAÇÃO | Perda Média: {epoch_val_loss:.4f} | Acurácia: {epoch_val_acc:.2f}%")
        print("-" * 50)

        # Salva o melhor checkpoint da RawNet2
        if epoch_val_loss < melhor_perda_val:
            melhor_perda_val = epoch_val_loss
            caminho_melhor_modelo = os.path.join(pasta_saida, "rawnet2_asvspoof_robusto_melhor.pth")
            torch.save(model.state_dict(), caminho_melhor_modelo)
            print(f"[SALVO] Melhor checkpoint da RawNet2 atualizado.")

    # Salva pesos finais e histórico
    torch.save(model.state_dict(), os.path.join(pasta_saida, "rawnet2_asvspoof_robusto_final.pth"))

    with open(os.path.join(pasta_saida, "historico_treino_rawnet2.json"), "w") as f:
        json.dump(history, f, indent=4)

    print(f"\n[FIM] Treinamento estendido da RawNet2 concluído!")
    gerar_graficos_treinamento(history, epochs, pasta_saida, "rawnet2")


def gerar_graficos_treinamento(history, epochs, pasta_saida, model_name):
    """
    Gera curvas de convergência de perda e evolução de acurácia.
    """
    epochs_range = range(1, epochs + 1)
    
    # 1. Curva de Perda (Loss)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history["train_loss"], 'b-o', label='Perda no Treinamento')
    plt.plot(epochs_range, history["val_loss"], 'r-s', label='Perda na Validação')
    plt.title(f"Curvas de Convergência da Perda ({model_name.upper()})")
    plt.xlabel('Épocas de Treinamento')
    plt.ylabel('Loss Média')
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.tight_layout()
    
    loss_path = os.path.join(pasta_saida, f"loss_convergence_{model_name}.png")
    plt.savefig(loss_path, dpi=300)
    plt.close()

    # 2. Curva de Acurácia (Accuracy)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history["train_acc"], 'b-o', label='Acurácia no Treinamento')
    plt.plot(epochs_range, history["val_acc"], 'r-s', label='Acurácia na Validação')
    plt.title(f"Evolução Temporal da Acurácia ({model_name.upper()})")
    plt.xlabel('Épocas de Treinamento')
    plt.ylabel('Acurácia (%)')
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.tight_layout()

    acc_path = os.path.join(pasta_saida, f"accuracy_evolution_{model_name}.png")
    plt.savefig(acc_path, dpi=300)
    plt.close()

    print(f"[IMAGEM] Gráficos de treinamento salvos em {pasta_saida}")
