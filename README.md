# PUC Minas - Framework Forense Híbrido Autônomo

Este repositório contém a conversão do notebook do Google Colab do TCC em uma aplicação de terminal em Python (CLI) profissional e modularizada. 

O framework realiza a análise de autenticidade de gravações de áudio para detecção de deepfakes de voz (síntese de fala/clonagem), aplicando tanto inteligência artificial profunda (ResNet-18 e RawNet2) quanto perícia acústica tradicional (extração de invariantes da laringe via Praat-Parselmouth).

---

## Estrutura do Projeto

* `main.py`: Entrada principal do programa (interface CLI).
* `requirements.txt`: Dependências do Python.
* `data/`: Diretório padrão para os arquivos de dados (metadados, blocos `.npy`, áudios).
* `models/`: Diretório de armazenamento de checkpoints e histórico dos modelos (`.pth`, `.json`).
* `output/`: Relatórios gerados em PDF e gráficos de treinamento.
* `src/`:
  * `config.py`: Variáveis de caminhos e parâmetros globais.
  * `augment.py`: Simulação pericial de degradações de canal móvel (`AdaptadorAcusticoForense`).
  * `dataset.py`: Manipulador de datasets com Lazy Loading (baixo consumo de RAM).
  * `models.py`: Arquiteturas neurais `ResNet-18` e `RawNet2Forense`.
  * `feature_extraction.py`: Esteira de exportação de dados em blocos.
  * `training.py`: Código para treino e geração de gráficos de convergência.
  * `evaluation.py`: Cálculo de métricas forenses científicas (EER e Cllr).
  * `praat_metrics.py`: Mapeamento de biometria vocal (Jitter, Shimmer, HNR, F1, F2).
  * `report.py`: Compilador oficial de laudos periciais em PDF (ReportLab).
  * `inference.py`: Pipeline de perícia integrada de áudio único.

---

## Como Preparar a Base de Dados (ASVspoof 2021 LA)

Para que o framework execute a extração de características e o treinamento automaticamente sem necessidade de qualquer alteração de código, você deve preparar a base de dados do desafio **ASVspoof 2021 Logical Access (LA)**:

1. **Obtenção dos Áudios (.flac)**:
   - Baixe o conjunto de dados oficial de avaliação (Evaluation Set) no site oficial do desafio [ASVspoof](https://www.asvspoof.org) ou diretamente pela página do Zenodo: [Zenodo ASVspoof 2021 LA Dataset](https://zenodo.org/records/4837263).
   - Extraia e copie os arquivos de áudio `.flac` para a pasta: `./data/audios/`

2. **Obtenção dos Metadados/Labels (trial_metadata.txt)**:
   - Baixe as chaves/metadados oficiais de avaliação do repositório do desafio: [LA CM Trial Metadata](https://raw.githubusercontent.com/asvspoof-challenge/2021/main/LA/CM/trial_metadata.txt).
   - Salve o arquivo com o nome `trial_metadata.txt` diretamente na pasta: `./data/`

Após colocar estes arquivos em suas devidas pastas, os caminhos configurados em `src/config.py` farão o mapeamento e a resolução de arquivos de forma 100% automatizada.

---

## Instalação

### Pré-requisitos
1. **Python 3.8+**
2. **FFmpeg**: Necessário para decodificar arquivos `.m4a` e outros formatos comprimidos. Certifique-se de que o executável `ffmpeg` está no PATH do seu sistema operacional.

### Passo a Passo

1. **Abra o terminal no diretório do projeto e crie um ambiente virtual**:
   ```bash
   py -m venv venv
   ```

2. **Ative o ambiente virtual**:
   * **Windows (Prompt de Comando)**:
     ```cmd
     venv\Scripts\activate.bat
     ```
   * **Windows (PowerShell)**:
     ```powershell
     .\venv\Scripts\activate.ps1
     ```
   * **Linux/macOS**:
     ```bash
     source venv/bin/activate
     ```

3. **Instale as dependências**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Como Usar (Comandos CLI)

### 1. Extração de Características
Prepara os espectrogramas Mel 2D ou as ondas raw 1D, aplicando a degradação de canal acústico para robustez e salvando em blocos paritários:

* **Para Espectrogramas Mel (ResNet18)**:
  ```bash
  python main.py extract-features --type spectrogram --metadata ./data/trial_metadata.txt --audio-dir ./data/audios
  ```
* **Para Onda Waveform Bruta (RawNet2)**:
  ```bash
  python main.py extract-features --type raw --metadata ./data/trial_metadata.txt --audio-dir ./data/audios
  ```

### 2. Treinamento
Treina a rede profunda selecionada apontando para a pasta onde os arquivos `.npy` foram salvos:

* **Treinar ResNet-18 (5 épocas por padrão)**:
  ```bash
  python main.py train --model resnet18
  ```
* **Treinar RawNet2 (20 épocas por padrão)**:
  ```bash
  python main.py train --model rawnet2
  ```

*Nota: Os gráficos de evolução do treinamento e os melhores pesos serão salvos automaticamente em `models/` e `output/`.*

### 3. Avaliação de Métricas Forenses
Calcula a taxa de erro igualitário (EER) e o Cllr a partir do conjunto de validação:

```bash
python main.py evaluate --model resnet18
python main.py evaluate --model rawnet2
```

### 4. Perícia Integrada em Áudio Único (Laudo PDF)
Realiza a perícia individual de uma gravação. O áudio é analisado pelas redes profundas, passa pela auditoria biométrica glótica e um laudo pericial oficial em PDF com assinatura e cadeia de custódia é gerado:

```bash
python main.py pericia --audio ./data/audios/LA_E_1000153.flac
```
O PDF final será salvo na pasta `output/` como `Laudo_Pericial_LA_E_1000153.pdf`.

### 5. Estatísticas em Lote por Veracidade
Lê todos os áudios comprimidos de uma pasta, extrai suas invariantes biológicas de laringe, calcula médias agrupando pelo termo `Verdade` ou `Mentira` contido no nome dos arquivos e gera uma tabela do TCC exportável:

```bash
python main.py batch-stats --dir ./data/audios/teste
```
A tabela compilada em CSV será exportada em `output/medias_tcc_fase2.csv`.
