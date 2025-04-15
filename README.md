# Multi-View Diffusion

## Installation

install env with uv package manager:

```bash
uv sync
```

## Training

run training:

```bash
uv run train.py
```

on athena:

```bash
sbatch train.sh
```

## Inference

run inference:

```bash
uv run infer.py --checkpoint path/to/checkpoint.ckpt
```

<!--

# Plan działania

1. Przygotowanie danych
- pobranie danych z Objaverse-XL
- dołożenie danych z innych datasetów
- preprocessing:
  - zamiast pair-wise generowanie widoków na podstawie jednej próbki
  - generowanie kolejnych widoków na podstawie kilku próbek
- dodanie warunkowania z użyciem SfM
- znaleźć benchmarkowe dane

2. Modyfikacje modelu
- sprawdzenie poprawności embeddingów parametrów kamer
- implementacja image conditioning
- dodanie dodatkowej atencji w UNET do warunkowania na parametrach kamer
- dodanie dodatkowej atencji w UNET do image conditioning

3. Trening
- lightning integration, wandb config and sweeps
- multi-gpu
- feature matching loss (between generated and source image)
- integracja Hydra z WANDB do zarządzania konfiguracją i logowania


# Użyte datasety
| Dataset | Description | Object/Scene |Access |
|---------|-------------|--------|--------|
| [Objaverse-XL](https://objaverse.allenai.org/)                        | Obiekty 3D z różnych źródeł | O | y |
| [MVImgNet](https://github.com/GAP-LAB-CUHK-SZ/MVImgNet/tree/main)     | Czekam na mail z hasłem dostępu | O | n |
| [CO3D](https://ai.meta.com/datasets/co3d-downloads/)                  | Bardzo duży, mam już dostęp, obrazy średniej jakości | O | y |
| [RealEstate10K](https://google.github.io/realestate10k/download.html) | Filmy z yt, trzeba pobrać i przetworzyć (nie koncentrują się na obiektach a przestrzeniach) | S | y |
| [RTMV](https://www.cs.umd.edu/~mmeshry/projects/rtmv/)                | Obiecujący, ale link nie działa | O | n |

-->
