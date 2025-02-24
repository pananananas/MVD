# Plan działania

1. Przygotowanie danych
- pobranie większej ilości próbek z CO3D
- dołożenie danych z innych datasetów
- preprocessing:
  - zamiast pair-wise generowanie widoków na podstawie jednej próbki
  - generowanie kolejnych widoków na podstawie kilku próbek
- dodanie warunkowania z użyciem SfM

2. Modyfikacje modelu
- Zbadanie poprawności embeddingów parametrów kamer
- implementacja image conditioning 
- dodanie dodatkowej atencji w UNET do warunkowania na parametrach kamer
- dodanie dodatkowej atencji w UNET do image conditioning

3. Trening 
- lightning integration, wandb config and sweeps
- multi-gpu
- feature matching loss (between generated and source image)
- integracja Hydra z WANDB do zarządzania konfiguracją i logowania

# Zbadane rzeczy
Konkatenacja embeddingów kamer z promptem tekstowym nie daje dobrych wyników.


# Użyte modele
- [stable-diffusion-v1-5](https://huggingface.co/Jiali/stable-diffusion-1.5)
- [stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1)


# Użyte datasety
| Dataset | Description | Object/Scene |Access |
|---------|-------------|--------|--------|
| [MVImgNet](https://github.com/GAP-LAB-CUHK-SZ/MVImgNet/tree/main)     | Czekam na mail z hasłem dostępu | O | n |
| [CO3D](https://ai.meta.com/datasets/co3d-downloads/)                  | Bardzo duży, mam już dostęp, obrazy średniej jakości | O | y |
| [RealEstate10K](https://google.github.io/realestate10k/download.html) | Filmy z yt, trzeba pobrać i przetworzyć (nie koncentrują się na obiektach a przestrzeniach) | S | y |
| [RTMV](https://www.cs.umd.edu/~mmeshry/projects/rtmv/)                | Obiecujący, ale link nie działa | O | n |


# Diagram test
```mermaid
  graph TD;
      A-->B;
      A-->C;
      B-->D;
      C-->D;
```