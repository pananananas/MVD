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
- Zbadanie czy model jest w stanie generować kolejne widoki po dłuższym treningu
- Czy CLIP jest nam potrzebny, co się stanie jak go usuniemy

3. Trening 
- multi-gpu
- feature matching loss
- trenowanie nie tylko nowych warstw ale fine tuning całości/LoRA adapter


# Zbadane rzeczy
Konkatenacja embeddingów kamer z promptem tekstowym nie daje dobrych wyników.


# Użyte modele
- [stable-diffusion-v1-5](https://huggingface.co/Jiali/stable-diffusion-1.5)


# Użyte datasety
| Dataset | Description | Access |
|---------|-------------|--------|
| [MVImgNet](https://github.com/GAP-LAB-CUHK-SZ/MVImgNet/tree/main)     | Czekam na mail z hasłem dostępu | n |
| [CO3D](https://ai.meta.com/datasets/co3d-downloads/)                  | Bardzo duży, mam już dostęp, obrazy średniej jakości | y |
| [RealEstate10K](https://google.github.io/realestate10k/download.html) | Filmy z yt, trzeba pobrać i przetworzyć (nie koncentrują się na obiektach a przestrzeniach) | y |
| [RTMV](https://www.cs.umd.edu/~mmeshry/projects/rtmv/)                | Obiecujący, ale link nie działa | n |


# Diagram test
```mermaid
  graph TD;
      A-->B;
      A-->C;
      B-->D;
      C-->D;
```