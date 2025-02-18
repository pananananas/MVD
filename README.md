# Plan działania

1. Przygotowanie danych
- zestaw obrazów z różnych punktów widzenia
- COLMAP do pozyskania macierzy rotacji i translacji pomiędzy poszczególnymi punktami widzenia i punktów kluczowych obiektu w 3D

2. Minimalna modyfikacja modelu
- najpierw modyfikacja tylko przekazywania informacji o kamerze
- encoding parametrów
- sprawdzenie czy model przyjmuje dane

3. Trening 
- na małym zbiorze danych
- prosty loss np L2 pomiędzy wygenerowanym a prawdziwym obrazem
- początkowo tylko nowe warstwy, pozostałe warstwy zostają zamrożone


# Pomysły
- feature matching loss


# Modele
- [stable-diffusion-v1-5](https://huggingface.co/Jiali/stable-diffusion-1.5)



# Datasets
| Dataset | Description | Access |
|---------|-------------|--------|
| [MVImgNet](https://github.com/GAP-LAB-CUHK-SZ/MVImgNet/tree/main)     | Czekam na mail z hasłem dostępu | n |
| [CO3D](https://ai.meta.com/datasets/co3d-downloads/)                  | Bardzo duży, mam już dostęp | y |
| [RealEstate10K](https://google.github.io/realestate10k/download.html) | Filmy z yt, trzeba pobrać i przetworzyć | y |
| [RTMV](https://www.cs.umd.edu/~mmeshry/projects/rtmv/)                | Obiecujący, ale link nie działa | n |



# Diagram in markdown
```mermaid
  graph TD;
      A-->B;
      A-->C;
      B-->D;
      C-->D;
```