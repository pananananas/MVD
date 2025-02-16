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