# Installation

## Voraussetzungen
- Python 3.12 empfohlen
- Optional: NVIDIA GPU + aktueller Treiber (für CUDA)

### Virtuelle Umgebung erstellen
- py -3.12 -m venv .venv
- .\.venv\Scripts\Activate.ps1

### Abhängigkeiten Installieren
- python -m pip install --upgrade pip
- pip install -r requirements.txt

# Ausführen der Experimente

## Gauss:
- python Experimente/scripts/gauss/single_theta_gauss.py
- python Experimente/scripts/gauss/multiple_theta_gauss.py

## 2D-Kegelverteilung:
- python Experimente/scripts/cone2d/single_theta_cone2d.py

## Hochdimensionale Kegelverteilung:
- python Experimente/scripts/conedirac/single_theta_conedirac.py





