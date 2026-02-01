# Projekt-Installation

Dieses Repository enthält die Implementierung der Experimente im Rahmen der Masterarbeit: Wasserstein-GANs: Eine theoretische und empirische Untersuchung.

## 1. Voraussetzungen
* **Python 3.12**
* **NVIDIA GPU** (optional für CUDA-Support)

## 2. Setup (Windows PowerShell)

Virtuelle Umgebung einrichten und die Abhängigkeiten installieren:

```powershell
# Umgebung erstellen
py -3.12 -m venv .venv

# Umgebung aktivieren
.\.venv\Scripts\Activate.ps1

# Installation der Pakete
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Ausführen der Experimente

### Gaus:
```powershell
python Experimente/scripts/gauss/single_theta_gauss.py
python Experimente/scripts/gauss/multiple_theta_gauss.py
```
### 2D-Kegelverteilung:
```powershell
python Experimente/scripts/cone2d/single_theta_cone2d.py
```
### Hochdimensionale Kegelverteilung:
```powershell
python Experimente/scripts/conedirac/single_theta_conedirac.py
```





