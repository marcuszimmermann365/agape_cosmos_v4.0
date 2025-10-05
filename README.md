[README.md](https://github.com/user-attachments/files/22706618/README.md)
# Agape Cosmos V4.0 Simulation

Dieses Paket enthält den vollständigen Python-Code aus Appendix A des Agape Cosmos V4.2-Papiers.

## Struktur
- `config.py` : Konfiguration der Simulation und Hyperparameter
- `objectives.py` : Definition der Agape-Zielfunktionen J1–J5
- `models.py` : Definition der Jiva-Agenten und des kosmischen Substrats
- `simulation.py` : Hauptklasse für die Multi-Agenten-Simulation

## Voraussetzungen
- Python 3.9+
- PyTorch 2.1+
- torchsde 0.2.5+
- NumPy 1.23+

Optional:
- GPU (z.B. NVIDIA RTX 4080 empfohlen)

## Installation
```bash
pip install torch torchsde numpy
```

## Ausführung
```bash
python simulation.py
```

## Hinweise
- Die Simulation nutzt eine Stochastic Runge-Kutta-Integration für Stratonovich-SDEs.
- Für reproduzierbare Ergebnisse können Seeds für NumPy und PyTorch gesetzt werden.
