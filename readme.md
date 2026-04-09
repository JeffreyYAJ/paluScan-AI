
# PaluScan : Détection Automatisée de la Malaria

PaluScan est une solution de santé numérique basée sur l'intelligence artificielle permettant d'identifier le parasite *Plasmodium* dans des images de frottis sanguins. Le projet combine un modèle de Deep Learning léger et une API haute performance pour un diagnostic rapide.

## Fonctionnalités
- **Classification Binaire :** Détecte si une cellule est infectée ou saine.
- **Deep Learning :** Utilise l'architecture **MobileNetV2** (Transfer Learning) pour un équilibre parfait entre précision et légèreté.
- **API REST :** Backend prêt à l'emploi avec **FastAPI**.
- **JSON Output :** Résultats structurés avec score de confiance pour intégration facile (React/Mobile).

## Stack Technique
* **Modélisation :** TensorFlow / Keras, Python.
* **Dataset :** NIH Malaria Dataset (via TFDS).
* **Backend :** FastAPI, Uvicorn.
* **Traitement d'image :** OpenCV, Pillow.

## Performance du Modèle
* **Précision (Accuracy) :** ~95% (après fine-tuning).
* **Vitesse d'inférence :** < 100ms par image sur CPU.
* **Architecture :** MobileNetV2 (pré-entraîné sur ImageNet).

## Installation

1. **Cloner le projet**
   ```bash
   git clone [https://github.com/JeffreyYAJ/PaluScan.git](https://github.com/JeffreyYAJ/PaluScan.git)
   cd PaluScan
   ```

## Installer les dépendances

```bash
pip install -r requirements.txt
```

## Lancer l'API

```bash
python main.py
```

## Utilisation de l'API
Envoyez une requête POST vers /predict avec un fichier image :

Exemple de réponse :

```json
{
  "prediction": "Parasité",
  "confidence": 98.45,
  "status": "success",
  "model_version": "v1.0"
}
```
