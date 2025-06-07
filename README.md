# TP Renforcement Informatique - Deep Learning & Computer Vision

Ce repo contient les travaux pratiques sur l'apprentissage profond et la vision par ordinateur, organisés en trois exercices principaux couvrant MNIST, AudioMNIST et l'analyse de sensibilité des réseaux convolutifs.

### Structure

```
├── ex1_mnist/          # Classification MNIST avec CNN
├── ex2_AudioMNIST/     # Reconnaissance audio et multimodale
├── ex3_CAM/            # Class Activation Mapping
├── requirements.txt    # Dépendances Python
├── Sujet.pdf
└── RELAVE_TROILLARD_RAPPORT_TP.pdf
```

## Exercice 1 : Classification MNIST
Description

Implémentation et analyse d'un réseau de neurones convolutif (CNN) pour la classification des chiffres manuscrits MNIST.

Contenu

- `main.py` : Script principal d'entraînement et de test
- `plots.ipynb` : Visualisation des résultats
- `exports/` : Modèles sauvegardés et graphiques

Objectifs

- Analyser l'évolution de la loss pendant l'apprentissage
- Comparer les performances train/test avec différentes configurations
- Analyser l'effet du dropout sur le surapprentissage

Utilisation

```bash
cd ex1_mnist
python main.py
```

Paramètres disponibles :

- `--batch-size` : Taille des lots d'entraînement
- `--test-batch-size` : Taille des lots de test
- `--epochs` : Nombre d'époques
- `--lr` : Taux d'apprentissage
- `--gamma` : Facteur de décroissance du learning rate
- `--no-cuda` : Désactiver CUDA
- `--dropout` : Activer/désactiver le dropout
- `--save-model` : Sauvegarder le modèle

## Exercice 2 : AudioMNIST et Multimodalité

Description

Reconnaissance de chiffres prononcés à partir d'enregistrements audio et proposition d'un système multimodal combinant vision et audio.

Contenu

- `AudioMNISTLoader.py` : Chargement des données audio
- `ShowAudioFile.py` : Visualisation des signaux audio
- `SplitTrainTest_original.py` : Division train/test
- `perception_acoustique.ipynb` : Notebook principal
- `recordings/`: Fichiers audio WAV (3000 échantillons)

Utilisation

```bash
cd ex2_AudioMNIST
# Visualiser un fichier audio
python ShowAudioFile.py recordings/0_george_0.wav --spectrogram

# Générer la division train/test
python SplitTrainTest_original.py
```

Format des fichiers audio

Les fichiers suivent le format : [chiffre]_[locuteur]_[répétition].wav

- Taux d'échantillonnage : 8000 Hz
- Durée : ~0.3-0.35 secondes
- 6 locuteurs différents

## Exercice 3 : Class Activation Mapping (CAM)

Description

Analyse de la sensibilité des réseaux convolutifs par visualisation des cartes d'activation de classe sur des images ImageNet.

Contenu

- `Home.py` : Interface Streamlit principale
- `cam_utils.py` : Utilitaires pour les méthodes CAM
- `image_augmenters.py` : Transformations d'images
- Images de test : tennis_ball.jpg, tennis_balls.jpg, zebra.jpg
- `torch-cam/` : Bibliothèque CAM
- `pages/` : Pages additionnelles Streamlit

Objectifs

- Implémenter différentes méthodes CAM (GradCAM, GradCAM++, ScoreCAM)
- Analyser l'activation sur trois types d'images :
    - Une image avec un seul objet de la classe cible
    - Une image avec plusieurs objets de la classe cible
    - Une image sans objet de la classe cible
- Étudier la robustesse aux transformations (luminosité, perspective)

Utilisation

```bash
cd ex3_CAM
streamlit run Home.py
```

## Installation

- Python 3.8+
- CUDA (optionnel, pour accélération GPU)

Installation des dépendances

```bash
pip install -r requirements.txt
```

Installer [Pytorch](https://pytorch.org/get-started/locally/) selon vos préférences et votre materiel

## Auteurs

- Dorian RELAVE ([Legolaswash](https://github.com/Legolaswash))
- Romain TROILLARD ([MrPaquiteau](https://github.com/MrPaquiteau))