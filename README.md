Ce dépôt contient les scripts nécessaires à la génération de données synthétiques et au fine-tuning du modèle TrOCR pour la reconnaissance de caractères sur des tranches de livres. Ces codes sont référencés dans l'Annexe E du rapport de projet Library Scanner.

Structure du Dépôt
synthetic_factory.py : Script de génération d'images synthétiques.

train_ocr.py : Script d'entraînement (fine-tuning) du modèle TrOCR.

DATASET_OCR_MASTER.csv : (Requis) Fichier CSV contenant les métadonnées (titres, auteurs) servant de base à la génération.

fonts/ : (Requis) Dossier contenant les polices TrueType (.ttf) pour la génération.

Synthèse des Scripts
1. synthetic_factory.py (Génération de Données)
Ce script génère un dataset massif de tranches de livres réalistes à partir de textes bruts. Il vise à simuler les conditions difficiles de prise de vue en bibliothèque (NIGHTMARE MODE).

Fonctionnalités clés :

Génération massive : Configuré pour produire jusqu'à 76 500 images uniques.

Variabilité Typographique : Sélection aléatoire de polices et gestion des mises en page (horizontale, verticale, inversée).

Augmentation "Hardcore" : Application de dégradations réalistes :

Bruit numérique important.

Reflets plastiques type flash.

Ombres portées et rayures d'usure.

Variations de contraste et de luminosité.

Sortie : Images JPEG et un fichier metadata.jsonl compatible avec le loader Hugging Face.

2. train_ocr.py (Fine-tuning TrOCR)
Ce script orchestre l'entraînement supervisé du modèle TrOCR (microsoft/trocr-base-printed) sur le dataset synthétique généré.

Configuration et Optimisations (RTX 4070 / Windows) :

Architecture : Vision-Encoder-Decoder (TrOCR).

Transfer Learning : Encodeur vision gelé (requires_grad = False) ; seul le décodeur est affiné pour optimiser les ressources et le temps.

Gestion Mémoire : Taille de batch réduite (4) combinée à l'accumulation de gradients (4) pour un batch effectif de 16, évitant les erreurs Out-Of-Memory (12Go VRAM).

Précision Mixte (FP16) : Activée si CUDA est disponible pour accélérer le calcul.

Entraînement : Utilisation du Seq2SeqTrainer de Hugging Face avec arrêt précoce (EarlyStopping).

Métrique : Évaluation basée sur le CER (Character Error Rate).

Utilisation Rapide
Prérequis : Installer les dépendances (PIL, pandas, numpy, torch, transformers, datasets, evaluate).

Génération : Placer DATASET_OCR_MASTER.csv et le dossier fonts/ à la racine, puis lancer :

Bash
python synthetic_factory.py
Les images seront générées dans synth_crops_v3_hardcore/.

Entraînement : Ajuster le chemin DATA_DIR dans train_ocr.py si nécessaire, puis lancer :

Bash
python train_ocr.py
Le modèle final sera sauvegardé dans trocr_books_model_final/.
