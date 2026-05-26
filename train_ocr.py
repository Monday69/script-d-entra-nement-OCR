"""
train_ocr.py — Fine-tuning TrOCR (Version Ultimate 2026)
========================================================
Optimisé pour RTX 4070 / Windows / Dataset Hardcore V3
"""

import os
import torch
import evaluate
from datasets import load_dataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    EarlyStoppingCallback,
)

# CONFIGURATION GLOBALE
# Définit les chemins des données et des modèles ainsi que les hyperparamètres.

DATA_DIR       = "C:/yolo/final_dataset"
OUTPUT_DIR     = "./trocr_books_model"  # Nouveau dossier de sortie pour ce run
FINAL_DIR      = "./trocr_books_model_final"
MODEL_BASE     = "microsoft/trocr-base-printed"

MAX_LABEL_LEN  = 128    # Assez grand pour les titres à rallonge
BATCH_SIZE     = 4      # Sécurité OOM pour 12Go VRAM
GRAD_ACCUM     = 4      # Batch effectif de 16
LEARNING_RATE  = 1e-6
NUM_EPOCHS     = 30
EVAL_STEPS     = 50
SAVE_STEPS     = 50


# 1. VÉRIFICATION DU DATASET

def verify_dataset(data_dir: str):
    """Vérifie la présence et la validité du dataset.

    Cette fonction s'assure que le fichier metadata_balanced.jsonl existe,
    qu'il contient au moins une entrée, et que chaque entrée a bien les
    champs attendus : file_name et text.
    """
    print("\n[1/6] Audit de sécurité du dataset...")
    meta_path = os.path.join(data_dir, "metadata_balanced.jsonl")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"metadata_balanced.jsonl introuvable dans {data_dir}.")
    
    # Lecture du JSONL ligne par ligne pour construire la liste des exemples
    with open(meta_path, encoding="utf-8") as f:
        import json
        lines = [json.loads(l) for l in f if l.strip()]
    
    if not lines:
        raise ValueError("metadata_balanced.jsonl est vide.")
    
    first = lines[0]
    if "file_name" not in first or "text" not in first:
        raise ValueError(f"Format JSONL incorrect. Attendu: {{file_name, text}}.")
    
    print(f"      [OK] {len(lines)} entrées détectées.")
    print(f"      [OK] Exemple : {first['file_name']} → \"{first['text'][:50]}...\"")
    return len(lines)


# 2. MÉTRIQUE CER (Character Error Rate)


def build_compute_metrics(processor):
    """Construit une fonction de métriques pour le trainer.

    La fonction retournée est utilisée par Hugging Face Trainer pour calculer
    le Character Error Rate (CER) à partir des prédictions et des labels.
    """
    # Chargement de la métrique de Character Error Rate (CER)
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        labels_ids  = pred.label_ids
        pred_ids    = pred.predictions

        # Décodage des prédictions en texte lisible
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)

        # Replace les positions masquées (-100) par le token de padding pour décoder correctement
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

        cer = cer_metric.compute(predictions=pred_str, references=label_str)

        print("\n--- Aperçu des prédictions (Validation) ---")
        for pred_s, label_s in zip(pred_str[:3], label_str[:3]):
            print(f"  Modèle : {pred_s}")
            print(f"  Réel   : {label_s}")
            print("-" * 30)

        return {"cer": cer}

    return compute_metrics



# 3. PIPELINE DE DONNÉES

def build_transform(processor):
    """Retourne une fonction de transformation pour le dataset.

    Cette transformation prépare les images et les textes pour l'entraînement.
    Les images sont converties en tenseurs GPU-ready, et les labels sont tokenizés
    avec padding et troncature. Les tokens de padding sont remplacés par -100
    pour que la fonction de perte les ignore.
    """
    # Cette fonction retourne un transformeur compatible avec les datasets HF
    def process_data(examples):
        # Convertit les images en RGB et calcule les tenseurs d'entrée
        images = [img.convert("RGB") for img in examples["image"]]
        pixel_values = processor(images=images, return_tensors="pt").pixel_values

        # Tokenize le texte de sortie avec padding et troncature
        labels = processor.tokenizer(
            examples["text"],
            padding="max_length",
            max_length=MAX_LABEL_LEN,
            truncation=True,
        ).input_ids

        labels_clean = [
            [tok if tok != processor.tokenizer.pad_token_id else -100 for tok in label]
            for label in labels
        ]

        return {"pixel_values": pixel_values, "labels": labels_clean}

    return process_data


# 4. MAIN

def main():
    """Point d'entrée principal du script.

    Cette fonction orchestre l'ensemble du pipeline : vérification du dataset,
    chargement des données, préparation du dataset, configuration du modèle,
    entraînement et sauvegarde du modèle final.
    """
    print("=" * 60)
    print("Lancement du Fine-Tuning TrOCR")
    print("=" * 60)
    
    nb_images = verify_dataset(DATA_DIR)
    
    # Chargement
    # Chargement personnalisé depuis JSONL + dossiers d'images
    print("\n[2/6] Chargement et découpage du dataset...")
    
    import json
    from datasets import Dataset, Image as HFImage
    
    # Lire le fichier de métadonnées qui liste les images et les textes associés
    metadata_path = os.path.join(DATA_DIR, "metadata_balanced.jsonl")
    with open(metadata_path, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    
    images_paths = []
    texts = []
    for entry in entries:
        img_path = os.path.join(DATA_DIR, entry["file_name"])
        if os.path.exists(img_path):
            images_paths.append(img_path)
            texts.append(entry["text"])
        else:
            print(f"Image manquante : {img_path}")
    
    # Création du dataset  à partir des chemins d'images et des textes
    dataset = Dataset.from_dict({
        "image": images_paths,
        "text": texts
    }).cast_column("image", HFImage())
    
    # Séparation en ensembles train / validation
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    
    train_dataset = dataset["train"]
    eval_dataset  = dataset["test"].shuffle(seed=42).select(range(min(1500, len(dataset["test"]))))
    
    print(f"      Train: {len(train_dataset)} images | Eval: {len(eval_dataset)} images (Optimisé)")

    # Modèle
    print(f"\n[3/6] Chargement de l'architecture {MODEL_BASE}...")
    processor = TrOCRProcessor.from_pretrained(MODEL_BASE)
    model     = VisionEncoderDecoderModel.from_pretrained(MODEL_BASE)

    # Gel de l'encodeur vision : seul le décodeur sera affiné
    for param in model.encoder.parameters():
        param.requires_grad = False
    print("      [OK] Encodeur vision gelé (transfer learning optimisé).")
    
    # 1. Configuration de base pour l'entraînement
    # On s'assure que le modèle utilise les bons tokens de début et de padding
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id           = processor.tokenizer.pad_token_id
    model.config.vocab_size             = model.config.decoder.vocab_size
    
    # 2. Configuration de génération pour l'évaluation et l'inférence
    # On fixe la longueur max et la stratégie de génération beam search
    model.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.generation_config.pad_token_id           = processor.tokenizer.pad_token_id
    model.generation_config.max_length             = MAX_LABEL_LEN
    model.generation_config.no_repeat_ngram_size   = 3
    model.generation_config.num_beams              = 4
    
    # Transforms
    print("\n[4/6] Application des transformations...")
    process_data = build_transform(processor)
    train_dataset.set_transform(process_data)
    eval_dataset.set_transform(process_data)
    
    # Arguments d'entraînement
    print("\n[5/6] Configuration des paramètres de combat...")
    # Paramètres du trainer HF : gestion du checkpoint, métriques, batchs et FP16
    training_args = Seq2SeqTrainingArguments(
            output_dir                  = OUTPUT_DIR,
            predict_with_generate       = True,
            eval_strategy               = "epoch",
            save_strategy               = "epoch",
            load_best_model_at_end      = True,
            metric_for_best_model       = "cer",
            greater_is_better           = False,
            per_device_train_batch_size = BATCH_SIZE,
            per_device_eval_batch_size  = 1,        # ← Changement clé
            gradient_accumulation_steps = GRAD_ACCUM,
            fp16                        = torch.cuda.is_available(),
            dataloader_num_workers      = 0,
            num_train_epochs            = NUM_EPOCHS,
            learning_rate               = LEARNING_RATE,
            warmup_ratio                = 0.1,
            logging_steps               = 10,
            remove_unused_columns       = False,
        )
        
    # Création du trainer Hugging Face qui va gérer l'entraînement et l'évaluation
    trainer = Seq2SeqTrainer(
        model            = model,
        processing_class = processor,
        args             = training_args,
        train_dataset    = train_dataset,
        eval_dataset     = eval_dataset,
        data_collator    = default_data_collator,
        compute_metrics  = build_compute_metrics(processor),
        callbacks        = [EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    print("\n[6/6] Analyse des Checkpoints...")
    # Recherche d'un checkpoint existant pour reprendre l'entraînement
    checkpoints = [d for d in os.listdir(OUTPUT_DIR) if "checkpoint" in d] if os.path.exists(OUTPUT_DIR) else []
    
    # Si des checkpoints existent, on reprend l'entraînement là où il s'est arrêté
    if checkpoints:
        print(f"\n Checkpoint détecté ! Reprise de l'entraînement là où il s'est arrêté.")
        trainer.train(resume_from_checkpoint=True)
    else:
        print(f"\n DÉMARRAGE DU RUN (Laisse ta RTX 4070 chauffer tranquillement)")
        trainer.train()
    
    # Sauvegarde du modèle et du processeur pour pouvoir réutiliser le modèle plus tard
    print(f"\n Sauvegarde du modèle final dans {FINAL_DIR}...")
    trainer.save_model(FINAL_DIR)
    processor.save_pretrained(FINAL_DIR)

if __name__ == "__main__":
    main()
