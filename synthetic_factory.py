"""synthetic_factory.py

Génère des tranches de livres synthétiques pour l'entraînement OCR.
Chaque image est produite avec du texte vertical/horizontal, bruit, reflets,
ombres de rayures, et variations de contraste, puis enregistrée avec
une métadonnée JSONL pour l'entraînement Hugging Face.
"""

import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import random, os, json, textwrap

# CONFIGURATION GLOBALE
# Chemins d'entrée et de sortie, et paramètres de génération.
CSV_PATH = "DATASET_OCR_MASTER.csv"  # Fichier source des titres et auteurs
OUTPUT_DIR = "synth_crops_v3_hardcore"  # Dossier de sortie des images générées
JSONL_PATH = os.path.join(OUTPUT_DIR, "metadata.jsonl")  # Fichier de métadonnées
FONTS_DIR = "fonts"  # Dossier de polices TrueType
NB_IMAGES_A_GENERER = 76500  # Nombre total d'images synthétiques à créer

os.makedirs(OUTPUT_DIR, exist_ok=True)
# Liste de toutes les polices .ttf disponibles pour varier l'apparence du texte
polices_disponibles = [os.path.join(FONTS_DIR, f) for f in os.listdir(FONTS_DIR) if f.endswith('.ttf')]

def ajouter_bruit(image):
    """Ajoute du bruit numérique aléatoire à l'image.

    Le bruit est généré avec une amplitude plus élevée que d'habitude pour
    simuler des photos de mauvaise qualité, des scanners sales ou des textures
    de papier irrégulières.
    """
    np_image = np.array(image)
    bruit = np.random.randint(-25, 25, np_image.shape, dtype='int16')  # large amplitude de bruit
    return Image.fromarray(np.clip(np_image + bruit, 0, 255).astype('uint8'))


def ajouter_effets_hardcore(image):
    """Ajoute des effets visuels réalistes aux tranches de livre.

    Cette fonction superpose des ombres, des reflets et des rayures puis
    modifie contraste et luminosité pour un rendu plus dur et plus réaliste.
    """
    w, h = image.size
    overlay = Image.new('RGBA', (w, h), (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # 1. Ombre latérale pour simuler un interstice ou une tranche en coin
    if random.random() > 0.3:
        largeur_ombre = random.randint(5, int(w * 0.2))
        intensite = random.randint(100, 220)
        if random.choice([True, False]):
            draw.rectangle([(0, 0), (largeur_ombre, h)], fill=(0, 0, 0, intensite))
        else:
            draw.rectangle([(w - largeur_ombre, 0), (w, h)], fill=(0, 0, 0, intensite))

    # 2. Reflet plastique / flash blanc intense
    if random.random() > 0.4:
        x_start = random.randint(-50, w)
        epaisseur = random.randint(20, 80)
        intensite_reflet = random.randint(40, 100)
        draw.polygon([
            (x_start, 0), (x_start + epaisseur, 0),
            (x_start + epaisseur + 40, h), (x_start + 40, h)
        ], fill=(255, 255, 255, intensite_reflet))

    # 3. Rayures d'usure pour simuler du papier ou de la couverture abîmée
    if random.random() > 0.5:
        for _ in range(random.randint(1, 6)):
            y_pos = random.randint(0, h)
            draw.line(
                [(0, y_pos), (w, y_pos + random.randint(-10, 10))],
                fill=(255, 255, 255, random.randint(30, 90)),
                width=random.randint(1, 3)
            )

    image = image.convert('RGBA')
    image = Image.alpha_composite(image, overlay).convert('RGB')

    # 4. Jitter de contraste et luminosité pour diversifier l'éclairage
    enhancer_c = ImageEnhance.Contrast(image)
    image = enhancer_c.enhance(random.uniform(0.6, 1.4))

    enhancer_b = ImageEnhance.Brightness(image)
    image = enhancer_b.enhance(random.uniform(0.5, 1.2))

    return image

def text_to_image(texte, police_path, max_width, max_height, vertical=False):
    """Génère une image texte adaptée à une zone donnée.

    Le texte est automatiquement renvoyé et redimensionné pour rentrer dans les
    dimensions max_width x max_height. Si vertical=True, le texte est tourné à 90°.
    """
    taille = 55
    padding = 15

    while taille > 10:
        font = ImageFont.truetype(police_path, taille)
        avg_char_w = taille * 0.55
        chars_per_line = max(8, int(max_width / avg_char_w))

        # Emballe le texte sur plusieurs lignes pour respecter la largeur
        lignes = textwrap.wrap(texte, width=chars_per_line)
        texte_wrap = "\n".join(lignes)

        temp_img = Image.new('RGBA', (1000, 1000))
        temp_draw = ImageDraw.Draw(temp_img)

        bbox = temp_draw.multiline_textbbox((0, 0), texte_wrap, font=font, align="center")
        w_text, h_text = bbox[2] - bbox[0], bbox[3] - bbox[1]

        # Si le texte rentre dans la zone, on conserve la taille de police
        if w_text <= max_width and h_text <= max_height:
            break

        # Sinon on réduit progressivement la taille de police
        taille -= 2

    couleur_texte = (
        random.randint(150, 255),
        random.randint(150, 255),
        random.randint(150, 255)
    )

    txt_img = Image.new(
        'RGBA',
        (int(w_text) + padding * 2, int(h_text) + padding * 2),
        (255, 255, 255, 0)
    )
    d = ImageDraw.Draw(txt_img)
    d.multiline_text(
        (int(padding - bbox[0]), int(padding - bbox[1])),
        texte_wrap,
        font=font,
        fill=couleur_texte,
        align="center"
    )

    # Rotation verticale optionnelle pour les tranches de livre verticales
    if vertical:
        txt_img = txt_img.rotate(random.choice([90, 270]), expand=True)

    return txt_img

def generer_tranche_banger(titre, auteur, index):
    """Crée une image de tranche de livre synthétique.

    Cette fonction génère une tranche avec un titre et un auteur, choisit un
    layout aléatoire, applique des effets visuels et enregistre l'image.
    Elle renvoie le nom de fichier et le texte concaténé pour le JSONL.
    """
    largeur = random.randint(85, 140)
    hauteur = random.randint(650, 800)
    couleur_fond = (
        random.randint(10, 120),
        random.randint(10, 120),
        random.randint(10, 120)
    )

    spine = Image.new('RGB', (largeur, hauteur), color=couleur_fond)

    # Choix du layout de texte sur la tranche
    layout = random.choice([
        'tout_vertical',
        'auteur_horizontal_titre_vertical',
        'titre_horizontal_auteur_vertical'
    ])

    # Inversion aléatoire de l'ordre du texte pour varier la disposition
    ordre = random.choice([True, False])
    elem1 = titre if ordre else auteur
    elem2 = auteur if ordre else titre
    texte_lu = f"{elem1} {elem2}"

    police_elem1 = random.choice(polices_disponibles)
    police_elem2 = random.choice(polices_disponibles)

    if layout == 'tout_vertical':
        # Deux blocs verticaux : texte centré verticalement sur la tranche
        img_e1 = text_to_image(elem1, police_elem1, max_width=(hauteur / 2) - 40, max_height=largeur - 20, vertical=True)
        img_e2 = text_to_image(elem2, police_elem2, max_width=(hauteur / 2) - 40, max_height=largeur - 20, vertical=True)
        spine.paste(img_e1, (int((largeur - img_e1.width) / 2), 20), img_e1)
        spine.paste(img_e2, (int((largeur - img_e2.width) / 2), int(hauteur / 2 + 10)), img_e2)

    elif layout == 'auteur_horizontal_titre_vertical':
        # Auteur en horizontal en haut, titre en vertical en bas
        img_e1 = text_to_image(auteur, police_elem1, max_width=largeur - 15, max_height=hauteur / 4, vertical=False)
        img_e2 = text_to_image(titre, police_elem2, max_width=hauteur - img_e1.height - 50, max_height=largeur - 20, vertical=True)
        spine.paste(img_e1, (int((largeur - img_e1.width) / 2), 15), img_e1)
        spine.paste(img_e2, (int((largeur - img_e2.width) / 2), img_e1.height + 40), img_e2)

    elif layout == 'titre_horizontal_auteur_vertical':
        # Titre en horizontal en haut, auteur vertical en dessous
        img_e1 = text_to_image(titre, police_elem1, max_width=largeur - 15, max_height=hauteur / 3, vertical=False)
        img_e2 = text_to_image(auteur, police_elem2, max_width=hauteur - img_e1.height - 50, max_height=largeur - 30, vertical=True)
        spine.paste(img_e1, (int((largeur - img_e1.width) / 2), 15), img_e1)
        spine.paste(img_e2, (int((largeur - img_e2.width) / 2), img_e1.height + 40), img_e2)

    # Application d'effets visuels hardcore pour rendre les tranches plus réalistes
    spine = ajouter_effets_hardcore(spine)

    # Flou directionnel aléatoire pour simuler une photo de mauvaise qualité
    if random.random() > 0.3:
        spine = spine.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.8)))

    spine = ajouter_bruit(spine)

    # Rotation finale aléatoire pour varier les cibles et l'orientation
    if random.random() > 0.8:
        spine = spine.rotate(random.choice([90, 270]), expand=True)

    nom_fichier = f"synth_{index:05d}.jpg"
    spine.save(os.path.join(OUTPUT_DIR, nom_fichier), quality=random.randint(40, 95))

    return nom_fichier, texte_lu

def main():
    """Point d'entrée : génère les images synthétiques et écrit les métadonnées.

    Cette fonction lit le CSV des titres et auteurs, mélange les exemples,
    génère chaque tranche de livre synthétique, puis enregistre les données
    dans un fichier JSONL prêt pour l'entraînement OCR.
    """
    print(" Lancement de la Factory V3 (NIGHTMARE MODE)...")

    # Lecture du CSV source et transformation des colonnes en chaînes
    df = pd.read_csv(CSV_PATH)
    data_propre = list(zip(df['title'].astype(str), df['authors'].astype(str)))
    random.shuffle(data_propre)

    nb_a_generer = min(NB_IMAGES_A_GENERER, len(data_propre))
    print(f" Génération de {nb_a_generer} tranches dans l'enfer visuel...")

    metadata = []

    for i in range(nb_a_generer):
        titre, auteur = data_propre[i]
        nom_fichier, texte_final = generer_tranche_banger(titre, auteur, i)

        # Enregistrement des infos image/texte pour le dataset OCR
        metadata.append({"file_name": nom_fichier, "text": texte_final})

        if (i + 1) % 1000 == 0:
            print(f" {i + 1}/{nb_a_generer} générées...")

    # Écriture du JSONL ligne par ligne avec encodage UTF-8
    with open(JSONL_PATH, 'w', encoding='utf-8') as f:
        for entry in metadata:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f" Terminé ! Les {nb_a_generer} pires cauchemars de l'OCR sont dans '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    main()