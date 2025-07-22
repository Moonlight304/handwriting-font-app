# filepath: handwriting-font-app/handwriting-font-app/src/main.py
# Handwriting Font Generation Application

import os
from pathlib import Path
from char_segmentation import extract_characters_from_images
from labeling import create_labels
from dataset import HandwritingDataset
from model import CVAE
from train import train_model
from generate import generate_characters
from svg_convert import convert_to_svg
from fontforge_script import create_font

DATA_DIR = Path('data/raw_pages')
SEGMENTED_DIR = Path('data/segmented_characters')
GENERATED_DIR = Path('data/generated_characters')
FONT_DIR = Path('data/fonts')
LABELS_CSV = 'labels.csv'

def main():
    # Step 1: Character Segmentation
    print("[INFO] Starting character segmentation")
    extract_characters_from_images(DATA_DIR, SEGMENTED_DIR)

    # Step 2: Labeling
    print("[INFO] Creating labels for segmented characters")
    create_labels(SEGMENTED_DIR, LABELS_CSV)

    # Step 3: Load Dataset
    print("[INFO] Loading dataset")
    dataset = HandwritingDataset(LABELS_CSV, str(SEGMENTED_DIR))
    
    # Step 4: Train the Model
    print("[INFO] Training the model")
    model = CVAE()
    train_model(model, dataset)

    # Step 5: Generate Characters
    print("[INFO] Generating new characters")
    generate_characters(model, GENERATED_DIR)

    # Step 6: Convert to SVG
    print("[INFO] Converting generated characters to SVG")
    convert_to_svg(GENERATED_DIR)

    # Step 7: Create Font
    print("[INFO] Creating font from SVG files")
    create_font(FONT_DIR)

    print("[INFO] Font generation complete")

if __name__ == "__main__":
    main()
    