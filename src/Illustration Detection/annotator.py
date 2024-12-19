import os
import random
import zipfile
import pandas as pd
from pathlib import Path
from PyPDF2 import PdfReader
import pygame
from pygame.locals import *
import fitz  # PyMuPDF for GPU-accelerated rendering
import threading

# Initialize pygame
pygame.init()

def get_pages_from_metadata(file):
    """Parse METADATA.txt and return a list of pages with illustrations."""
    for line in file:
        key, value = line.strip().decode('utf-8').split(": ")
        if key == "good_pages":
            return eval(value)

def render_pages(window, surfaces, current_page, window_size):
    """Render the current page and previews of the next three pages."""
    window.fill((0, 0, 0))

    # Calculate layout for main page and previews
    main_page_area = (0, 0, window_size[0] * 3 // 4, window_size[1])
    preview_area_width = window_size[0] // 4
    preview_area_height = window_size[1] // 4

    # Render current page
    if current_page in surfaces:
        main_surface = pygame.transform.scale(surfaces[current_page], main_page_area[2:])
        window.blit(main_surface, (0, 0))

    # Render previews for the next three pages
    for i in range(1, 4):
        page_num = current_page + i
        if page_num in surfaces:
            preview_surface = pygame.transform.scale(
                surfaces[page_num], (preview_area_width, preview_area_height)
            )
            window.blit(preview_surface, (main_page_area[2], (i - 1) * preview_area_height))

    pygame.display.flip()


def pre_render_pages_with_fitz(pdf_path, result_dict):
    """Pre-render all pages of the PDF using PyMuPDF (GPU-accelerated)."""
    doc = fitz.open(pdf_path)
    for page_number in range(len(doc)):
        page = doc[page_number]
        pix = page.get_pixmap()
        image = pygame.image.frombuffer(pix.samples, (pix.width, pix.height), "RGB")
        result_dict[page_number + 1] = pygame.transform.scale(image, (800, 600))

def main():
    pdf_dir = Path("./pdfs")
    processed_zip = Path("preprocessed.zip")
    output_csv = Path("output.csv")

    # Initialize output DataFrame
    if output_csv.exists():
        results_df = pd.read_csv(output_csv)
    else:
        results_df = pd.DataFrame(columns=["PDF", "User_Illustrations", "Extractor_Illustrations", "TP", "FP", "TN", "FN"])

    TP, FP, TN, FN = results_df[["TP", "FP", "TN", "FN"]].sum(axis=0)
    print(f"Cumulative Results: TP={TP}, FP={FP}, TN={TN}, FN={FN}")


    # Set up pygame window
    window_size = (800, 600)
    window = pygame.display.set_mode(window_size)
    pygame.display.set_caption("PDF Page Review")

    running = True

    while running:
        # Select a random PDF
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            print("No PDF files found in the directory.")
            break

        random_pdf = random.choice(pdf_files)
        pdf_name = random_pdf.stem

        # Extract metadata
        with zipfile.ZipFile(processed_zip, 'r') as z:
            metadata_path = f"{pdf_name}/METADATA.txt"
            if metadata_path in z.namelist():
                with z.open(metadata_path) as f:
                    metadata_pages = get_pages_from_metadata(f)
            else:
                print(f"No METADATA.txt found for {pdf_name}.")
                continue

        user_illustrations = []

        # Pre-render all pages using a separate thread
        pre_rendered_images = {}
        render_thread = threading.Thread(
            target=pre_render_pages_with_fitz, args=(random_pdf, pre_rendered_images)
        )
        render_thread.start()

        num_pages = len(PdfReader(random_pdf).pages)

        for page_number in range(1, num_pages+1):
            page_id = page_number-1
            # Wait for the page to be rendered
            while page_number not in pre_rendered_images:
                pygame.time.wait(10)

            # Display the page
            render_pages(window, pre_rendered_images, page_number, window_size)

            # Handle user input
            while True:
                for event in pygame.event.get():
                    if event.type == QUIT:
                        running = False
                        break
                    elif event.type == KEYDOWN:
                        if event.key == K_o:  # Mark as illustration
                            user_illustrations.append(page_id)
                            print(f"Page {page_id}: Marked as illustration.")
                            break
                        elif event.key == K_n:  # Mark as no illustration
                            print(f"Page {page_id}: Marked as no illustration.")
                            break
                        elif event.key == K_SPACE: # Mark previous page as illustration
                            if page_id > 1:
                                user_illustrations.append(page_id - 1)
                                print(f"Page {page_id - 1}: Marked as illustration.")
                        elif event.key == K_f:  # Finish
                            print("Exiting...")
                            results_df.to_csv(output_csv, index=False)
                            running = False
                            break
                else:
                    continue
                break

            if not running:
                break

        if not running:
            break

        # Compare user input with metadata
        user_set = set(user_illustrations)
        metadata_set = set(metadata_pages)

        tp = len(user_set & metadata_set)
        fp = len(user_set - metadata_set)
        fn = len(metadata_set - user_set)
        tn = num_pages - (tp + fp + fn)

        TP += tp
        FP += fp
        FN += fn
        TN += tn

        # Save results for the current PDF
        results_df = pd.concat([
            results_df,
            pd.DataFrame({
                "PDF": [pdf_name],
                "User_Illustrations": [list(user_set)],
                "Extractor_Illustrations": [list(metadata_set)],
                "TP": [tp],
                "FP": [fp],
                "TN": [tn],
                "FN": [fn]
            })
        ], ignore_index=True)

        results_df.to_csv(output_csv, index=False)

        # Display summary
        print(f"Processed {len(results_df)} PDFs.")
        print(f"Results for {pdf_name}: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        print(f"Cumulative Results: TP={TP}, FP={FP}, TN={TN}, FN={FN}")

    pygame.quit()

if __name__ == "__main__":
    main()
