"""
Changes from v1:
- Asked user for area threshold, 0.25 seems to be a good value (more sensitive, detects illustrations with less black and merged with text)
- Exclude first and last pages (covers)
- No longer saves the detected illustration, only the whole page
- Improved metadata

outliers:
    boysrevoltstoryo00otis
amusinghistoryof00littiala
blindmansfriendo00normiala
bobbseytwinsonbl00hope
browniesthroughu00coxp2
    childreninwood00harviala
"""

from PIL import Image
import numpy as np
import fitz  # PyMuPDF
import os
import io
import sys
import time

def get_image_data(page, pdf_document):
    # Extract image information
    images = page.get_images(full=True)
    xref = images[0][0]
    base_image = pdf_document.extract_image(xref)

    # Extract image data
    image_bytes = base_image["image"]
    image = Image.open(io.BytesIO(image_bytes))
    image_np = np.array(image)
    return image_np


def white_pages_filter(image_data, area_threshold):
    # Make a grayscale image
    image_np = np.array(image_data)
    image_gray = np.mean(image_np, axis=-1)

    # Check if the image is an illustration, if not: remove it
    # FIXME: only works for black illustrations
    is_illustration = image_gray.min() < 100 and image_gray[image_gray < 100].sum() > area_threshold * image_gray.size
    return is_illustration


def save_as_png(image_data, output_path):
    image = Image.fromarray(image_data)
    image.save(output_path, format="PNG")


def save_metadata(metadata, output_folder):
    with open(os.path.join(output_folder, "METADATA.txt"), "w") as file:
        # as metadata is a dictionary, we can iterate over its keys and values
        for key, value in metadata.items():
            file.write(f"{key}: {value}\n")


def extract_metadata(input_folder):
    metadata = {}
    with open(os.path.join(input_folder, "METADATA.txt"), "r") as file:
        for line in file:
            key, value = line.strip().split(": ")
            metadata[key] = value
    return metadata


def extract_illustration_from_pdf(pdf_path, output_folder, area_threshold):
    """
    Extracts the first image from each page of the PDF file and saves it as PNG files in the output folder.
    The images are saved with the following naming convention: pageX_illustration.png

    :param pdf_path: path to the PDF file
    :param output_folder: path to the folder where the images will be saved

    :return: metadata dictionary with the number of pages and the list of pages with illustrations
    """
    pages = []
    # create output folders if they don't exist
    os.makedirs(output_folder, exist_ok=True)

    pdf_document = fitz.open(pdf_path)
    for page_num in range(len(pdf_document)):
        # excluding first and last page
        if page_num == 0 or page_num == len(pdf_document)-1:
            continue
         
        page = pdf_document[page_num]
        image_data = get_image_data(page, pdf_document)
        
        if white_pages_filter(image_data, area_threshold):
            # Save the whole pdf page as a PNG file
            output_path = os.path.join(output_folder, f"page{page_num}.png")
            page.get_pixmap().save(output_path, output="png")

            pages.append(page_num)
        
        # Verbose, every 10%
        if len(pdf_document) > 10 and page_num % (len(pdf_document)//10) == 0:
            print(f"Extracted {page_num+1}/{len(pdf_document)} pages")
        
    pdf_document.close()
    metadata = {
        "number_of_pages": len(pages),
        "good_pages": pages
    }
    save_metadata(metadata, output_folder)
    print(f"Illustrations extracted to '{output_folder}'")
    return metadata


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:\tpython extractor.py <pdf_path> <output_folder>")
        print("or, for several pdfs:\tpython extractor.py <input_folder_path> <output_folder> -all")
        sys.exit(1)
    
    # Ask user for the area threshold, default to 0.5 if no input
    area_threshold = float(input("Enter the area threshold (default 0.25): ") or 0.25)

    pages = []
    input_folder_path = sys.argv[1]
    output_folder_path = sys.argv[2]
    process_all = len(sys.argv) == 4 and sys.argv[3] == "-all"

    if process_all:
        time_start = time.time()
        for pdf_file in os.listdir(input_folder_path):
            print(f"Extracting illustrations from {pdf_file}...")
            output_folder = os.path.join(output_folder_path, pdf_file.split(".")[0])
            input_path = os.path.join(input_folder_path, pdf_file)
            metadata = extract_illustration_from_pdf(input_path, output_folder, area_threshold)
            pages.append((pdf_file, metadata["number_of_pages"]))
            print()
        
        # save metadata for all pdfs
        metadata = {
            "area_threshold": area_threshold,
            "pdfs": pages,
        }
        save_metadata(metadata, output_folder_path)
        print(f"Time to finish: {time.strftime('%M:%S', time.gmtime(time.time()-time_start))}")
    else:
        extract_illustration_from_pdf(input_folder_path, output_folder_path, area_threshold)