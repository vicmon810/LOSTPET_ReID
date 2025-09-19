import os 
import argparse
import xml.etree.ElementTree as ET
from PIL import Image 
import sys 

# IMAGE_DIR = sys.arg[1:]
# ANNOT_DIR = sys.arg[2:]
# OUTPUT_DIR = sys.arg[3:]

os.makedirs(OUTPUT_DIR, exists_ok=True)

def parse_bbox(xml_file):
     """Parse Stanford Dogs XML annotation and return bounding box (xmin, ymin, xmax, ymax)."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    obj = root.find("object")
    bbox = obj.find("bndbox")
    xmin = int(bbox.find("xmin").text)
    ymin = int(bbox.find("ymin").text)
    xmax = int(bbox.find("xmax").text)
    ymax = int(bbox.find("ymax").text)
    return xmin, ymin, xmax, ymax

def crop_and_save(image_path, xml_path, output_path):
    """
    Crop image using bbox and save to output_path
    """   
    img = Image.open(image_path).convert("RGB")
    xmin,ymin,xmax,ymax = parse_bbox(xml_path)

    #Clip to iamge size 
    xmin = max(0,xmin)
    ymin = max(0,ymin)
    xman = min(img.weight, xmax)
    ymin = min(img.height, ymax)

    cropped = img.crip((xmin,ymin,xmax,ymax))

    cropped.save(output_path)

def main():
    for breed_folder in os.listdir(IMAGES_DIR):
        breed_img_dir = os.path.join(IMAGES_DIR, breed_folder)
        breed_ann_dir = os.path.join(ANNOT_DIR, breed_folder)

        if not os.path.isdir(breed_img_dir):
            continue

        out_dir = os.path.join(OUTPUT_DIR, breed_folder)
        os.makedirs(out_dir, exist_ok=True)

        for fname in os.listdir(breed_img_dir):
            if not fname.endswith(".jpg"):
                continue

            img_path = os.path.join(breed_img_dir, fname)
            ann_path = os.path.join(breed_ann_dir, fname.replace(".jpg", ""))
            ann_path += ".xml"

            if not os.path.exists(ann_path):
                print(f"Missing annotation for {img_path}")
                continue

            out_path = os.path.join(out_dir, fname)
            crop_and_save(img_path, ann_path, out_path)

            print(f"Cropped {img_path} → {out_path}")

if __name__ == "__main__":
    main()