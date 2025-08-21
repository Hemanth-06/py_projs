import warnings
warnings.filterwarnings("ignore")

import os
import shutil
import re
from pathlib import Path
from PIL import Image, ImageFile, UnidentifiedImageError
import face_recognition
import cv2
import numpy as np
import time

# ---------------- CONFIG ----------------
input_images_folder = Path(r"D:\FSAPP\sample_out\images\other_images")
converted_folder = input_images_folder.parent / "converted"
output_folder = input_images_folder.parent / "sorted_faces"
tolerance = 0.5
min_face_size = 100
# -----------------------------------------

# Enable loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

def is_valid_image(file_path):
    """Check if file is actually a valid image before processing"""
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except (IOError, SyntaxError, UnidentifiedImageError) as e:
        print(f"   ⚠️ Invalid image file: {file_path.name} - {str(e)}")
        return False

def convert_to_8bit(src_path, dst_path):
    """Convert any image to clean 8-bit RGB JPEG with multiple safety checks"""
    try:
        # First verify it's a valid image
        if not is_valid_image(src_path):
            return None
            
        with Image.open(src_path) as img:
            # Convert to RGB if not already
            if img.mode != 'RGB':
                if img.mode == 'RGBA':
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])
                    img = background
                else:
                    img = img.convert('RGB')
            
            # Save as JPEG with quality 95
            dst_path = dst_path.with_suffix(".jpg")
            img.save(dst_path, "JPEG", quality=95, subsampling=0)
            
            # Double-check the saved image
            if not is_valid_image(dst_path):
                os.remove(dst_path)  # Delete the corrupted output
                return None
                
            return dst_path
    except Exception as e:
        print(f"   ⚠️ Error converting {src_path.name}: {str(e)}")
        return None

def sort_person_labels(labels):
    """Stable numeric sort: person_1, person_2, person_10... then any non-matching labels."""
    def key(label):
        m = re.fullmatch(r"person_(\d+)", label)
        return (0, int(m.group(1))) if m else (1, label.lower())
    return sorted(labels, key=key)

def load_image_safe(path):
    """Load image with multiple fallback methods and validation"""
    try:
        # First try with PIL as it's more forgiving with formats
        with Image.open(path) as img:
            img = img.convert('RGB')
            img_array = np.array(img)
            
            # Validate the numpy array
            if img_array.dtype != np.uint8:
                img_array = img_array.astype(np.uint8)
            if len(img_array.shape) != 3 or img_array.shape[2] != 3:
                raise ValueError(f"Invalid image shape: {img_array.shape}")
                
            return img_array
    except Exception as pil_error:
        # Fallback to OpenCV if PIL fails
        try:
            img = cv2.imread(str(path))
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img_rgb
            raise ValueError("OpenCV failed to read image")
        except Exception as cv_error:
            raise ValueError(f"PIL error: {str(pil_error)} | OpenCV error: {str(cv_error)}")

def main():
    start_time = time.time()
    print("🟢 Step 1: Preparing folders...")
    converted_folder.mkdir(exist_ok=True)
    output_folder.mkdir(exist_ok=True)
    (output_folder / "others").mkdir(exist_ok=True)
    (output_folder / "group_pics").mkdir(exist_ok=True)
    previews_folder = output_folder / "previews"
    previews_folder.mkdir(exist_ok=True)

    print("🟢 Step 2: Converting images to 8-bit...")
    converted_files = []
    for fname in os.listdir(input_images_folder):
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")):
            src = input_images_folder / fname
            dst = converted_folder / Path(fname).stem
            if (converted_folder / f"{dst.stem}.jpg").exists():
                print(f"   ✓ Already converted: {fname}")
                continue
            new_path = convert_to_8bit(src, dst)
            if new_path:
                converted_files.append(new_path)
                print(f"   ✅ Converted: {fname}")
            else:
                print(f"   ❌ Failed to convert: {fname}")

    print("🟢 Step 3: Running face recognition on converted images...")
    known_face_encodings = []
    known_face_names = []
    face_id = 1  # start from person_1

    for img_path in converted_folder.glob("*.jpg"):
        try:
            print(f"   🔍 Processing {img_path.name}...")

            # Load image
            image = load_image_safe(img_path)
            # face_locations = face_recognition.face_locations(image)
            face_locations = face_recognition.face_locations(image, model="hog")
                
            valid_locations = []
            for (top, right, bottom, left) in face_locations:
                w, h = right - left, bottom - top
                aspect = w / float(h)
                # Accept only large, roughly square faces
                if w >= min_face_size and h >= min_face_size and 0.75 <= aspect <= 1.3:
                    valid_locations.append((top, right, bottom, left))

            if not valid_locations:
                shutil.copy(img_path, output_folder / "others" / img_path.name)
                print(f"   ⚠️ {img_path.name} → only tiny/background faces")
                continue

            encodings = face_recognition.face_encodings(image, valid_locations)
            face_labels = []

            for i, encoding in enumerate(encodings):
                matches = face_recognition.compare_faces(known_face_encodings, encoding, tolerance)
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                else:
                    name = f"person_{face_id}"
                    known_face_encodings.append(encoding)
                    known_face_names.append(name)
                    face_id += 1

                face_labels.append(name)

                # Save the correct face image for preview
                # Use the corresponding location for this encoding
                top, right, bottom, left = valid_locations[i]
                if not (previews_folder / f"{name}.jpg").exists():
                    face_crop = image[top:bottom, left:right]
                    cv2.imwrite(str(previews_folder / f"{name}.jpg"), cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))

            # Determine unique persons
            unique_persons = sort_person_labels(list(set(face_labels)))
            num_people = len(unique_persons)

            # CASE 1: Single person → copy to person's folder
            if num_people == 1:
                person_folder = output_folder / unique_persons[0]
                person_folder.mkdir(exist_ok=True)
                shutil.copy(img_path, person_folder / img_path.name)
                print(f"   ✅ {img_path.name} → {unique_persons[0]}")

            # CASE 2–3: Group folder
            elif 2 <= num_people <= 3:
                group_name = f"group_{num_people}_" + "_".join(unique_persons)
                group_folder = output_folder / group_name
                group_folder.mkdir(exist_ok=True)
                shutil.copy(img_path, group_folder / img_path.name)
                print(f"   👥 {img_path.name} → {group_folder.name}")

            # CASE >3: Too many faces → group_pics
            else:
                shutil.copy(img_path, output_folder / "group_pics" / img_path.name)
                print(f"   ⚠️ {img_path.name} → more than 3 faces → group_pics")

        except Exception as e:
            print(f"   ⚠️ Error processing {img_path.name}: {str(e)}")
            continue

    duration = time.time() - start_time
    print(f"🟢 Step 4: Completed in {duration:.2f} seconds! All faces sorted into:")
    print(f"   📂 {output_folder}")

    # Show sample face previews for each recognized person
    print("\n🟢 Sample faces for recognized persons saved in previews folder:")
    for sample_img in sorted(previews_folder.glob("*.jpg")):
        print(f"   - {sample_img.name}")

if __name__ == "__main__":
    main()