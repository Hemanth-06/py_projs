import warnings
warnings.filterwarnings("ignore")
import os
import shutil
from pathlib import Path
from PIL import Image, ImageFile, UnidentifiedImageError
import face_recognition
import cv2
import numpy as np

# ---------------- CONFIG ----------------
input_images_folder = Path(r"D:\FSAPP\sample_out\images\other_images")
converted_folder = input_images_folder.parent / "converted"
output_folder = input_images_folder.parent / "sorted_faces"
tolerance = 0.5
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
    print("🟢 Step 1: Preparing folders...")
    converted_folder.mkdir(exist_ok=True)
    output_folder.mkdir(exist_ok=True)

    print("🟢 Step 2: Converting images to 8-bit...")
    converted_files = []
    for fname in os.listdir(input_images_folder):
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")):
            src = input_images_folder / fname
            dst = converted_folder / Path(fname).stem
            
            # Skip already converted files to avoid reprocessing
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
    face_id = 0

    for img_path in converted_folder.glob("*.jpg"):
        try:
            print(f"   🔍 Processing {img_path.name}...")
            
            # Load image
            image = load_image_safe(img_path)
            
            # Detect faces
            face_locations = face_recognition.face_locations(image)
            if not face_locations:
                print(f"   ⚠️ No faces found in {img_path.name}")
                continue
                
            # Get encodings
            encodings = face_recognition.face_encodings(image, face_locations)
            
            for encoding in encodings:
                matches = face_recognition.compare_faces(known_face_encodings, encoding, tolerance)
                name = None
                
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                else:
                    name = f"person_{face_id}"
                    known_face_encodings.append(encoding)
                    known_face_names.append(name)
                    face_id += 1

                # Copy image to person's folder
                person_folder = output_folder / name
                person_folder.mkdir(exist_ok=True)
                shutil.copy(img_path, person_folder / img_path.name)
                print(f"   ✅ {img_path.name} → {name}")
                
        except Exception as e:
            print(f"   ⚠️ Error processing {img_path.name}: {str(e)}")
            continue

    print("🟢 Step 4: Completed! All faces sorted into:")
    print(f"   📂 {output_folder}")

if __name__ == "__main__":
    main()