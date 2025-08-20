import os
import shutil
from pathlib import Path

# ---------- CONFIG ----------
LOW_KB_THRESHOLD = 50  # KB threshold for screenshots
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv'}
# ----------------------------

def get_clean_path(path: str) -> str:
    """Strip quotes and whitespace from user path input."""
    return path.strip().strip('"').strip("'")


def create_unique_output_dir(base_dir: Path, base_name: str = "sample_output") -> Path:
    """Create a unique output folder inside base_dir."""
    counter = 0
    while True:
        folder_name = base_name if counter == 0 else f"{base_name}_{counter}"
        full_path = base_dir / folder_name
        if not full_path.exists():
            full_path.mkdir(parents=True, exist_ok=True)
            return full_path
        counter += 1

def scan_for_media(input_path: Path):
    """Check if folder contains any images or videos."""
    for file in input_path.rglob("*"):
        if file.is_file():
            ext = file.suffix.lower()
            if ext in IMAGE_EXTS or ext in VIDEO_EXTS:
                return True
    return False

def create_main_folders(base_path: Path):
    """Create the main output folder structure."""
    folders = [
        base_path / "images" / "screenshots",
        base_path / "images" / "other_images",
        base_path / "videos"
    ]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
    print(f"[+] Created folder structure under: {base_path}")

def classify_and_move_files(input_path: Path, output_path: Path):
    """Classify files into screenshots, other_images, or videos."""
    for file in input_path.rglob("*"):
        if file.is_file():
            ext = file.suffix.lower()
            size_kb = file.stat().st_size / 1024

            if ext in IMAGE_EXTS:
                if size_kb <= LOW_KB_THRESHOLD:
                    shutil.copy2(file, output_path / "images" / "screenshots" / file.name)
                else:
                    shutil.copy2(file, output_path / "images" / "other_images" / file.name)

            elif ext in VIDEO_EXTS:
                shutil.copy2(file, output_path / "videos" / file.name)

def run_folder_organizer(input_dir: str, output_dir: str):
    """Main function for CLI & GUI use."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"[!] Input path does not exist: {input_dir}")
        return

    # Step 1: Check for media files first
    if not scan_for_media(input_path):
        print("[!] No images or videos present in the folder. Please check your folder.")
        return
    
    # Step 2: Create folders and classify
    create_main_folders(output_path)
    classify_and_move_files(input_path, output_path)
    print("[✓] File classification complete.")

# CLI Example
if __name__ == "__main__":
    # Get input directory
    input_dir_raw = input("Enter input folder path: ").strip()
    input_dir = get_clean_path(input_dir_raw)

    if not os.path.isdir(input_dir):
        print(f"[!] Invalid input folder: {input_dir}")
        exit(1)

    # Get output directory
    output_dir_raw = input("Enter output folder path (leave blank to auto-create): ").strip()

    if output_dir_raw:
        output_dir = get_clean_path(output_dir_raw)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    else:
        parent_dir = Path(input_dir).parent
        output_dir = create_unique_output_dir(parent_dir)

    print(f"✅ Using input folder: {input_dir}")
    print(f"✅ Using output folder: {output_dir}")

    run_folder_organizer(input_dir, output_dir)
