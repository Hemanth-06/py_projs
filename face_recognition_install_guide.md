# Face Recognition Environment Setup Guide (Windows + Python 3.10)

This guide explains how to set up a Python virtual environment and install all dependencies for face recognition using precompiled packages to avoid compilation errors on Windows.

---

## Step 1: Create a virtual environment

1. Open Command Prompt in your project directory (e.g., D:\FSAPP).
2. Create a virtual environment (named `venv`):
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   ```
   venv\Scripts\activate
   ```
   You should see `(venv)` at the beginning of your command line.

---

## Step 2: Upgrade pip

Upgrade pip to the latest version:
```
python -m pip install --upgrade pip
```

---

## Step 3: Download precompiled dlib wheel

1. Download the precompiled wheel for Python 3.10 (64-bit) from:
```
https://huggingface.co/hanamizuki-ai/pypi-wheels/blob/a056889004afa48f7b178e35846788eb72002073/dlib/dlib-19.24.1-cp310-cp310-win_amd64.whl
```
2. Save it somewhere accessible, e.g., `C:\Users\<YourUser>\Downloads\dlib-19.24.1-cp310-cp310-win_amd64.whl`.

---

## Step 4: Install dlib

Install the downloaded wheel:
```
pip install "C:\Users\<YourUser>\Downloads\dlib-19.24.1-cp310-cp310-win_amd64.whl"
```
> ⚠️ Make sure to use quotes around the path if it contains spaces.

---

## Step 5: Install face_recognition and other dependencies

Once `dlib` is installed, install `face_recognition` and its dependencies:
```
pip install face_recognition==1.3.0 numpy==1.23.5 Pillow==9.3.0 opencv-python==4.6.0.66
```
> ⚠️ Use the exact versions for compatibility.

---

## Step 6: Verify installation

Run a quick test to make sure everything is installed correctly:
```
python -c "import face_recognition; import dlib; import cv2; import numpy; import PIL; print('All libraries loaded successfully')"
```
You should see:
```
All libraries loaded successfully
```

---

## Step 7: Notes

- Always activate the virtual environment before working on this project:
  ```
  venv\Scripts\activate
  ```
- If upgrading Python or moving to a new machine, use the same wheel for `dlib` to avoid compilation issues.
- This guide avoids building `dlib` from source, which requires Visual Studio C++ on Windows.

