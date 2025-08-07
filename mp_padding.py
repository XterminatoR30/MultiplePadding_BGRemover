import os
import zipfile
import shutil
import time
import datetime
from PIL import Image, ImageDraw, ImageCms
from io import BytesIO
import io
from rembg import remove
import gradio as gr
import pillow_avif
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModelForImageSegmentation, pipeline
import numpy as np
import json
import requests
from dotenv import load_dotenv
import torch
from torchvision import transforms
from functools import lru_cache
import cv2
import threading
from collections import Counter
import easyocr
import re
import base64
import tempfile
import pandas as pd
from openai import OpenAI

# Add rawpy for RAW file support
try:
    import rawpy
    RAW_SUPPORT = True
    print("RAW file support enabled (rawpy library found)")
    print(f"rawpy version: {rawpy.__version__}")
except ImportError:
    RAW_SUPPORT = False
    print("RAW file support disabled (rawpy library not found). Install with: pip install rawpy")
except Exception as e:
    RAW_SUPPORT = False
    print(f"RAW file support disabled due to error: {e}")

stop_event = threading.Event()

# Load environment
load_dotenv()
PHOTOROOM_API_KEY = ""

_birefnet_model = None
_birefnet_transform = None

_birefnet_hr_model = None
_birefnet_hr_transform = None

# Global EasyOCR reader instance for better performance
_easyocr_reader = None

def get_easyocr_reader():
    """
    Get or initialize EasyOCR reader instance (singleton pattern for performance)
    """
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            print("Initializing EasyOCR reader...")
            _easyocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
            print("EasyOCR reader initialized successfully")
        except Exception as e:
            print(f"Error initializing EasyOCR: {e}")
            _easyocr_reader = None
    return _easyocr_reader

# ------------------ Qwen 2.5VL Inference Functions & Model Loading ------------------

def encode_image(image_path):
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        return base64.b64encode(image_bytes).decode('utf-8')
    except Exception as e:
        print(f"Error in encode_image: {str(e)}")
        raise

# Initialize OpenAI client for Qwen API
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY", ''),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)

def inference_with_api(image_path, prompt, sys_prompt="You are a helpful visual analysis assistant that specializes in determining how products and people should be positioned on a canvas for optimal visual presentation.", model_id="qwen2.5-vl-72b-instruct", min_pixels=512*28*28, max_pixels=2048*28*28):
    try:
        base64_image = encode_image(image_path)
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": sys_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels,
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        retries = 3
        for attempt in range(retries):
            try:
                completion = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    timeout=30  # Increase timeout to 30 seconds
                )
                return completion.choices[0].message.content
            except Exception as inner_e:
                # If the error message contains "Connection error", retry
                if "Connection error" in str(inner_e):
                    print(f"Connection error on attempt {attempt+1}: {inner_e}. Retrying in 2 seconds...")
                    time.sleep(2)
                else:
                    raise
        raise Exception("Failed to complete API call after multiple retries due to connection errors.")
    except Exception as e:
        print(f"Error in inference_with_api: {str(e)}")
        raise

def classify_image(image_path, unique_items):
    try:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224), Image.LANCZOS)
        
        print(f"Classifying image: {image_path} (resized to {image.size})")
        prompt = (
            f"Classify this image into one of these categories: {', '.join(unique_items)}. "
            f"Be sensitive to sizes of an object, e.g. 'small' or 'medium' or 'large', especially for bags. "
            f"If a hand is detected, only pick classifications that mention 'hand', however if it's a human, only pick classifications which mentioned 'human'. "
            f"Return only the classification word, nothing else."
        )
        
        # Save resized image to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            image.save(temp_file.name, format='PNG')
            temp_image_path = temp_file.name
        
        # Get raw classification from API with retry logic
        classification_result = inference_with_api(temp_image_path, prompt)
        print(f"Raw API response for {image_path}: '{classification_result}'")
        
        # Clean up temporary file
        os.unlink(temp_image_path)
        
        # Parse and match the classification result
        classification_result = classification_result.strip().lower()
        for item in unique_items:
            if item.lower() in classification_result:
                print(f"Matched classification for {image_path}: '{item}'")
                return item
        
        print(f"No matching classification found in response: '{classification_result}'. Expected one of: {unique_items}")
        return None
    
    except Exception as e:
        print(f"Error during classification for {image_path}: {str(e)}")
        return None

def debug_color_values(img, x, y, title="Color values"):
    """Debug function - disabled in production"""
    pass

def convert_rgb_to_cmyk_with_profile(rgb_img, original_cmyk_profile=None):
    """
    Convert RGB image back to CMYK using original profile if available
    
    Args:
        rgb_img: PIL Image in RGB mode
        original_cmyk_profile: Original CMYK ICC profile bytes
        
    Returns:
        PIL Image in CMYK mode
    """
    try:
        if original_cmyk_profile:
            # Create profiles
            srgb_profile = ImageCms.createProfile("sRGB")
            cmyk_profile_io = io.BytesIO(original_cmyk_profile)
            cmyk_profile = ImageCms.ImageCmsProfile(cmyk_profile_io)
            
            # Create transform from RGB to CMYK
            transform = ImageCms.buildTransformFromOpenProfiles(
                srgb_profile,
                cmyk_profile,
                "RGB", "CMYK"
            )
            
            # Apply transform
            cmyk_image = ImageCms.applyTransform(rgb_img, transform)
            
            # Preserve the original profile
            cmyk_image.info['icc_profile'] = original_cmyk_profile
            
            return cmyk_image
            
        else:
            # Fallback to standard conversion without profile
            return rgb_img.convert("CMYK")
            
    except Exception:
        # Fallback to standard conversion
        return rgb_img.convert("CMYK")

def ensure_color_fidelity(original_img, processed_img, sample_points=0):
    """
    Ensure the processed image maintains the original colors where alpha > 0
    
    Args:
        original_img: The original image (can be any mode)
        processed_img: The processed image with alpha channel
        sample_points: Number of sample points for debugging color values (disabled in production)
        
    Returns:
        A PIL Image with original colors and processed alpha
    """
    # Early return if inputs are invalid
    if not isinstance(original_img, Image.Image) or not isinstance(processed_img, Image.Image):
        return processed_img
    
    # Get original image in RGB for consistent processing
    try:
        original_for_processing, _ = convert_image_to_rgb(original_img)
    except Exception:
        original_for_processing = original_img
    
    # Ensure processed image is RGBA
    if processed_img.mode != "RGBA":
        processed_img = processed_img.convert("RGBA")
    
    # Handle size mismatch by resizing original to match processed
    if original_for_processing.size != processed_img.size:
        original_for_processing = original_for_processing.resize(processed_img.size, Image.LANCZOS)
    
    # Convert to numpy arrays for efficient processing
    try:
        # Convert original to RGBA to match processed format
        if original_for_processing.mode != "RGBA":
            orig_rgba = original_for_processing.convert("RGBA")
        else:
            orig_rgba = original_for_processing
            
        orig_array = np.array(orig_rgba)
        proc_array = np.array(processed_img)
        
        # Create a mask for non-transparent pixels (alpha > 128)
        alpha_mask = proc_array[:,:,3] > 128
        
        # Only copy RGB values where alpha is significant to preserve transparency
        if alpha_mask.any():  # Check if there are any non-transparent pixels
            # Copy only the RGB channels from original to processed where alpha > 128
            proc_array[alpha_mask, 0:3] = orig_array[alpha_mask, 0:3]
            
            # Create the result image
            result = Image.fromarray(proc_array)
            
            return result
        else:
            return processed_img
            
    except Exception:
        # Return the processed image unchanged if there's an error
        return processed_img

def convert_image_to_rgb(img):
    """
    Consistently convert images of any mode to RGB while preserving color accuracy
    
    Args:
        img: PIL Image object, potentially in any supported mode
        
    Returns:
        tuple: (converted_image, icc_profile)
    """
    if not isinstance(img, Image.Image):
        raise ValueError("Input must be a PIL Image")
        
    # Store original profile
    icc_profile = img.info.get('icc_profile')
    
    # Handle different modes
    if img.mode == "RGB":
        # Already RGB, no conversion needed
        return img, icc_profile
    elif img.mode == "RGBA":
        # Keep alpha information
        return img, icc_profile
    elif img.mode == "CMYK":
        # Use specialized CMYK conversion with profile if available
        if icc_profile:
            try:
                # Create an RGB profile (sRGB is standard)
                srgb_profile = ImageCms.createProfile("sRGB")
                
                # Create CMYK profile from embedded profile
                cmyk_profile = io.BytesIO(icc_profile)
                
                # Create color transform
                transform = ImageCms.buildTransformFromOpenProfiles(
                    ImageCms.ImageCmsProfile(cmyk_profile),
                    ImageCms.ImageCmsProfile(srgb_profile),
                    "CMYK", "RGB"
                )
                
                # Apply transform
                rgb_image = ImageCms.applyTransform(img, transform)
                
                # Use sRGB profile for output
                new_profile = srgb_profile.tobytes() if hasattr(srgb_profile, 'tobytes') else None
                
                return rgb_image, new_profile
            except Exception:
                pass
        
        # Fallback to standard conversion if no profile or error occurred
        return img.convert("RGB"), None
    elif img.mode == "L":
        # Grayscale to RGB
        return img.convert("RGB"), icc_profile
    elif img.mode == "LA":
        # Grayscale with alpha to RGBA
        return img.convert("RGBA"), icc_profile
    elif img.mode == "P":
        # Palettized to RGB
        return img.convert("RGB"), icc_profile
    elif img.mode == "PA":
        # Palettized with alpha to RGBA
        return img.convert("RGBA"), icc_profile
    elif img.mode == "1":
        # Bilevel to RGB
        return img.convert("RGB"), icc_profile
    elif img.mode == "HSV":
        # HSV to RGB
        return img.convert("RGB"), icc_profile
    elif img.mode == "YCbCr":
        # YCbCr to RGB
        return img.convert("RGB"), icc_profile
    elif img.mode == "LAB":
        # LAB to RGB
        return img.convert("RGB"), icc_profile
    else:
        # Unknown mode, try generic conversion
        try:
            return img.convert("RGB"), icc_profile
        except Exception:
            # Last resort, create a new RGB image
            rgb_img = Image.new("RGB", img.size, (255, 255, 255))
            try:
                rgb_img.paste(img)
            except:
                pass
            return rgb_img, None

# Replace the existing convert_cmyk_to_rgb_with_profile function with the more general one
convert_cmyk_to_rgb_with_profile = convert_image_to_rgb

@lru_cache(maxsize=1)
def get_birefnet_model():
    global _birefnet_model, _birefnet_transform
    if _birefnet_model is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _birefnet_model = AutoModelForImageSegmentation.from_pretrained(
            'ZhengPeng7/BiRefNet',
            trust_remote_code=True,
            torch_dtype=torch.float32
        ).to(device)
        _birefnet_model.eval()
        _birefnet_transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return _birefnet_model, _birefnet_transform

def get_birefnet_hr_model():
    global _birefnet_hr_model, _birefnet_hr_transform
    if _birefnet_hr_model is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _birefnet_hr_model = AutoModelForImageSegmentation.from_pretrained(
            'ZhengPeng7/BiRefNet_HR',
            trust_remote_code=True,
            torch_dtype=torch.float32
        ).to(device)
        _birefnet_hr_model.eval()
        _birefnet_hr_transform = transforms.Compose([
            transforms.Resize((2048, 2048)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return _birefnet_hr_model, _birefnet_hr_transform

def remove_background_rembg(input_path):
    # Get original image with color profile - load image fully in memory
    original_img = Image.open(input_path)
    original_size = original_img.size
    icc_profile = original_img.info.get('icc_profile')
    
    # Handle images with our consolidated function for any mode
    original, icc_profile = convert_image_to_rgb(original_img)
    
    # Debug original color at center point
    width, height = original.size
    center_x, center_y = width//2, height//2
    debug_color_values(original, center_x, center_y, "Original color")
    
    # Process with rembg
    with open(input_path, 'rb') as f:
        input_image = f.read()
    out_data = remove(input_image)
    
    # Get the result from rembg
    rembg_result = Image.open(io.BytesIO(out_data))
    
    # Check if rembg result is distorted (has different aspect ratio)
    if rembg_result.size != original_size:
        # Resize rembg result back to original size before extracting alpha
        rembg_result = rembg_result.resize(original_size, Image.LANCZOS)
    
    # Get only the alpha channel from rembg result
    if rembg_result.mode == 'RGBA':
        mask = rembg_result.split()[3]
    else:
        # If not RGBA, convert to RGBA first
        mask = rembg_result.convert('RGBA').split()[3]
    
    # Create a new RGBA image with original RGB channels
    result = original.copy()
    result.putalpha(mask)
    
    # Ensure colors match the original exactly where alpha > 0
    result = ensure_color_fidelity(original_img, result)
    
    # Debug processed color at same center point
    safe_x = min(center_x, result.width - 1)
    safe_y = min(center_y, result.height - 1)
    debug_color_values(result, safe_x, safe_y, "Processed color")
    
    return result

def remove_background_bria(input_path):
    # Get original image with color profile - load fully in memory
    original_img = Image.open(input_path)
    original_size = original_img.size
    icc_profile = original_img.info.get('icc_profile')
    
    # Handle CMYK images with our consolidated function
    if original_img.mode == "CMYK":
        original, icc_profile = convert_cmyk_to_rgb_with_profile(original_img)
    else:
        original = original_img.convert("RGB")
    
    # Debug original color at center point
    width, height = original.size
    center_x, center_y = width//2, height//2
    debug_color_values(original, center_x, center_y, "Original color (bria)")
    
    # ADJUSTED: Dynamically set the device (-1 for CPU) to avoid errors if CUDA isn't available.
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True, device=device)
    result = pipe(input_path)
    if isinstance(result, list) and len(result) > 0 and "mask" in result[0]:
        mask = result[0]["mask"]
    else:
        mask = result
    
    # Check for aspect ratio distortion
    if hasattr(mask, 'size') and mask.size != original_size:
        # Resize mask back to original size
        if mask.mode == "RGBA":
            # Extract alpha, resize, then apply back
            alpha_channel = mask.split()[3]
            alpha_channel = alpha_channel.resize(original_size, Image.LANCZOS)
            mask = alpha_channel
        else:
            # If not already an alpha channel, resize as is
            mask = mask.resize(original_size, Image.LANCZOS)
    
    # If mask is not single channel, extract alpha
    if hasattr(mask, 'mode'):
        if mask.mode == "RGBA":
            mask = mask.split()[3]
        elif mask.mode != "L":
            mask = mask.convert("L")
    
    # Apply mask to original image
    result = original.copy()
    result.putalpha(mask)
    
    # Ensure colors match the original
    result = ensure_color_fidelity(original_img, result)
    
    # Debug processed color
    safe_x = min(center_x, result.width - 1)
    safe_y = min(center_y, result.height - 1)
    debug_color_values(result, safe_x, safe_y, "Processed color (bria)")
    
    return result

def remove_background_birefnet(input_path):
    try:
        # Get original image with color profile - load fully in memory
        original_img = Image.open(input_path)
        icc_profile = original_img.info.get('icc_profile')
        
        # Handle CMYK images with our consolidated function
        if original_img.mode == "CMYK":
            original, icc_profile = convert_cmyk_to_rgb_with_profile(original_img)
        else:
            original = original_img.convert("RGB")
            
        # Debug original color
        width, height = original.size
        center_x, center_y = width//2, height//2
        debug_color_values(original, center_x, center_y, "Original color (birefnet)")
        
        # Get or initialize model and transform
        model, transform_image = get_birefnet_model()
        device = next(model.parameters()).device

        # Apply transforms
        input_tensor = transform_image(original).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            try:
                preds = model(input_tensor)[-1].sigmoid()
                pred_mask = preds[0].squeeze().cpu()
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    input_tensor = input_tensor.cpu()
                    model = model.cpu()
                    preds = model(input_tensor)[-1].sigmoid()
                    pred_mask = preds[0].squeeze()
                    model = model.to(device)
                else:
                    raise e

        # Convert mask to PIL and resize
        mask_pil = transforms.ToPILImage()(pred_mask)
        mask_resized = mask_pil.resize(original.size, Image.LANCZOS)

        # Create result image
        result = original.copy()
        result.putalpha(mask_resized)

        # Convert to numpy array for advanced processing
        result_array = np.array(result)
        alpha = result_array[:, :, 3]

        # Enhanced noise reduction pipeline
        # 1. Initial very aggressive threshold
        _, alpha = cv2.threshold(alpha, 248, 255, cv2.THRESH_BINARY)

        # 2. Create multiple kernel sizes for multi-scale processing
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_medium = np.ones((5, 5), np.uint8)
        kernel_large = np.ones((9, 9), np.uint8)  # Increased kernel size

        # 3. Apply Gaussian blur to smooth the mask
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)

        # 4. Apply morphological operations to clean up the mask
        # Remove small noise with opening operation
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel_small, iterations=3)
        
        # Fill small holes with closing operation
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel_medium, iterations=3)
        
        # Smooth edges with larger kernel
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel_large, iterations=2)

        # 5. Apply edge-preserving filters
        # Bilateral filter for edge-aware smoothing
        alpha = cv2.bilateralFilter(alpha, 9, 100, 100)
        
        # Additional median blur to remove remaining noise
        alpha = cv2.medianBlur(alpha, 5)

        # 6. Final aggressive threshold
        _, alpha = cv2.threshold(alpha, 248, 255, cv2.THRESH_BINARY)

        # 7. Final cleanup pass
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel_small, iterations=2)
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel_small, iterations=2)

        # 8. Edge refinement
        edges = cv2.Canny(alpha, 100, 200)
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel_medium, iterations=1)
        alpha = cv2.subtract(alpha, edges)

        # Update alpha channel
        result_array[:, :, 3] = alpha

        # Convert back to PIL Image
        result = Image.fromarray(result_array)
        
        # Ensure colors match the original
        result = ensure_color_fidelity(original_img, result)
        
        # Debug processed color
        safe_x = min(center_x, result.width - 1)
        safe_y = min(center_y, result.height - 1)
        debug_color_values(result, safe_x, safe_y, "Processed color (birefnet)")

        # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    except Exception as e:
        print(f"Error in remove_background_birefnet: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
        
def remove_background_birefnet_2(input_path):
    try:
        # Get original image with color profile - load fully in memory
        original_img = Image.open(input_path)
        icc_profile = original_img.info.get('icc_profile')
        
        # Handle CMYK images with our consolidated function
        if original_img.mode == "CMYK":
            original, icc_profile = convert_cmyk_to_rgb_with_profile(original_img)
        else:
            original = original_img.convert("RGB")
            
        # Debug original color
        width, height = original.size
        center_x, center_y = width//2, height//2
        debug_color_values(original, center_x, center_y, "Original color (birefnet_2)")
        
        # Get or initialize model and transform
        model, transform_image = get_birefnet_model()
        device = next(model.parameters()).device

        # Apply transforms
        input_tensor = transform_image(original).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            try:
                preds = model(input_tensor)[-1].sigmoid()
                pred_mask = preds[0].squeeze().cpu()
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # Try again with smaller image
                    input_tensor = input_tensor.cpu()
                    model = model.cpu()
                    preds = model(input_tensor)[-1].sigmoid()
                    pred_mask = preds[0].squeeze()
                    model = model.to(device)
                else:
                    raise e

        # Convert mask to PIL and resize
        mask_pil = transforms.ToPILImage()(pred_mask)
        mask_resized = mask_pil.resize(original.size, Image.LANCZOS)

        # Create result image
        result = original.copy()
        result.putalpha(mask_resized)
        
        # Ensure colors match the original
        result = ensure_color_fidelity(original_img, result)
        
        # Debug processed color
        safe_x = min(center_x, result.width - 1)
        safe_y = min(center_y, result.height - 1)
        debug_color_values(result, safe_x, safe_y, "Processed color (birefnet_2)")

        # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result
    except Exception as e:
        print(f"Error in remove_background_birefnet_2: {str(e)}")
        return None

def remove_background_birefnet_hr(input_path):
    try:
        # Get original image with color profile - load fully in memory
        original_img = Image.open(input_path)
        icc_profile = original_img.info.get('icc_profile')
        
        # Handle CMYK images with our consolidated function
        if original_img.mode == "CMYK":
            original, icc_profile = convert_cmyk_to_rgb_with_profile(original_img)
        else:
            original = original_img.convert("RGB")
            
        # Debug original color
        width, height = original.size
        center_x, center_y = width//2, height//2
        debug_color_values(original, center_x, center_y, "Original color (birefnet_hr)")
        
        model, transform_img = get_birefnet_hr_model()
        device = next(model.parameters()).device
        t_in = transform_img(original).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = model(t_in)[-1].sigmoid()
            mask = preds[0].squeeze().cpu()
        mask_pil = transforms.ToPILImage()(mask).resize(original.size, Image.LANCZOS)
        
        # Create result image with original colors
        result = original.copy()
        result.putalpha(mask_pil)
        
        # Ensure colors match the original
        result = ensure_color_fidelity(original_img, result)
        
        # Debug processed color
        safe_x = min(center_x, result.width - 1)
        safe_y = min(center_y, result.height - 1)
        debug_color_values(result, safe_x, safe_y, "Processed color (birefnet_hr)")
        
        return result
    except Exception:
        return None

def remove_background_photoroom(input_path):
    try:
        # Get original image with color profile - load fully in memory
        original_img = Image.open(input_path)
        icc_profile = original_img.info.get('icc_profile')
        
        # Handle CMYK images with our consolidated function
        if original_img.mode == "CMYK":
            original, icc_profile = convert_cmyk_to_rgb_with_profile(original_img)
        else:
            original = original_img.convert("RGB")
            
        # Debug original color
        width, height = original.size
        center_x, center_y = width//2, height//2
        debug_color_values(original, center_x, center_y, "Original color (photoroom)")
        
        # Handle AVIF format if needed
        input_file = input_path
        if input_path.lower().endswith('.avif'):
            input_file = convert_avif(input_path, input_path.rsplit('.', 1)[0] + '.png', 'PNG')
            
        if not PHOTOROOM_API_KEY:
            raise ValueError("Photoroom API key missing.")
            
        url = "https://sdk.photoroom.com/v1/segment"
        headers = {"Accept": "image/png, application/json", "x-api-key": PHOTOROOM_API_KEY}
        
        with open(input_file, "rb") as f:
            resp = requests.post(url, headers=headers, files={"image_file": f})
            
        if resp.status_code != 200:
            raise Exception(f"PhotoRoom API error: {resp.status_code} - {resp.text}")
            
        # Get the photoroom result
        photoroom_result = Image.open(BytesIO(resp.content))
        
        # Extract just the alpha channel
        if photoroom_result.mode == "RGBA":
            mask = photoroom_result.split()[3]
        else:
            # If it's not RGBA, convert to grayscale as mask
            mask = photoroom_result.convert("L")
        
        # Apply mask to original image to preserve colors
        result = original.copy()
        result.putalpha(mask)
        
        # Ensure colors match the original
        result = ensure_color_fidelity(original_img, result)
        
        # Debug processed color
        safe_x = min(center_x, result.width - 1)
        safe_y = min(center_y, result.height - 1)
        debug_color_values(result, safe_x, safe_y, "Processed color (photoroom)")
        
        return result
    except Exception:
        return None

def remove_background_none(input_path):
    """
    Simply convert the image to RGBA without removing background.
    Special handling for all image modes to preserve original colors.
    """
    # Get original image with color profile
    original_img = Image.open(input_path)
    icc_profile = original_img.info.get('icc_profile')
    
    # Use our converter that properly handles all modes
    rgb_image, new_profile = convert_image_to_rgb(original_img)
    
    # If we have an alpha channel, keep it, otherwise add one
    if rgb_image.mode == 'RGBA':
        result = rgb_image
    else:
        result = rgb_image.convert("RGBA")
    
    # Set profile
    if new_profile:
        result.info['icc_profile'] = new_profile
    elif icc_profile:
        result.info['icc_profile'] = icc_profile
        
    return result

def get_dominant_color(image):
    tmp = image.convert("RGBA")
    tmp.thumbnail((100, 100))
    ccount = Counter(tmp.getdata())
    return ccount.most_common(1)[0][0]

def convert_avif(input_path, output_path, output_format='PNG'):
    with Image.open(input_path) as img:
        if output_format == 'JPG':
            img.convert("RGB").save(output_path, "JPEG")  # Convert to JPG (RGB mode)
        else:
            img.save(output_path, "PNG")  # Convert to PNG

    return output_path

def convert_arw_to_pil(arw_path):
    """
    Convert Sony RAW (.arw) file to PIL Image
    
    Args:
        arw_path: Path to the .arw file
        
    Returns:
        PIL Image object or None if conversion fails
    """
    if not RAW_SUPPORT:
        print(f"Cannot process {arw_path}: RAW support not available. Install rawpy with: pip install rawpy")
        return None
        
    try:
        print(f"Converting RAW file: {os.path.basename(arw_path)}")
        
        # Read RAW file with rawpy
        with rawpy.imread(arw_path) as raw:
            # Process the RAW data using compatible parameters
            try:
                # Try with newer rawpy parameters first
                rgb_array = raw.postprocess(
                    use_camera_wb=True,      # Use camera white balance
                    half_size=False,         # Full resolution
                    no_auto_bright=False,    # Allow auto brightness
                    output_bps=8             # 8-bit output
                )
            except Exception as e1:
                print(f"First attempt failed: {e1}")
                try:
                    # Fallback to minimal parameters for maximum compatibility
                    rgb_array = raw.postprocess(
                        use_camera_wb=True,
                        output_bps=8
                    )
                except Exception as e2:
                    print(f"Second attempt failed: {e2}")
                    # Final fallback with default parameters
                    rgb_array = raw.postprocess()
        
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(rgb_array)
        
        print(f"Successfully converted RAW file: {pil_image.size} pixels")
        return pil_image
        
    except Exception as e:
        print(f"Error converting RAW file {arw_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def convert_arw_to_temp_file(arw_path, temp_dir="temp_converted"):
    """
    Convert .arw file to a temporary PNG file for processing
    
    Args:
        arw_path: Path to the .arw file
        temp_dir: Directory to store temporary files
        
    Returns:
        Path to temporary PNG file or None if conversion fails
    """
    if not RAW_SUPPORT:
        return None
        
    # Create temp directory if it doesn't exist
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    try:
        # Convert to PIL Image
        pil_image = convert_arw_to_pil(arw_path)
        if pil_image is None:
            return None
            
        # Create temporary file path
        base_name = os.path.splitext(os.path.basename(arw_path))[0]
        temp_path = os.path.join(temp_dir, f"{base_name}_converted.png")
        
        # Save as PNG
        pil_image.save(temp_path, "PNG")
        
        print(f"Temporary PNG created: {temp_path}")
        return temp_path
        
    except Exception as e:
        print(f"Error creating temporary file for {arw_path}: {e}")
        return None

def rotate_image(image, rotation, direction):
    if not image or rotation == "None":
        return image
    if rotation == "90 Degrees":
        angle = 90 if direction == "Clockwise" else -90
    elif rotation == "180 Degrees":
        angle = 180
    else:
        angle = 0
    return image.rotate(angle, expand=True)

def flip_image(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def get_bounding_box_with_threshold(image, threshold=10):
    arr = np.array(image)
    alpha = arr[:, :, 3]
    rows = np.any(alpha > threshold, axis=1)
    cols = np.any(alpha > threshold, axis=0)
    r_idx = np.where(rows)[0]
    c_idx = np.where(cols)[0]
    if r_idx.size == 0 or c_idx.size == 0:
        return None
    top, bottom = r_idx[0], r_idx[-1]
    left, right = c_idx[0], c_idx[-1]
    if left < right and top < bottom:
        return (left, top, right, bottom)
    else:
        return None

# ADJUSTED: Added a new parameter "bg_method" (defaulting to None) to avoid a NameError.
def position_logic_old(image_path, canvas_size, padding_top, padding_right, padding_bottom, padding_left, use_threshold=True, bg_method=None, is_person=False, snap_to_top=False, snap_to_bottom=False, snap_to_left=False, snap_to_right=False):
    """
    Position and resize an image on a canvas based on snapping, cropped sides, and birefnet logic.
    
    Args:
        image_path (str): Path to the input image.
        canvas_size (tuple): Target canvas size (width, height).
        padding_top, padding_right, padding_bottom, padding_left (int): Padding on each side.
        use_threshold (bool): Use threshold-based bounding box detection.
        bg_method (str): Background removal method ('birefnet', 'birefnet_2', etc.).
        is_person (bool): Treat as a person image (snaps to bottom by default).
        snap_to_top, snap_to_bottom, snap_to_left, snap_to_right (bool): Snap to respective sides.

    Returns:
        tuple: (log, resized_image, x_position, y_position)
    """
    image = Image.open(image_path)
    image = image.convert("RGBA")
    
    # Get the bounding box of the non-blank area with threshold
    if use_threshold:
        bbox = get_bounding_box_with_threshold(image, threshold=10)
    else:
        bbox = image.getbbox()
    log = []
    x,y = 0,0

    if bbox:
        # Check 1 pixel around the image for non-transparent pixels
        width, height = image.size
        cropped_sides = []
        
        # Define tolerance for transparency
        tolerance = 30  # Adjust this value as needed
        
        # Check top edge
        if any(image.getpixel((x, 0))[3] > tolerance for x in range(width)):
            cropped_sides.append("top")
        
        # Check bottom edge
        if any(image.getpixel((x, height-1))[3] > tolerance for x in range(width)):
            cropped_sides.append("bottom")
        
        # Check left edge
        if any(image.getpixel((0, y))[3] > tolerance for y in range(height)):
            cropped_sides.append("left")
        
        # Check right edge
        if any(image.getpixel((width-1, y))[3] > tolerance for y in range(height)):
            cropped_sides.append("right")
        
        if cropped_sides:
            info_message = f"Info for {os.path.basename(image_path)}: The following sides of the image may contain cropped objects: {', '.join(cropped_sides)}"
            print(info_message)
            log.append({"info": info_message})
        else:
            info_message = f"Info for {os.path.basename(image_path)}: The image is not cropped."
            print(info_message)
            log.append({"info": info_message})
        
        # Crop the image to the bounding box
        image = image.crop(bbox)
        log.append({"action": "crop", "bbox": [str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3])]})
        
        # Calculate the new size to expand the image
        target_width, target_height = canvas_size
        aspect_ratio = image.width / image.height
        
        # Determine active snaps
        snaps_active = []
        if padding_top == 0 or snap_to_top:
            snaps_active.append("top")
        if padding_bottom == 0 or snap_to_bottom or is_person:
            snaps_active.append("bottom")
        if padding_left == 0 or snap_to_left:
            snaps_active.append("left")
        if padding_right == 0 or snap_to_right:
            snaps_active.append("right")
        
        # Snap handling
        if snaps_active:
            if "top" in snaps_active and "bottom" in snaps_active:
                # Dual vertical snap: fill height
                new_height = target_height
                new_width = int(new_height * aspect_ratio)
                image = image.resize((new_width, new_height), Image.LANCZOS)
                y = 0
                if "left" in snaps_active:
                    x = 0
                elif "right" in snaps_active:
                    x = target_width - new_width
                else:
                    x = (target_width - new_width) // 2
                log.append({"action": "resize_snap_vertical", "new_width": str(new_width), "new_height": str(new_height)})
                log.append({"action": "position_snap_vertical", "x": str(x), "y": str(y)})
            elif "left" in snaps_active and "right" in snaps_active:
                # Dual horizontal snap: fill width
                new_width = target_width
                new_height = int(new_width / aspect_ratio)
                image = image.resize((new_width, new_height), Image.LANCZOS)
                x = 0
                if "top" in snaps_active:
                    y = 0
                elif "bottom" in snaps_active:
                    y = target_height - new_height
                else:
                    y = (target_height - new_height) // 2
                log.append({"action": "resize_snap_horizontal", "new_width": str(new_width), "new_height": str(new_height)})
                log.append({"action": "position_snap_horizontal", "x": str(x), "y": str(y)})
            else:
                # Original snap logic
                available_width = target_width
                available_height = target_height
                if "left" not in snaps_active:
                    available_width -= padding_left
                if "right" not in snaps_active:
                    available_width -= padding_right
                if "top" not in snaps_active:
                    available_height -= padding_top
                if "bottom" not in snaps_active:
                    available_height -= padding_bottom
                
                if aspect_ratio < 1:  # Portrait
                    new_height = available_height
                    new_width = int(new_height * aspect_ratio)
                    if new_width > available_width:
                        new_width = available_width
                        new_height = int(new_width / aspect_ratio)
                else:  # Landscape
                    new_width = available_width
                    new_height = int(new_width / aspect_ratio)
                    if new_height > available_height:
                        new_height = available_height
                        new_width = int(new_height * aspect_ratio)
                
                image = image.resize((new_width, new_height), Image.LANCZOS)
                if "left" in snaps_active:
                    x = 0
                elif "right" in snaps_active:
                    x = target_width - new_width
                else:
                    x = padding_left + (available_width - new_width) // 2
                if "top" in snaps_active:
                    y = 0
                elif "bottom" in snaps_active:
                    y = target_height - new_height
                else:
                    y = padding_top + (available_height - new_height) // 2
                log.append({"action": "resize", "new_width": str(new_width), "new_height": str(new_height)})
                log.append({"action": "position", "x": str(x), "y": str(y)})
        else:
            # No snaps: use original cropped sides logic
            if len(cropped_sides) == 4:
                # If the image is cropped on all sides, center crop it to fit the canvas
                if aspect_ratio > 1:  # Landscape
                    new_height = target_height
                    new_width = int(new_height * aspect_ratio)
                    left = (new_width - target_width) // 2
                    image = image.resize((new_width, new_height), Image.LANCZOS)
                    image = image.crop((left, 0, left + target_width, target_height))
                else:  # Portrait or square
                    new_width = target_width
                    new_height = int(new_width / aspect_ratio)
                    top = (new_height - target_height) // 2
                    image = image.resize((new_width, new_height), Image.LANCZOS)
                    image = image.crop((0, top, target_width, top + target_height))
                log.append({"action": "center_crop_resize", "new_size": f"{target_width}x{target_height}"})
                x, y = 0, 0
            elif not cropped_sides:
                # If the image is not cropped, expand it from center until it touches the padding
                new_height = target_height - padding_top - padding_bottom
                new_width = int(new_height * aspect_ratio)
                
                if new_width > target_width - padding_left - padding_right:
                    # If width exceeds available space, adjust based on width
                    new_width = target_width - padding_left - padding_right
                    new_height = int(new_width / aspect_ratio)
                
                # Resize the image
                image = image.resize((new_width, new_height), Image.LANCZOS)
                log.append({"action": "resize", "new_width": str(new_width), "new_height": str(new_height)})
                
                x = (target_width - new_width) // 2
                y = target_height - new_height - padding_bottom
            else:
                # Existing logic for partial cropping (unchanged for compatibility)
                # New logic for handling cropped top and left, or top and right
                if set(cropped_sides) == {"top", "left"} or set(cropped_sides) == {"top", "right"}:
                    new_height = target_height - padding_bottom
                    new_width = int(new_height * aspect_ratio)
                    
                    # If new width exceeds canvas width, adjust based on width
                    if new_width > target_width:
                        new_width = target_width
                        new_height = int(new_width / aspect_ratio)
                    
                    # Resize the image
                    image = image.resize((new_width, new_height), Image.LANCZOS)
                    log.append({"action": "resize", "new_width": str(new_width), "new_height": str(new_height)})
                    
                    # Set position
                    if "left" in cropped_sides:
                        x = 0
                    else:  # right in cropped_sides
                        x = target_width - new_width
                    y = 0
                    
                    # If the resized image is taller than the canvas minus padding, crop from the bottom
                    if new_height > target_height - padding_bottom:
                        crop_bottom = new_height - (target_height - padding_bottom)
                        image = image.crop((0, 0, new_width, new_height - crop_bottom))
                        new_height = target_height - padding_bottom
                        log.append({"action": "crop_vertical", "bottom_pixels_removed": str(crop_bottom)})
                    
                    log.append({"action": "position", "x": str(x), "y": str(y)})
                elif set(cropped_sides) == {"bottom", "left"} or set(cropped_sides) == {"bottom", "right"}:
                    # Handle bottom & left or bottom & right cropped images
                    new_height = target_height - padding_top
                    new_width = int(new_height * aspect_ratio)
                    
                    # If new width exceeds canvas width, adjust based on width
                    if new_width > target_width - padding_left - padding_right:
                        new_width = target_width - padding_left - padding_right
                        new_height = int(new_width / aspect_ratio)
                    
                    # Resize the image without cropping or stretching
                    image = image.resize((new_width, new_height), Image.LANCZOS)
                    log.append({"action": "resize", "new_width": str(new_width), "new_height": str(new_height)})
                    
                    # Set position
                    if "left" in cropped_sides:
                        x = 0
                    else:  # right in cropped_sides
                        x = target_width - new_width
                    y = target_height - new_height
                    
                    log.append({"action": "position", "x": str(x), "y": str(y)})
                elif set(cropped_sides) == {"bottom", "left", "right"}:
                    # Expand the image from the center
                    new_width = target_width
                    new_height = int(new_width / aspect_ratio)
                    
                    if new_height < target_height:
                        new_height = target_height
                        new_width = int(new_height * aspect_ratio)
                    
                    image = image.resize((new_width, new_height), Image.LANCZOS)
                    
                    # Crop to fit the canvas
                    left = (new_width - target_width) // 2
                    top = 0
                    image = image.crop((left, top, left + target_width, top + target_height))
                    
                    log.append({"action": "expand_and_crop", "new_size": f"{target_width}x{target_height}"})
                    x, y = 0, 0
                elif cropped_sides == ["top"]:
                    # New logic for handling only top-cropped images
                    if image.width > image.height:
                        new_width = target_width
                        new_height = int(target_width / aspect_ratio)
                    else:
                        new_height = target_height - padding_bottom
                        new_width = int(new_height * aspect_ratio)
                    
                    # Resize the image
                    image = image.resize((new_width, new_height), Image.LANCZOS)
                    log.append({"action": "resize", "new_width": str(new_width), "new_height": str(new_height)})
                    
                    x = (target_width - new_width) // 2
                    y = 0  # Align to top
                    
                    # Apply padding only to non-cropped sides
                    x = max(padding_left, min(x, target_width - new_width - padding_right))
                elif cropped_sides in [["right"], ["left"]]:
                    # New logic for handling only right-cropped or left-cropped images
                    if image.width > image.height:
                        new_width = target_width - max(padding_left, padding_right)
                        new_height = int(new_width / aspect_ratio)
                    else:
                        new_height = target_height - padding_top - padding_bottom
                        new_width = int(new_height * aspect_ratio)
                    
                    # Resize the image
                    image = image.resize((new_width, new_height), Image.LANCZOS)
                    log.append({"action": "resize", "new_width": str(new_width), "new_height": str(new_height)})
                    
                    if cropped_sides == ["right"]:
                        x = target_width - new_width  # Align to right
                    else:  # cropped_sides == ["left"]
                        x = 0  # Align to left
                    y = target_height - new_height - padding_bottom  # Respect bottom padding
                    
                    # Ensure top padding is respected
                    if y < padding_top:
                        y = padding_top
                        
                    log.append({"action": "position", "x": str(x), "y": str(y)})
                elif set(cropped_sides) == {"left", "right"}:
                    # Logic for handling images cropped on both left and right sides
                    new_width = target_width  # Expand to full width of canvas
                    
                    # Calculate the aspect ratio of the original image
                    aspect_ratio = image.width / image.height
                    
                    # Calculate the new height while maintaining aspect ratio
                    new_height = int(new_width / aspect_ratio)
                    
                    # Resize the image
                    image = image.resize((new_width, new_height), Image.LANCZOS)
                    log.append({"action": "resize", "new_width": str(new_width), "new_height": str(new_height)})
                    
                    # Set horizontal position (always 0 as it spans full width)
                    x = 0
                    
                    # Calculate vertical position to respect bottom padding
                    y = target_height - new_height - padding_bottom
                    
                    # If the resized image is taller than the canvas, crop from the top only
                    if new_height > target_height - padding_bottom:
                        crop_top = new_height - (target_height - padding_bottom)
                        image = image.crop((0, crop_top, new_width, new_height))
                        new_height = target_height - padding_bottom
                        y = 0
                        log.append({"action": "crop_vertical", "top_pixels_removed": str(crop_top)})
                    else:
                        # Align the image to the bottom with padding
                        y = target_height - new_height - padding_bottom
                    
                    log.append({"action": "position", "x": str(x), "y": str(y)})
                elif cropped_sides == ["bottom"]:
                    # Logic for handling images cropped on the bottom side
                    # Calculate the aspect ratio of the original image
                    aspect_ratio = image.width / image.height
                    
                    if aspect_ratio < 1:  # Portrait orientation
                        new_height = target_height - padding_top  # Full height with top padding
                        new_width = int(new_height * aspect_ratio)
                        
                        # If the new width exceeds the canvas width, adjust it
                        if new_width > target_width:
                            new_width = target_width
                            new_height = int(new_width / aspect_ratio)
                    else:  # Landscape orientation
                        new_width = target_width - padding_left - padding_right
                        new_height = int(new_width / aspect_ratio)
                        
                        # If the new height exceeds the canvas height, adjust it
                        if new_height > target_height:
                            new_height = target_height
                            new_width = int(new_height * aspect_ratio)
                    
                    # Resize the image
                    image = image.resize((new_width, new_height), Image.LANCZOS)
                    log.append({"action": "resize", "new_width": str(new_width), "new_height": str(new_height)})
                    
                    # Set horizontal position (centered)
                    x = (target_width - new_width) // 2
                    
                    # Set vertical position (touching bottom edge for all cases)
                    y = target_height - new_height
                    
                    log.append({"action": "position", "x": str(x), "y": str(y)})
                else:
                    # Use the original resizing logic for other partially cropped images
                    if image.width > image.height:
                        new_width = target_width
                        new_height = int(target_width / aspect_ratio)
                    else:
                        new_height = target_height
                        new_width = int(target_height * aspect_ratio)
                    
                    # Resize the image
                    image = image.resize((new_width, new_height), Image.LANCZOS)
                    log.append({"action": "resize", "new_width": str(new_width), "new_height": str(new_height)})
                    
                    # Center horizontally for all images
                    x = (target_width - new_width) // 2
                    y = target_height - new_height - padding_bottom
                    
                    # Adjust positions for cropped sides
                    if "top" in cropped_sides:
                        y = 0
                    elif "bottom" in cropped_sides:
                        y = target_height - new_height
                    if "left" in cropped_sides:
                        x = 0
                    elif "right" in cropped_sides:
                        x = target_width - new_width
                    
                    # Apply padding only to non-cropped sides, but keep horizontal centering
                    if "left" not in cropped_sides and "right" not in cropped_sides:
                        x = (target_width - new_width) // 2  # Always center horizontally
                    if "top" not in cropped_sides and "bottom" not in cropped_sides:
                        y = max(padding_top, min(y, target_height - new_height - padding_bottom))

        # Additional logic for birefnet methods if specified
        if bg_method == 'birefnet' or bg_method == 'birefnet_2':
            # Calculate target size (half of canvas size)
            target_width = min(canvas_size[0] // 2, image.width)
            target_height = min(canvas_size[1] // 2, image.height)
            
            # Maintain aspect ratio while resizing
            aspect_ratio = image.width / image.height
            if aspect_ratio > 1:  # Landscape
                new_width = target_width
                new_height = int(new_width / aspect_ratio)
            else:  # Portrait or square
                new_height = target_height
                new_width = int(new_height * aspect_ratio)
            
            # Resize the image
            image = image.resize((new_width, new_height), Image.LANCZOS)
            
            # Center the image on canvas
            x = (canvas_size[0] - new_width) // 2
            y = (canvas_size[1] - new_height) // 2
            
            log.append({
                "action": "birefnet_resize",
                "new_size": f"{new_width}x{new_height}",
                "position": f"{x},{y}"
            })
            
            return log, image, x, y

    return log, image, x, y

def position_logic_none(image, canvas_size):
    target_width, target_height = canvas_size
    
    # Calculate aspect ratio
    aspect_ratio = image.width / image.height
    
    # Calculate new size that fits the canvas
    if aspect_ratio > 1:  # Landscape
        new_width = target_width
        new_height = int(new_width / aspect_ratio)
    else:  # Portrait or square
        new_height = target_height
        new_width = int(new_height * aspect_ratio)
    
    # Resize image
    image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Calculate centered position
    x = (target_width - new_width) // 2
    y = (target_height - new_height) // 2
    
    log = [{"action": "resize_and_center", "new_size": f"{new_width}x{new_height}", "position": f"{x},{y}"}]
    
    return log, image, x, y

def position_with_padding(image, canvas_size, padding_top, padding_right, padding_bottom, padding_left):
    """
    Resize & position image so that it fills the available box defined by padding,
    without leaving white-space on left/right sides.
    
    Args:
        image: PIL Image object
        canvas_size: tuple (width, height) of canvas
        padding_top, padding_right, padding_bottom, padding_left: padding values
        
    Returns:
        tuple: (resized_image, x_position, y_position, log)
    """
    # Canvas dimensions
    W, H = canvas_size
    
    # Compute available area
    avail_w = W - padding_left - padding_right
    avail_h = H - padding_top - padding_bottom
    
    # Ensure available area is positive
    if avail_w <= 0 or avail_h <= 0:
        print(f"Warning: Available area is too small ({avail_w}x{avail_h}). Using minimal padding.")
        avail_w = max(avail_w, W // 2)
        avail_h = max(avail_h, H // 2)
        padding_left = min(padding_left, (W - avail_w) // 2)
        padding_top = min(padding_top, (H - avail_h) // 2)
    
    # Original aspect ratio
    w0, h0 = image.size
    aspect = w0 / h0
    
    print(f"Canvas: {W}x{H}, Available: {avail_w}x{avail_h}, Original: {w0}x{h0}, Aspect: {aspect:.3f}")
    print(f"Padding: top={padding_top}, right={padding_right}, bottom={padding_bottom}, left={padding_left}")
    
    # ALWAYS fill the full width available (no whitespace on sides)
    new_w = avail_w
    new_h = int(new_w / aspect)
    
    print(f"Calculated size to fill width: {new_w}x{new_h}")
    
    # Resize the image maintaining aspect ratio (fill width completely)
    img_resized = image.resize((new_w, new_h), Image.LANCZOS)
    
    # If the resized height exceeds available height, crop from bottom
    if new_h > avail_h:
        print(f"Height {new_h} exceeds available height {avail_h} - cropping from bottom")
        print(f"Will crop {new_h - avail_h}px from bottom to fit in {avail_h}px height")
        
        # Crop from bottom while maintaining the full width
        # This preserves aspect ratio and prevents stretching
        crop_bottom = new_h - avail_h
        img_resized = img_resized.crop((0, 0, new_w, new_h - crop_bottom))
        new_h = avail_h  # Update height after cropping
        
        print(f"After cropping: {new_w}x{new_h} (no stretching, cropped from bottom)")
    else:
        print(f"Image fits perfectly: {new_w}x{new_h} (no cropping needed)")
    
    print(f"Final size: {new_w}x{new_h} (maintains aspect ratio, fills full width)")
    
    # Position: x always starts at padding_left to align with left padding
    x = padding_left
    
    # Y position: align to top padding (as requested by user)
    y = padding_top
    
    # Verify no whitespace will remain
    actual_right_edge = x + new_w
    expected_right_edge = W - padding_right
    
    print(f"Whitespace check: right edge at {actual_right_edge}, should reach {expected_right_edge}")
    
    if actual_right_edge < expected_right_edge:
        print(f"⚠️  Warning: Will have {expected_right_edge - actual_right_edge}px whitespace on right")
    else:
        print(f"✅ No whitespace - image fills full available width")
    
    # Create detailed log
    log = [
        {"action": "calculate_available_area", "available_size": f"{avail_w}x{avail_h}"},
        {"action": "resize_maintain_aspect", "new_size": f"{new_w}x{new_h}", "method": "fill_width_preserve_ratio"},
        {"action": "position_with_padding", "position": f"({x},{y})", 
         "padding_applied": f"top={padding_top}, left={padding_left}"},
        {"action": "whitespace_check", "right_edge": f"{actual_right_edge} (target: {expected_right_edge})"}
    ]
    
    # Add cropping info to log if cropping occurred
    original_h_calc = int(avail_w / aspect)
    if original_h_calc > avail_h:
        crop_amount = original_h_calc - avail_h
        log.append({"action": "crop_bottom", "cropped_pixels": f"{crop_amount}px", "reason": "maintain_aspect_ratio"})
    
    print(f"Final positioning: ({x}, {y}) with size {new_w}x{new_h}")
    
    return img_resized, x, y, log

def preserve_color_profile(original_path, img):
    """Preserve the original image's color profile if it exists"""
    try:
        # Try to get the original ICC profile
        with open(original_path, 'rb') as f:
            original = Image.open(f)
            if 'icc_profile' in original.info:
                return img, original.info['icc_profile']
    except Exception:
        pass
    
    # Return the original image if no profile was found or error occurred
    return img, None

def auto_detect_zero_padding(image_path, bg_method='bria', threshold=30, edge_threshold=0.10, distance_threshold=0.15):
    """
    Automatically detect which sides should have zero padding based on advanced object analysis
    Uses multiple detection methods including edge detection, object boundary analysis, and distance calculations
    
    Args:
        image_path: Path to the image file
        bg_method: Background removal method to use for analysis
        threshold: Alpha threshold for detecting non-transparent pixels
        edge_threshold: Percentage of edge that needs to have content to be considered "touching"
        distance_threshold: Distance threshold for snap detection
        
    Returns:
        list: Sides that should have zero padding (e.g., ["top", "left"])
    """
    try:
        print(f"\n=== AUTO SNAP DETECTION for {os.path.basename(image_path)} ===")
        
        # Quick dependency check
        try:
            cv2_version = cv2.__version__
            numpy_version = np.__version__
            print(f"📦 Dependencies: OpenCV {cv2_version}, NumPy {numpy_version}")
        except Exception as dep_e:
            print(f"⚠️  Dependency check failed: {dep_e}")
            return []
        
        # Load original image for comprehensive analysis
        original_img = Image.open(image_path)
        
        # Convert to RGB for consistent processing
        if original_img.mode == "CMYK":
            rgb_img, _ = convert_image_to_rgb(original_img)
        else:
            rgb_img = original_img.convert('RGB')
        
        img_array = np.array(rgb_img)
        height, width = img_array.shape[:2]
        
        print(f"Image dimensions: {width}x{height}")
        
        zero_padding_sides = []
        contours = None
        # Initialize distance variables to avoid UnboundLocalError
        dist_to_top = dist_to_bottom = dist_to_left = dist_to_right = 0
        x = y = w = h = 0
        
        # === METHOD 1: ADVANCED OBJECT BOUNDARY ANALYSIS ===
        print("Method 1: Advanced object boundary analysis...")
        
        try:
            # Create multiple masks for better object detection
            # 1. Non-white mask (for white backgrounds)
            lower_white = np.array([240, 240, 240])
            upper_white = np.array([255, 255, 255])
            white_mask = cv2.inRange(img_array, lower_white, upper_white)
            object_mask_white = cv2.bitwise_not(white_mask)
            
            # 2. Edge detection mask
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # 3. Color variance mask (areas with significant color variation)
            blur = cv2.GaussianBlur(img_array, (5, 5), 0)
            diff = cv2.absdiff(img_array, blur)
            variance_mask = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
            _, variance_mask = cv2.threshold(variance_mask, 10, 255, cv2.THRESH_BINARY)
            
            # Combine masks for comprehensive object detection
            combined_mask = cv2.bitwise_or(object_mask_white, edges)
            combined_mask = cv2.bitwise_or(combined_mask, variance_mask)
            
            # Clean up the mask
            kernel = np.ones((3, 3), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Find contours to get main object
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get the largest contour (main object)
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                print(f"Main object bounding box: x={x}, y={y}, w={w}, h={h}")
                
                # Calculate distances from object to image edges (in pixels)
                dist_to_top = y
                dist_to_bottom = height - (y + h)
                dist_to_left = x
                dist_to_right = width - (x + w)
                
                print(f"Distances to edges (pixels): top={dist_to_top}, bottom={dist_to_bottom}, left={dist_to_left}, right={dist_to_right}")
                
                # === SMART THRESHOLD CALCULATION ===
                # Base thresholds on image size - larger images can tolerate larger distances
                canvas_diagonal = (width**2 + height**2)**0.5
                
                # Dynamic thresholds based on image size
                if width >= 1000 or height >= 1000:  # Large images
                    # Special case for 1000x1000 images
                    if width == 1000 and height == 1000:
                        # Custom thresholds for 1000x1000 images - fine-tuned based on user feedback
                        top_threshold = -1  # No top snaps allowed
                        bottom_threshold = 1  # Only distance 0 triggers bottom
                        left_threshold = 100  # Allows 65,98 but not 120,212,230
                        right_threshold = 130  # Allows 99,127 but not higher distances
                        print(f"Using custom 1000x1000 thresholds")
                    # Special case for 1080x1080 images
                    elif width == 1080 and height == 1080:
                        # Custom thresholds for 1080x1080 images
                        top_threshold = 70  # Changed back from 68 to 70
                        bottom_threshold = 160  # Keep at 160 (allows 159 to snap)
                        left_threshold = 64.0  # Keep at 64.0 (180 won't snap)
                        right_threshold = 120  # Reduced from 201 to 120 (so 160 and 198 won't trigger, but 59 will)
                        print(f"Using custom 1080x1080 thresholds")
                    else:
                        # Original logic for other large images
                        base_threshold = 80
                        relative_threshold = 0.08  # 8% of dimension
                        top_threshold = min(base_threshold, height * relative_threshold)
                        bottom_threshold = min(base_threshold * 1.5, height * relative_threshold * 1.5)
                        left_threshold = min(base_threshold * 0.8, width * relative_threshold * 0.8)
                        right_threshold = min(base_threshold, width * relative_threshold)
                elif width >= 500 or height >= 500:  # Medium images
                    base_threshold = 50
                    relative_threshold = 0.10  # 10% of dimension
                    top_threshold = min(base_threshold, height * relative_threshold)
                    bottom_threshold = min(base_threshold * 1.5, height * relative_threshold * 1.5)
                    left_threshold = min(base_threshold * 0.8, width * relative_threshold * 0.8)
                    right_threshold = min(base_threshold, width * relative_threshold)
                else:  # Small images
                    base_threshold = 30
                    relative_threshold = 0.12  # 12% of dimension
                    top_threshold = min(base_threshold, height * relative_threshold)
                    bottom_threshold = min(base_threshold * 1.5, height * relative_threshold * 1.5)
                    left_threshold = min(base_threshold * 0.8, width * relative_threshold * 0.8)
                    right_threshold = min(base_threshold, width * relative_threshold)
                
                print(f"Calculated thresholds: top={top_threshold:.1f}, bottom={bottom_threshold:.1f}, left={left_threshold:.1f}, right={right_threshold:.1f}")
                
                # Check each edge for snap conditions
                if dist_to_top <= top_threshold:
                    zero_padding_sides.append("top")
                    print(f"✓ SNAP TO TOP: distance {dist_to_top} <= threshold {top_threshold:.1f}")
                
                if dist_to_bottom <= bottom_threshold:
                    zero_padding_sides.append("bottom")
                    print(f"✓ SNAP TO BOTTOM: distance {dist_to_bottom} <= threshold {bottom_threshold:.1f}")
                
                if dist_to_left <= left_threshold:
                    zero_padding_sides.append("left")
                    print(f"✓ SNAP TO LEFT: distance {dist_to_left} <= threshold {left_threshold:.1f}")
                
                if dist_to_right <= right_threshold:
                    zero_padding_sides.append("right")
                    print(f"✓ SNAP TO RIGHT: distance {dist_to_right} <= threshold {right_threshold:.1f}")
            else:
                print("⚠️  No contours found in Method 1")
                
        except Exception as e:
            print(f"⚠️  Method 1 failed: {e}")
            print("Continuing with fallback methods...")
        
        # === METHOD 2: SIMPLE COLOR-BASED ANALYSIS (Backup) ===
        if not zero_padding_sides:
            print("Method 2: Simple color-based analysis (backup)...")
            try:
                # Convert to grayscale for simpler analysis
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                
                # Simple threshold to find non-background areas
                _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
                
                # Find bounding box of non-white areas
                coords = cv2.findNonZero(binary)
                if coords is not None:
                    x, y, w, h = cv2.boundingRect(coords)
                    
                    print(f"Simple method - Object bounds: x={x}, y={y}, w={w}, h={h}")
                    
                    # Calculate distances
                    dist_to_top = y
                    dist_to_bottom = height - (y + h)
                    dist_to_left = x
                    dist_to_right = width - (x + w)
                    
                    print(f"Simple method - Distances: top={dist_to_top}, bottom={dist_to_bottom}, left={dist_to_left}, right={dist_to_right}")
                    
                    # Use simpler thresholds - match the main method for consistency
                    if width == 1000 and height == 1000:
                        # Use same custom thresholds as Method 1 for 1000x1000 images - fine-tuned
                        simple_top_threshold = -1  # No top snaps allowed
                        simple_bottom_threshold = 1  # Only distance 0 triggers bottom
                        simple_left_threshold = 100  # Allows 65,98 but not 120,212,230
                        simple_right_threshold = 130  # Allows 99,127 but not higher distances
                        print(f"Using custom simple thresholds for 1000x1000: top={simple_top_threshold}, bottom={simple_bottom_threshold}, left={simple_left_threshold}, right={simple_right_threshold}")
                    elif width == 1080 and height == 1080:
                        # Use same custom thresholds as Method 1 for 1080x1080 images
                        simple_top_threshold = 70  # Changed back from 68 to 70
                        simple_bottom_threshold = 160  # Keep at 160 (allows 159 to snap)
                        simple_left_threshold = 64.0  # Keep at 64.0 (180 won't snap)
                        simple_right_threshold = 120  # Reduced from 201 to 120 (so 160 and 198 won't trigger, but 59 will)
                        print(f"Using custom 1080x1080 thresholds")
                    else:
                        # Original simple threshold logic for other sizes
                        simple_threshold = 60 if width >= 1000 else 40
                        simple_top_threshold = simple_threshold
                        simple_bottom_threshold = simple_threshold * 1.5
                        simple_left_threshold = simple_threshold * 0.8
                        simple_right_threshold = simple_threshold
                    
                    if dist_to_top <= simple_top_threshold:
                        zero_padding_sides.append("top")
                        print(f"✓ SIMPLE: SNAP TO TOP ({dist_to_top} <= {simple_top_threshold})")
                    
                    if dist_to_bottom <= simple_bottom_threshold:
                        zero_padding_sides.append("bottom")
                        print(f"✓ SIMPLE: SNAP TO BOTTOM ({dist_to_bottom} <= {simple_bottom_threshold})")
                    
                    if dist_to_left <= simple_left_threshold:
                        zero_padding_sides.append("left")
                        print(f"✓ SIMPLE: SNAP TO LEFT ({dist_to_left} <= {simple_left_threshold})")
                    
                    if dist_to_right <= simple_right_threshold:
                        zero_padding_sides.append("right")
                        print(f"✓ SIMPLE: SNAP TO RIGHT ({dist_to_right} <= {simple_right_threshold})")
                else:
                    print("⚠️  No non-white areas found in simple method")
                    
            except Exception as e:
                print(f"⚠️  Method 2 failed: {e}")
        
        # === METHOD 3: BACKGROUND REMOVAL ANALYSIS (Confirmation) ===
        if len(zero_padding_sides) < 2:  # Only use as fallback if we didn't find enough snaps
            print("Method 3: Background removal analysis for confirmation...")
            
            try:
                mask = None
                if bg_method == 'rembg':
                    mask = remove_background_rembg(image_path)
                elif bg_method == 'bria':
                    mask = remove_background_bria(image_path)
                elif bg_method == 'photoroom':
                    mask = remove_background_photoroom(image_path)
                elif bg_method == 'birefnet':
                    mask = remove_background_birefnet(image_path)
                elif bg_method == 'birefnet_2':
                    mask = remove_background_birefnet_2(image_path)
                elif bg_method == 'birefnet_hr':
                    mask = remove_background_birefnet_hr(image_path)
                else:
                    mask = Image.open(image_path).convert("RGBA")
                
                if mask and hasattr(mask, 'size'):
                    mask_width, mask_height = mask.size
                    
                    # Convert to numpy for faster processing
                    mask_array = np.array(mask)
                    alpha_channel = mask_array[:, :, 3] if mask_array.shape[2] == 4 else np.ones((mask_height, mask_width), dtype=np.uint8) * 255
                    
                    # Find bounding box of the object
                    rows = np.any(alpha_channel > threshold, axis=1)
                    cols = np.any(alpha_channel > threshold, axis=0)
                    
                    if np.any(rows) and np.any(cols):
                        # Get object boundaries
                        top_bound = np.argmax(rows)
                        bottom_bound = len(rows) - 1 - np.argmax(rows[::-1])
                        left_bound = np.argmax(cols)
                        right_bound = len(cols) - 1 - np.argmax(cols[::-1])
                        
                        print(f"BG removal analysis - Object bounds: top={top_bound}, bottom={bottom_bound}, left={left_bound}, right={right_bound}")
                        
                        # Calculate edge distances as percentage of image size
                        bg_top_distance = top_bound
                        bg_bottom_distance = mask_height - 1 - bottom_bound
                        bg_left_distance = left_bound
                        bg_right_distance = mask_width - 1 - right_bound
                        
                        print(f"BG removal distances: top={bg_top_distance}, bottom={bg_bottom_distance}, left={bg_left_distance}, right={bg_right_distance}")
                        
                        # Use similar thresholds as Method 1
                        bg_top_threshold = min(50, mask_height * 0.08)
                        bg_bottom_threshold = min(75, mask_height * 0.12)
                        bg_left_threshold = min(40, mask_width * 0.06)
                        bg_right_threshold = min(50, mask_width * 0.08)
                        
                        # Only add if not already detected and passes threshold
                        if bg_top_distance < bg_top_threshold and "top" not in zero_padding_sides:
                            zero_padding_sides.append("top")
                            print(f"✓ BG ANALYSIS: Added SNAP TO TOP")
                        
                        if bg_left_distance < bg_left_threshold and "left" not in zero_padding_sides:
                            zero_padding_sides.append("left")
                            print(f"✓ BG ANALYSIS: Added SNAP TO LEFT")
                        
                        if bg_right_distance < bg_right_threshold and "right" not in zero_padding_sides:
                            zero_padding_sides.append("right")
                            print(f"✓ BG ANALYSIS: Added SNAP TO RIGHT")
                        
                        if bg_bottom_distance < bg_bottom_threshold and "bottom" not in zero_padding_sides:
                            zero_padding_sides.append("bottom")
                            print(f"✓ BG ANALYSIS: Added SNAP TO BOTTOM")
                    else:
                        print("⚠️  No object found in background removal analysis")
                else:
                    print("⚠️  Background removal failed or returned None")
            
            except Exception as e:
                print(f"⚠️  Background removal analysis failed: {e}")
        
        # === METHOD 4: INTELLIGENT COMBINATION RULES ===
        if zero_padding_sides:
            print("Method 4: Applying intelligent combination rules...")
            
            # Store original detections for analysis
            original_detections = zero_padding_sides.copy()
        
        # Rule 1: Handle conflicting vertical combinations
        if "top" in zero_padding_sides and "bottom" in zero_padding_sides:
            # Special exception for 1080x1080 images - allow both top and bottom
            if width == 1080 and height == 1080:
                print("✅ 1080x1080 image - keeping both TOP and BOTTOM snaps (special rule)")
            else:
                print("⚠️  Both top and bottom detected - analyzing which is more critical...")
                if contours:
                    # Check which side has smaller distance to make better decision
                    if dist_to_top < dist_to_bottom:
                        zero_padding_sides.remove("bottom")
                        print("   → Kept TOP (closer to edge), removed BOTTOM")
                    elif dist_to_bottom < dist_to_top:
                        zero_padding_sides.remove("top")
                        print("   → Kept BOTTOM (closer to edge), removed TOP")
                    else:
                        # Equal distances - keep both for full height
                        print("   → Kept BOTH (equal distances - will fill full height)")
        
        # Rule 2: Handle conflicting horizontal combinations
        if "left" in zero_padding_sides and "right" in zero_padding_sides:
            print("⚠️  Both left and right detected - analyzing which is more critical...")
            if contours:
                # Check which side has smaller distance
                if dist_to_left < dist_to_right:
                    zero_padding_sides.remove("right")
                    print("   → Kept LEFT (closer to edge), removed RIGHT")
                elif dist_to_right < dist_to_left:
                    zero_padding_sides.remove("left")
                    print("   → Kept RIGHT (closer to edge), removed LEFT")
                else:
                    # Equal distances - keep both for full width
                    print("   → Kept BOTH (equal distances - will fill full width)")
        
        # Rule 3: Smart combinations based on object shape and position
        if contours and len(zero_padding_sides) >= 2:
            # Calculate object aspect ratio and position
            object_aspect_ratio = w / h if h > 0 else 1
            center_x_relative = (x + w/2) / width
            center_y_relative = (y + h/2) / height
            
            print(f"Object analysis: aspect_ratio={object_aspect_ratio:.2f}, center=({center_x_relative:.2f}, {center_y_relative:.2f})")
            
            # Rule 3a: For wide objects (landscape), prefer horizontal snaps
            if object_aspect_ratio > 1.5:  # Wide object
                if len([s for s in zero_padding_sides if s in ["left", "right"]]) > 0:
                    print("   💡 Wide object detected - prioritizing horizontal snaps")
                    # If we have both vertical and horizontal snaps, keep horizontals
                    if len(zero_padding_sides) > 2:
                        # Keep horizontal snaps, remove some vertical if needed
                        if "top" in zero_padding_sides and "bottom" in zero_padding_sides:
                            # Remove the less critical vertical snap
                            if center_y_relative < 0.5:  # Object closer to top
                                if "bottom" in zero_padding_sides:
                                    zero_padding_sides.remove("bottom")
                                    print("     → Removed BOTTOM (object closer to top)")
                            else:  # Object closer to bottom
                                if "top" in zero_padding_sides:
                                    zero_padding_sides.remove("top")
                                    print("     → Removed TOP (object closer to bottom)")
            
            # Rule 3b: For tall objects (portrait), prefer vertical snaps
            elif object_aspect_ratio < 0.7:  # Tall object
                if len([s for s in zero_padding_sides if s in ["top", "bottom"]]) > 0:
                    print("   💡 Tall object detected - prioritizing vertical snaps")
                    # If we have both vertical and horizontal snaps, keep verticals
                    if len(zero_padding_sides) > 2:
                        # Keep vertical snaps, remove some horizontal if needed
                        if "left" in zero_padding_sides and "right" in zero_padding_sides:
                            # Remove the less critical horizontal snap
                            if center_x_relative < 0.5:  # Object closer to left
                                if "right" in zero_padding_sides:
                                    zero_padding_sides.remove("right")
                                    print("     → Removed RIGHT (object closer to left)")
                            else:  # Object closer to right
                                if "left" in zero_padding_sides:
                                    zero_padding_sides.remove("left")
                                    print("     → Removed LEFT (object closer to right)")
        
        # Rule 4: Prevent excessive snapping (max 2-3 sides for most cases)
        if len(zero_padding_sides) > 3:
            print("⚠️  Too many snaps detected (>3) - applying priority filter...")
            # Priority order based on common design patterns
            priority_order = ["bottom", "left", "right", "top"]
            
            # Keep only top 2-3 most important snaps
            filtered_snaps = []
            for priority_snap in priority_order:
                if priority_snap in zero_padding_sides:
                    filtered_snaps.append(priority_snap)
                if len(filtered_snaps) >= 3:  # Limit to 3 max
                    break
            
            zero_padding_sides = filtered_snaps
            print(f"   → Filtered to priority snaps: {zero_padding_sides}")
        
        # Rule 5: Validate final combination makes sense
        final_combination = set(zero_padding_sides)
        
        # Check for invalid combinations
        if len(final_combination) == 4:
            print("⚠️  All four sides selected - this will fill entire canvas")
        elif "top" in final_combination and "bottom" in final_combination and len(final_combination) == 2:
            print("✅ Vertical fill detected - object will fill full height")
        elif "left" in final_combination and "right" in final_combination and len(final_combination) == 2:
            print("✅ Horizontal fill detected - object will fill full width")
        elif len(final_combination) == 1:
            print("✅ Single side snap detected - object will align to one edge")
        elif len(final_combination) == 3:
            print("✅ Three-side snap detected - object will fill canvas except one side")
        
        # Final results
        print(f"=== FINAL AUTO SNAP DETECTION RESULT ===")
        if zero_padding_sides:
            if original_detections != zero_padding_sides:
                print(f"📊 Original detections: {original_detections}")
                print(f"🧠 After intelligent filtering: {zero_padding_sides}")
            print(f"✅ Recommended snaps for {os.path.basename(image_path)}: {zero_padding_sides}")
        else:
            print(f"❌ No snaps recommended for {os.path.basename(image_path)}")
        print("=" * 45)
        
        return zero_padding_sides
        
    except Exception as e:
        print(f"Error in auto snap detection for {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return []

def test_auto_snap_detection():
    """
    Simple test function to verify auto snap detection works
    """
    print("\n🧪 TESTING AUTO SNAP DETECTION...")
    
    # Create a simple test image with object near edges
    test_img = Image.new('RGB', (1080, 1080), 'white')
    draw = ImageDraw.Draw(test_img)
    
    # Draw object close to left and top edges (should trigger left and top snaps)
    draw.rectangle([30, 20, 800, 900], fill='blue')
    
    # Save test image
    test_path = "test_auto_snap.png"
    test_img.save(test_path)
    
    try:
        # Test the detection
        result = auto_detect_zero_padding(test_path, 'none')
        print(f"🧪 Test result: {result}")
        
        # Clean up
        if os.path.exists(test_path):
            os.remove(test_path)
            
        return result
        
    except Exception as e:
        print(f"🧪 Test failed: {e}")
        if os.path.exists(test_path):
            os.remove(test_path)
        return []

def process_single_image(
    image_path,
    output_folder,
    bg_method,
    canvas_size_name,
    output_format,
    bg_choice,
    custom_color,
    watermark_path=None,
    twibbon_path=None,
    rotation=None,
    direction=None,
    flip=False,
    use_old_position=True,
    zero_padding=None,  # New parameter: list of sides with zero padding ["top", "bottom", "left", "right"]
    auto_detect_padding=False,  # New parameter: automatically detect zero padding
    sheet_data=None,  # New parameter: DataFrame with sheet data (if provided)
    use_qwen=False,  # New parameter: enable Qwen classification
    snap_to_bottom=False,
    snap_to_top=False,
    snap_to_left=False,
    snap_to_right=False,
    auto_snap=False
):
    # ========== RAW FILE CONVERSION ==========
    # Check if input is a RAW (.arw) file and convert it first
    was_arw_converted = False
    arw_temp_path = None
    
    if image_path.lower().endswith('.arw'):
        print(f"Detected RAW file: {os.path.basename(image_path)}")
        arw_temp_path = convert_arw_to_temp_file(image_path)
        if arw_temp_path:
            working_image_path = arw_temp_path
            was_arw_converted = True
            print(f"RAW file converted for processing: {working_image_path}")
        else:
            print(f"Failed to convert RAW file: {image_path}")
            return None, None, None
    else:
        working_image_path = image_path
    
    # ========== TEXT DETECTION AND AUTO CROPPING ==========
    # Check for reference text and crop if found
    processed_image_path = crop_bottom_if_reference_text(working_image_path, crop_pixels=36)
    
    # Track if cropping was applied for logging
    was_cropped = processed_image_path != working_image_path
    if was_cropped:
        print(f"Using cropped image for processing: {processed_image_path}")
    
    # Use the processed (potentially cropped) image path for the rest of the processing
    working_image_path = processed_image_path
    
    filename = os.path.basename(image_path)  # Keep original filename for output
    base_no_ext, ext = os.path.splitext(filename.lower())
    add_padding_line = False
    classification_result = None  # Initialize classification result

    # ================== FULL SET OF CANVAS SIZE IFS ==================
    # Initialize default padding values first
    if canvas_size_name == 'Aetrex-L/S':
        canvas_size = (1080, 1080)
        padding_top = 180
        padding_right = 200
        padding_bottom = 180
        padding_left = 200
    elif canvas_size_name == 'Allbirds-L/S':
        canvas_size = (1080, 1080)
        padding_top = 180
        padding_right = 200
        padding_bottom = 180
        padding_left = 200
    elif canvas_size_name == 'Backjoy-L/S':
        canvas_size = (1080, 1080)
        padding_top = 180
        padding_right = 72
        padding_bottom = 180
        padding_left = 72
    elif canvas_size_name == 'Beecho-L/S':
        canvas_size = (1080, 1080)
        padding_top = 200
        padding_right = 200
        padding_bottom = 180
        padding_left = 200
    elif canvas_size_name == 'Billabong-L/S':
        canvas_size = (1080, 1080)
        padding_top = 180
        padding_right = 200
        padding_bottom = 180
        padding_left = 200
    elif canvas_size_name == 'Birkenstock-L/S':
        canvas_size = (1080, 1080)
        padding_top = 180
        padding_right = 200
        padding_bottom = 180
        padding_left = 200
    elif canvas_size_name == 'Bratpack-L/S':
        canvas_size = (1080, 1080)
        padding_top = 180
        padding_right = 200
        padding_bottom = 180
        padding_left = 200
    elif canvas_size_name == 'Ccilu-L/S':
        canvas_size = (1080, 1080)
        padding_top = 180
        padding_right = 200
        padding_bottom = 180
        padding_left = 200
    elif canvas_size_name == 'Columbia-L/S':
        canvas_size = (1080, 1080)
        padding_top = 180
        padding_right = 200
        padding_bottom = 180
        padding_left = 200
    elif canvas_size_name == 'DC-L/S':
        canvas_size = (1080, 1080)
        padding_top = 180
        padding_right = 200
        padding_bottom = 180
        padding_left = 200
    elif canvas_size_name == 'Delsey-L/S':
        canvas_size = (1080, 1080)
        padding_top = 180
        padding_right = 72
        padding_bottom = 180
        padding_left = 72
    elif canvas_size_name == 'Drmartens-L/S':
        canvas_size = (1080, 1080)
        padding_top = 200
        padding_right = 200
        padding_bottom = 180
        padding_left = 200
    elif canvas_size_name == 'Fitflop-L/S':
        canvas_size = (1080, 1080)
        padding_top = 200
        padding_right = 200
        padding_bottom = 180
        padding_left = 200
    elif canvas_size_name == 'Fjallraven-L/S':
        canvas_size = (1080, 1080)
        padding_top = 200
        padding_right = 200
        padding_bottom = 180
        padding_left = 200
    elif canvas_size_name == 'Fox-L/S':
        canvas_size = (1080, 1080)
        padding_top = 180
        padding_right = 200
        padding_bottom = 180
        padding_left = 200
    elif canvas_size_name == 'FreedomMoses-L/S':
        canvas_size = (1080, 1080)
        padding_top = 180
        padding_right = 72
        padding_bottom = 180
        padding_left = 72
    elif canvas_size_name == 'Gregory-L/S':
        canvas_size = (1080, 1080)
        padding_top = 180
        padding_right = 200
        padding_bottom = 180
        padding_left = 200
    elif canvas_size_name == 'Hedgren-L/S':
        canvas_size = (1080, 1080)
        padding_top = 180
        padding_right = 200
        padding_bottom = 180
        padding_left = 200
    elif canvas_size_name == 'Hellolulu-L/S':
        canvas_size = (1080, 1080)
        padding_top = 200
        padding_right = 200
        padding_bottom = 180
        padding_left = 200
    elif canvas_size_name == 'Herschel-L/S':
        canvas_size = (1080, 1080)
        padding_top = 200
        padding_right = 200
        padding_bottom = 180
        padding_left = 200
    elif canvas_size_name == 'Heydude-L/S':
        canvas_size = (1080, 1080)
        padding_top = 180
        padding_right = 200
        padding_bottom = 180
        padding_left = 200
    elif canvas_size_name == 'Hydroflask-L/S':
        canvas_size = (1080, 1080)
        padding_top = 180
        padding_right = 315
        padding_bottom = 180
        padding_left = 315
    elif canvas_size_name == 'Ipanema-L/S':
        canvas_size = (1080, 1080)
        padding_top = 180
        padding_right = 72
        padding_bottom = 180
        padding_left = 72
    elif canvas_size_name == 'Jansport-L/S':
        canvas_size = (1080, 1080)
        padding_top = 180
        padding_right = 200
        padding_bottom = 180
        padding_left = 200
    elif canvas_size_name == 'Keen-L/S':
        canvas_size = (1080, 1080)
        padding_top = 180
        padding_right = 200
        padding_bottom = 180
        padding_left = 200
    elif canvas_size_name == 'Livall-L/S':
        canvas_size = (1080, 1080)
        padding_top = 180
        padding_right = 200
        padding_bottom = 180
        padding_left = 200
    elif canvas_size_name == 'Lixil-American Standard-L/S':
        canvas_size = (1080, 1080)
        padding_top = 180
        padding_right = 200
        padding_bottom = 180
        padding_left = 200
    elif canvas_size_name == 'Lojel-L/S':
        canvas_size = (1080, 1080)
        padding_top = 180
        padding_right = 200
        padding_bottom = 180
        padding_left = 200
    elif canvas_size_name == 'Moleskine-L/S':
        canvas_size = (1080, 1080)
        padding_top = 180
        padding_right = 200
        padding_bottom = 180
        padding_left = 200
    elif canvas_size_name == 'Native-L/S':
        canvas_size = (1080, 1080)
        padding_top = 180
        padding_right = 200
        padding_bottom = 180
        padding_left = 200
    elif canvas_size_name == 'Osprey-L/S':
        canvas_size = (1080, 1080)
        padding_top = 200
        padding_right = 200
        padding_bottom = 180
        padding_left = 200
    elif canvas_size_name == 'Parkland-L/S':
        canvas_size = (1080, 1080)
        padding_top = 180
        padding_right = 200
        padding_bottom = 180
        padding_left = 200
    elif canvas_size_name == 'Poler-L/S':
        canvas_size = (1080, 1080)
        padding_top = 180
        padding_right = 200
        padding_bottom = 180
        padding_left = 200
    elif canvas_size_name == 'Quiksilver-L/S':
        canvas_size = (1080, 1080)
        padding_top = 200
        padding_right = 200
        padding_bottom = 180
        padding_left = 200
    elif canvas_size_name == 'Reef-L/S':
        canvas_size = (1080, 1080)
        padding_top = 180
        padding_right = 200
        padding_bottom = 180
        padding_left = 200
    elif canvas_size_name == 'RTR-L/S':
        canvas_size = (1080, 1080)
        padding_top = 180
        padding_right = 200
        padding_bottom = 180
        padding_left = 200
    elif canvas_size_name == 'Rider-L/S':
        canvas_size = (1080, 1080)
        padding_top = 180
        padding_right = 72
        padding_bottom = 180
        padding_left = 72
    elif canvas_size_name == 'Rockport-L/S':
        canvas_size = (1080, 1080)
        padding_top = 180
        padding_right = 200
        padding_bottom = 180
        padding_left = 200
    elif canvas_size_name == 'Roxy-L/S':
        canvas_size = (1080, 1080)
        padding_top = 200
        padding_right = 200
        padding_bottom = 180
        padding_left = 200
    elif canvas_size_name == 'RVCA-L/S':
        canvas_size = (1080, 1080)
        padding_top = 180
        padding_right = 200
        padding_bottom = 180
        padding_left = 200
    elif canvas_size_name == 'Sakroots-L/S':
        canvas_size = (1080, 1080)
        padding_top = 180
        padding_right = 200
        padding_bottom = 180
        padding_left = 200
    elif canvas_size_name == 'Salomon-L/S':
        canvas_size = (1080, 1080)
        padding_top = 200
        padding_right = 200
        padding_bottom = 180
        padding_left = 200
    elif canvas_size_name == 'SeaToSummit-L/S':
        canvas_size = (1080, 1080)
        padding_top = 180
        padding_right = 200
        padding_bottom = 180
        padding_left = 200
    elif canvas_size_name == 'Sledgers-L/S':
        canvas_size = (1080, 1080)
        padding_top = 180
        padding_right = 200
        padding_bottom = 180
        padding_left = 200
    elif canvas_size_name == 'Theragun-L/S':
        canvas_size = (1080, 1080)
        padding_top = 180
        padding_right = 200
        padding_bottom = 180
        padding_left = 200
    elif canvas_size_name == 'Timbuk2-L/S':
        canvas_size = (1080, 1080)
        padding_top = 180
        padding_right = 200
        padding_bottom = 180
        padding_left = 200
    elif canvas_size_name == 'TNF-L/S':
        canvas_size = (1080, 1080)
        padding_top = 180
        padding_right = 200
        padding_bottom = 180
        padding_left = 200
    elif canvas_size_name == 'TomsSG-L/S':
        canvas_size = (1080, 1080)
        padding_top = 180
        padding_right = 200
        padding_bottom = 180
        padding_left = 200
    elif canvas_size_name == 'TopoDesigns-L/S':
        canvas_size = (1080, 1080)
        padding_top = 200
        padding_right = 200
        padding_bottom = 180
        padding_left = 200
    elif canvas_size_name == 'Travelon-L/S':
        canvas_size = (1080, 1080)
        padding_top = 180
        padding_right = 200
        padding_bottom = 180
        padding_left = 200
    elif canvas_size_name == 'Tretorn-L/S':
        canvas_size = (1080, 1080)
        padding_top = 180
        padding_right = 200
        padding_bottom = 180
        padding_left = 200
    elif canvas_size_name == 'WorldTraveller-L/S':
        canvas_size = (1080, 1080)
        padding_top = 201
        padding_right = 72
        padding_bottom = 180
        padding_left = 72
    elif canvas_size_name == 'Zaxy-L/S':
        canvas_size = (1080, 1080)
        padding_top = 200
        padding_right = 72
        padding_bottom = 180
        padding_left = 72
    elif canvas_size_name == 'Allbirds-Zalora':
        canvas_size = (762, 1100)
        padding_top = 51
        padding_right = 51
        padding_bottom = 202
        padding_left = 51
    elif canvas_size_name == 'Backjoy-Zalora':
        canvas_size = (762, 1100)
        padding_top = 51
        padding_right = 51
        padding_bottom = 201
        padding_left = 51
    elif canvas_size_name == 'Columbia-Zalora':
        canvas_size = (762, 1100)
        padding_top = 51
        padding_right = 51
        padding_bottom = 202
        padding_left = 51
    elif canvas_size_name == 'DC-Zalora':
        canvas_size = (762, 1100)
        padding_top = 51
        padding_right = 51
        padding_bottom = 202
        padding_left = 51
    elif canvas_size_name == 'Delsey-Zalora':
        canvas_size = (762, 1100)
        padding_top = 51
        padding_right = 51
        padding_bottom = 202
        padding_left = 51
    elif canvas_size_name == 'Drmartens-Zalora':
        canvas_size = (762, 1100)
        padding_top = 51
        padding_right = 51
        padding_bottom = 202
        padding_left = 51
    elif canvas_size_name == 'FreedomMoses-Zalora':
        canvas_size = (762, 1100)
        padding_top = 51
        padding_right = 51
        padding_bottom = 201
        padding_left = 51
    elif canvas_size_name == 'Hedgren-Zalora':
        canvas_size = (762, 1100)
        padding_top = 51
        padding_right = 51
        padding_bottom = 202
        padding_left = 51
    elif canvas_size_name == 'Herschel-Zalora':
        canvas_size = (762, 1100)
        padding_top = 51
        padding_right = 51
        padding_bottom = 202
        padding_left = 51
    elif canvas_size_name == 'Hydroflask-Zalora':
        canvas_size = (762, 1100)
        padding_top = 51
        padding_right = 51
        padding_bottom = 202
        padding_left = 51
    elif canvas_size_name == 'Ipanema-Zalora':
        canvas_size = (762, 1100)
        padding_top = 51
        padding_right = 51
        padding_bottom = 202
        padding_left = 51
    elif canvas_size_name == 'Jansport-Zalora':
        canvas_size = (762, 1100)
        padding_top = 51
        padding_right = 51
        padding_bottom = 202
        padding_left = 51
    elif canvas_size_name == 'LCS (LeCoqSportif)-Zalora':
        canvas_size = (762, 1100)
        padding_top = 51
        padding_right = 51
        padding_bottom = 202
        padding_left = 51
    elif canvas_size_name == 'Rider-Zalora':
        canvas_size = (762, 1100)
        padding_top = 51
        padding_right = 51
        padding_bottom = 202
        padding_left = 51
    elif canvas_size_name == 'Roxy-Zalora':
        canvas_size = (762, 1100)
        padding_top = 51
        padding_right = 51
        padding_bottom = 202
        padding_left = 51
    elif canvas_size_name == 'Sakroots-Zalora':
        canvas_size = (762, 1100)
        padding_top = 51
        padding_right = 51
        padding_bottom = 202
        padding_left = 51
    elif canvas_size_name == 'Salomon-Zalora':
        canvas_size = (762, 1100)
        padding_top = 400 #51
        padding_right = 51
        padding_bottom = 202
        padding_left = 51
    elif canvas_size_name == 'SeaToSummit-Zalora':
        canvas_size = (762, 1100)
        padding_top = 51
        padding_right = 51
        padding_bottom = 202
        padding_left = 51
    elif canvas_size_name == 'TNF-Zalora':
        canvas_size = (762, 1100)
        padding_top = 51
        padding_right = 51
        padding_bottom = 202
        padding_left = 51
    elif canvas_size_name == 'Travelon-Zalora':
        canvas_size = (762, 1100)
        padding_top = 51
        padding_right = 51
        padding_bottom = 202
        padding_left = 51
    elif canvas_size_name == 'Tretorn-Zalora':
        canvas_size = (762, 1100)
        padding_top = 51
        padding_right = 51
        padding_bottom = 202
        padding_left = 51
    elif canvas_size_name == 'Zaxy-Zalora':
        canvas_size = (762, 1100)
        padding_top = 51
        padding_right = 51
        padding_bottom = 202
        padding_left = 51
    elif canvas_size_name == 'Aetrex-DOTCOM':
        canvas_size = (2000, 2000)
        padding_top = 333
        padding_right = 133
        padding_bottom = 333
        padding_left = 133
    elif canvas_size_name == 'Allbirds-DOTCOM':
        canvas_size = (1124, 1285)
        padding_top = 175
        padding_right = 132
        padding_bottom = 80
        padding_left = 132
    elif canvas_size_name == 'Backjoy-DOTCOM':
        canvas_size = (1200, 1200)
        padding_top = 217
        padding_right = 205
        padding_bottom = 224
        padding_left = 205
    elif canvas_size_name == 'Bratpack-DOTCOM':
        canvas_size = (900, 1200)
        padding_top = 72
        padding_right = 66
        padding_bottom = 63
        padding_left = 66
    elif canvas_size_name == 'Bratpack Hydroflask-DOTCOM':
        canvas_size = (900, 1200)
        padding_top = 72
        padding_right = 201
        padding_bottom = 64
        padding_left = 201
    elif canvas_size_name == 'Bratpack Parkland-DOTCOM':
        canvas_size = (900, 1200)
        padding_top = 72
        padding_right = 66
        padding_bottom = 63
        padding_left = 66   
    else:
        canvas_size = (1080, 1080)
        padding_top = 100
        padding_right = 100
        padding_bottom = 100
        padding_left = 100

    # Auto-detect zero padding if enabled (skip for none_with_padding as it handles padding differently)
    if auto_detect_padding and bg_method != 'none_with_padding':
        auto_detected_padding = auto_detect_zero_padding(working_image_path, bg_method)
        
        if auto_detected_padding:
            if zero_padding:
                # Merge manual and auto-detected padding
                combined_padding = list(set(zero_padding + auto_detected_padding))
                zero_padding = combined_padding
            else:
                # Use only auto-detected padding
                zero_padding = auto_detected_padding
    
    # Auto Snap Detection - NEW FEATURE (skip for none_with_padding as it uses custom positioning)
    if auto_snap and bg_method != 'none_with_padding':
        print(f"\n🤖 AUTO SNAP DETECTION ACTIVATED for {filename}")
        
        # Run quick test to verify system is working (only on first file)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                print("🧪 Running quick system test...")
                test_result = test_auto_snap_detection()
                if test_result:
                    print(f"✅ System test passed: {test_result}")
                else:
                    print("⚠️  System test returned no results")
            except Exception as test_e:
                print(f"⚠️  System test failed: {test_e}")
        
        auto_detected_snaps = auto_detect_zero_padding(working_image_path, bg_method)
        
        if auto_detected_snaps:
            print(f"🎯 Auto-detected snaps for {filename}: {auto_detected_snaps}")
            
            # Apply auto-detected snaps
            if "top" in auto_detected_snaps:
                snap_to_top = True
                padding_top = 0
                print(f"   ✅ Applied SNAP TO TOP for {filename}")
            
            if "bottom" in auto_detected_snaps:
                snap_to_bottom = True
                padding_bottom = 0
                print(f"   ✅ Applied SNAP TO BOTTOM for {filename}")
            
            if "left" in auto_detected_snaps:
                snap_to_left = True
                padding_left = 0
                print(f"   ✅ Applied SNAP TO LEFT for {filename}")
            
            if "right" in auto_detected_snaps:
                snap_to_right = True
                padding_right = 0
                print(f"   ✅ Applied SNAP TO RIGHT for {filename}")
            
            # DO NOT set zero_padding for auto snap - let it use manual snap logic through position_logic_old
            # This ensures consistent behavior between auto and manual snap
            print(f"   🎯 Auto snap will use manual snap logic (position_logic_old) instead of zero padding logic")
                
            print(f"   📋 Final padding after auto-snap: top={padding_top}, bottom={padding_bottom}, left={padding_left}, right={padding_right}")
            print(f"   📋 Snap flags: top={snap_to_top}, bottom={snap_to_bottom}, left={snap_to_left}, right={snap_to_right}")
        else:
            print(f"   ❌ No auto-snaps detected for {filename} - using default padding")
            print(f"   📋 Image info: {os.path.getsize(working_image_path)} bytes, exists: {os.path.exists(working_image_path)}")
            
            # Add additional debug info
            try:
                with Image.open(working_image_path) as debug_img:
                    print(f"   📋 Image mode: {debug_img.mode}, size: {debug_img.size}")
            except Exception as debug_e:
                print(f"   ⚠️  Could not open image for debug: {debug_e}")
    
    # Log info about zero padding
    if zero_padding:
        print(f"File {filename}: Applying zero padding on sides: {zero_padding}")
    else:
        print(f"File {filename}: No zero padding specified")

    # ========== QWEN CLASSIFICATION LOGIC ==========
    # Perform classification if enabled and sheet data is provided
    classification_result = None
    if use_qwen and sheet_data is not None:
        try:
            print(f"Starting Qwen classification for {filename}")
            
            # Extract unique categories from sheet data - use 'Classification' column
            unique_items = []
            if 'Classification' in sheet_data.columns:
                unique_items = sheet_data['Classification'].str.strip().str.lower().unique().tolist()
            elif 'object' in sheet_data.columns:
                # Fallback to 'object' column if 'Classification' not found
                unique_items = sheet_data['object'].dropna().unique().tolist()
            
            if unique_items:
                print(f"Unique items for classification of {filename}: {unique_items}")
                classification_result = classify_image(working_image_path, unique_items)
                if classification_result:
                    classification = classification_result.strip().lower()
                    print(f"Final classification for {filename}: '{classification}'")
                    
                    # Check for person detection - apply snap_to_bottom logic
                    if any(term in classification.lower() for term in ["human", "person", "model"]):
                        print(f"Person detected, setting bottom padding to 0 for {filename}")
                        snap_to_bottom = True
                    
                    # Look up settings from sheet data based on classification
                    if 'Classification' in sheet_data.columns:
                        matching_rows = sheet_data[sheet_data['Classification'].str.strip().str.lower() == classification]
                    else:
                        matching_rows = sheet_data[sheet_data['object'].str.lower() == classification]
                        
                    if not matching_rows.empty:
                        # Use first matching row for settings
                        row = matching_rows.iloc[0]
                        
                        # Override padding if specified in sheet
                        if 'padding_top' in row and pd.notna(row['padding_top']):
                            padding_top = int(row['padding_top'])
                            print(f"Using padding_top from sheet: {padding_top}")
                        if 'padding_bottom' in row and pd.notna(row['padding_bottom']):
                            padding_bottom = int(row['padding_bottom'])
                            print(f"Using padding_bottom from sheet: {padding_bottom}")
                        if 'padding_left' in row and pd.notna(row['padding_left']):
                            padding_left = int(row['padding_left'])
                            print(f"Using padding_left from sheet: {padding_left}")
                        if 'padding_right' in row and pd.notna(row['padding_right']):
                            padding_right = int(row['padding_right'])
                            print(f"Using padding_right from sheet: {padding_right}")
                        
                        # Override canvas size if specified in sheet
                        if 'canvas_size_name' in row and pd.notna(row['canvas_size_name']):
                            canvas_size_name = row['canvas_size_name']
                            print(f"Using canvas size from sheet: {canvas_size_name}")
                        
                        # Override other settings if available
                        if 'bg_method' in row and pd.notna(row['bg_method']):
                            bg_method = row['bg_method']
                            print(f"Using bg_method from sheet: {bg_method}")
                            
                        if 'output_format' in row and pd.notna(row['output_format']):
                            output_format = row['output_format']
                            print(f"Using output_format from sheet: {output_format}")
                            
                        if 'bg_choice' in row and pd.notna(row['bg_choice']):
                            bg_choice = row['bg_choice']
                            print(f"Using bg_choice from sheet: {bg_choice}")
                            
                        if 'custom_color' in row and pd.notna(row['custom_color']):
                            custom_color = row['custom_color']
                            print(f"Using custom_color from sheet: {custom_color}")
                            
                        print(f"Padding overridden for {filename}: top={padding_top}, bottom={padding_bottom}, left={padding_left}, right={padding_right}\n")
                    else:
                        print(f"No matching sheet data found for classification: {classification_result}")
                else:
                    print(f"Classification failed for {filename}")
            else:
                print(f"No unique items found in sheet data for classification")
        except Exception as e:
            print(f"Error during Qwen classification for {filename}: {e}")
            classification_result = None
    else:
        print(f"Qwen classification not used or no sheet data for {filename}. Using default padding.")

    # Determine snap settings from padding and parameters
    # Convert zero_padding to snap_to for backward compatibility
    if zero_padding:
        if "top" in zero_padding:
            snap_to_top = True
            padding_top = 0
        if "bottom" in zero_padding:
            snap_to_bottom = True
            padding_bottom = 0
        if "left" in zero_padding:
            snap_to_left = True
            padding_left = 0
        if "right" in zero_padding:
            snap_to_right = True
            padding_right = 0
        if "all" in zero_padding:
            snap_to_top = snap_to_bottom = snap_to_left = snap_to_right = True
            padding_top = padding_bottom = padding_left = padding_right = 0

    # Also set snap from padding values directly
    if not snap_to_top and padding_top == 0:
        snap_to_top = True
    if not snap_to_bottom and padding_bottom == 0:
        snap_to_bottom = True
    if not snap_to_left and padding_left == 0:
        snap_to_left = True
    if not snap_to_right and padding_right == 0:
        snap_to_right = True

    # Store padding used for classification data
    padding_used = {
        "top": int(padding_top),
        "bottom": int(padding_bottom),
        "left": int(padding_left),
        "right": int(padding_right)
    }

    if stop_event.is_set():
        print("Stop event triggered, no processing.")
        return None, None

    print(f"Processing image: {filename}")
    
    # Get original image with color profile before any processing
    original_img_with_profile = Image.open(working_image_path)  # Use working_image_path
    original_mode = original_img_with_profile.mode
    icc_profile = original_img_with_profile.info.get('icc_profile', None)
    original_size = original_img_with_profile.size
    
    # Store original CMYK profile for potential conversion back
    original_cmyk_profile = None
    if original_mode == "CMYK" and icc_profile:
        original_cmyk_profile = icc_profile
    
    print(f"Original image mode: {original_mode}")
    print(f"Original image size: {original_size}")
    
    # Add logging for cropping
    if was_cropped:
        print(f"Reference text detected - image was cropped by 36px from bottom")
    
    # Check if image has an embedded color profile
    if icc_profile:
        print(f"Image has embedded ICC profile: {len(icc_profile)} bytes")
    else:
        print("No ICC profile found in image")
    
    # CMYK handling - convert to RGB for processing but preserve profile
    # Use the consolidated helper function for consistent conversion
    if original_mode == "CMYK":
        rgb_image, new_profile = convert_image_to_rgb(original_img_with_profile)
        original_img_with_profile = rgb_image
        
        # If we got a new profile, use it; otherwise keep the original
        if new_profile:
            icc_profile = new_profile
    
    # Debug original image color at center
    if isinstance(original_img_with_profile, Image.Image):
        width, height = original_img_with_profile.size
        center_x, center_y = width//2, height//2
        debug_color_values(original_img_with_profile, center_x, center_y, "Original image color")
    
    # Always use RGB/RGBA mode for consistent processing
    original_img = original_img_with_profile.convert("RGBA")

    # Special handling for "none" method with zero padding
    if zero_padding and bg_method == 'none':
        # For 'none' method, we'll handle positioning directly
        mask = original_img.copy()
        
        # Apply zero padding directly without background removal
        # Get bounding box of entire image (we don't remove background for 'none')
        bbox = (0, 0, mask.width, mask.height)
        
        # Crop to content (which is the entire image for 'none')
        cropped_content = mask
        
        # SIMPLIFIED ZERO PADDING LOGIC
        # Calculate sizes and positions based ONLY on selected padding options
        aspect_ratio = cropped_content.width / cropped_content.height
        
        # Determine which dimensions are constrained by zero padding
        width_constrained = "left" in zero_padding or "right" in zero_padding or "all" in zero_padding
        height_constrained = "top" in zero_padding or "bottom" in zero_padding or "all" in zero_padding
        
        if "all" in zero_padding:
            # Fill entire canvas
            new_width = canvas_size[0]
            new_height = canvas_size[1]
            x, y = 0, 0
        elif width_constrained and height_constrained:
            # Both dimensions constrained - fill entire canvas
            new_width = canvas_size[0]
            new_height = canvas_size[1]
            x, y = 0, 0
        elif width_constrained:
            # Only width constrained - fit to width, maintain aspect ratio
            new_width = canvas_size[0]
            new_height = int(new_width / aspect_ratio)
            
            # Position horizontally (always 0 since width fills canvas)
            x = 0
            
            # Position vertically based on constraints
            if "top" in zero_padding:
                y = 0
            elif "bottom" in zero_padding:
                y = canvas_size[1] - new_height
            else:
                # Center vertically if no vertical constraint
                y = (canvas_size[1] - new_height) // 2
                
        elif height_constrained:
            # Only height constrained - fit to height, maintain aspect ratio
            new_height = canvas_size[1]
            new_width = int(new_height * aspect_ratio)
            
            # Position vertically (always 0 since height fills canvas)
            y = 0
            
            # Position horizontally based on constraints
            if "left" in zero_padding:
                x = 0
            elif "right" in zero_padding:
                x = canvas_size[0] - new_width
            else:
                # Center horizontally if no horizontal constraint
                x = (canvas_size[0] - new_width) // 2
                
        else:
            # No constraints - fit within canvas maintaining aspect ratio
            canvas_aspect = canvas_size[0] / canvas_size[1]
            
            if aspect_ratio > canvas_aspect:
                # Content is wider - fit to width
                new_width = canvas_size[0]
                new_height = int(new_width / aspect_ratio)
            else:
                # Content is taller - fit to height
                new_height = canvas_size[1]
                new_width = int(new_height * aspect_ratio)
            
            # Center both dimensions
            x = (canvas_size[0] - new_width) // 2
            y = (canvas_size[1] - new_height) // 2
        
        # Ensure dimensions don't exceed canvas
        if new_width > canvas_size[0]:
            new_width = canvas_size[0]
        if new_height > canvas_size[1]:
            new_height = canvas_size[1]
        
        # Resize the content
        cropped_content = cropped_content.resize((new_width, new_height), Image.LANCZOS)
        
        # Create a canvas with the proper background
        if bg_choice == 'white':
            canvas = Image.new("RGBA", canvas_size, "WHITE")
        elif bg_choice == 'custom':
            canvas = Image.new("RGBA", canvas_size, custom_color)
        elif bg_choice == 'dominant':
            dom_col = get_dominant_color(original_img)
            canvas = Image.new("RGBA", canvas_size, dom_col)
        else:
            canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
        
        # Paste the content onto the canvas
        canvas.paste(cropped_content, (x, y))
        
        # Handle rotation and flip
        if flip:
            canvas = flip_image(canvas)
        
        if rotation != "None" and (rotation == "180 Degrees" or direction != "None"):
            if rotation == "90 Degrees":
                angle = 90 if direction == "Clockwise" else -90
            elif rotation == "180 Degrees":
                angle = 180
            else:
                angle = 0
            canvas = canvas.rotate(angle, expand=True, resample=Image.BICUBIC)
            
            # If expanded, crop back to canvas size
            if canvas.size != canvas_size:
                # Center crop
                left = (canvas.width - canvas_size[0]) // 2
                top = (canvas.height - canvas_size[1]) // 2
                canvas = canvas.crop((left, top, left + canvas_size[0], top + canvas_size[1]))
        
        # Add watermark and twibbon if needed
        if (base_no_ext.endswith("_01") or base_no_ext.endswith("_1") or base_no_ext.endswith("_001")) and watermark_path:
            try:
                w_img = Image.open(watermark_path).convert("RGBA")
                canvas.paste(w_img, (0, 0), w_img)
            except Exception:
                pass
        
        if twibbon_path:
            try:
                twb = Image.open(twibbon_path).convert("RGBA")
                canvas.paste(twb, (0, 0), twb)
            except Exception:
                pass
        
        # Save the result - Auto-detect CMYK and force JPG output for CMYK inputs
        if original_mode == "CMYK":
            out_ext = "jpg"
            force_cmyk_output = True
        else:
            out_ext = "jpg" if output_format == "JPG" else "png"
            force_cmyk_output = False
            
        out_filename = f"{os.path.splitext(filename)[0]}.{out_ext}"
        out_path = os.path.join(output_folder, out_filename)
        
        if output_format == "JPG" or force_cmyk_output:
            if force_cmyk_output and original_mode == "CMYK":
                # Auto CMYK conversion for CMYK inputs
                print(f"Auto-converting final image back to CMYK (original mode: {original_mode})")
                
                # First convert canvas to RGB if it's not already
                rgb_canvas = canvas.convert("RGB")
                
                # Convert RGB back to CMYK using original profile if available
                cmyk_canvas = convert_rgb_to_cmyk_with_profile(rgb_canvas, original_cmyk_profile)
                print(f"Auto-converted to CMYK mode: {cmyk_canvas.mode}")
                
                try:
                    # Save as CMYK JPEG
                    if original_cmyk_profile:
                        cmyk_canvas.save(out_path, "JPEG", quality=100, optimize=True, icc_profile=original_cmyk_profile)
                        print(f"Auto-saved CMYK JPG with original ICC profile")
                    else:
                        cmyk_canvas.save(out_path, "JPEG", quality=100, optimize=True)
                        print(f"Auto-saved CMYK JPG without ICC profile")
                except Exception as e:
                    print(f"Error saving CMYK JPG: {e}, falling back to RGB")
                    rgb_canvas = canvas.convert("RGB")
                    rgb_canvas.save(out_path, "JPEG", quality=100, optimize=True)
            else:
                # Regular RGB JPG
                rgb_canvas = canvas.convert("RGB")
                if icc_profile:
                    try:
                        rgb_canvas.save(out_path, "JPEG", quality=100, optimize=True, icc_profile=icc_profile)
                    except Exception as e:
                        print(f"Error saving with ICC profile: {e}, falling back to standard save")
                        rgb_canvas.save(out_path, "JPEG", quality=100, optimize=True)
                else:
                    rgb_canvas.save(out_path, "JPEG", quality=100, optimize=True)
        else:
            # PNG output
            if icc_profile:
                try:
                    canvas.save(out_path, "PNG", optimize=True, icc_profile=icc_profile)
                except Exception as e:
                    print(f"Error saving PNG with ICC profile: {e}, falling back to standard save")
                    canvas.save(out_path, "PNG", optimize=True)
            else:
                canvas.save(out_path, "PNG", optimize=True)
        
        print(f"Processed image with 'none' method and zero padding => {out_path}")
        
        # Clean up temporary cropped file if it was created
        if was_cropped and os.path.exists(working_image_path):
            try:
                os.remove(working_image_path)
                print(f"Cleaned up temporary cropped file: {working_image_path}")
            except Exception as e:
                print(f"Error cleaning up temporary file {working_image_path}: {e}")
        
        return [(out_path, image_path)], [{"action": "zero_padding_with_none", "sides": zero_padding}], classification_result
    
    # Continue with standard background removal methods
    # BG removal
    if bg_method == 'rembg':
        mask = remove_background_rembg(working_image_path)
    elif bg_method == 'bria':
        mask = remove_background_bria(working_image_path)
    elif bg_method == 'photoroom':
        mask = remove_background_photoroom(working_image_path)
    elif bg_method == 'birefnet':
        mask = remove_background_birefnet(working_image_path)
        if not mask:
            # Clean up temporary files if created
            if was_cropped and os.path.exists(working_image_path):
                os.remove(working_image_path)
            if was_arw_converted and arw_temp_path and os.path.exists(arw_temp_path):
                os.remove(arw_temp_path)
            return None, None
    elif bg_method == 'birefnet_2':
        mask = remove_background_birefnet_2(working_image_path)
        if not mask:
            # Clean up temporary files if created
            if was_cropped and os.path.exists(working_image_path):
                os.remove(working_image_path)
            if was_arw_converted and arw_temp_path and os.path.exists(arw_temp_path):
                os.remove(arw_temp_path)
            return None, None
    elif bg_method == 'birefnet_hr':
        mask = remove_background_birefnet_hr(working_image_path)
        if not mask:
            # Clean up temporary files if created
            if was_cropped and os.path.exists(working_image_path):
                os.remove(working_image_path)
            if was_arw_converted and arw_temp_path and os.path.exists(arw_temp_path):
                os.remove(arw_temp_path)
            return None, None
    elif bg_method == 'none':
        # For 'none' method, we need to handle CMYK images specially
        mask = remove_background_none(working_image_path)
        # Preserve the ICC profile for the 'none' method
        if icc_profile and original_mode == "CMYK":
            # Create a new image that has both the original colors and the ICC profile
            mask.info['icc_profile'] = icc_profile
    elif bg_method == 'none_with_padding':
        # For 'none_with_padding' method, similar to 'none' but with custom positioning
        mask = remove_background_none(working_image_path)
        # Preserve the ICC profile for the 'none_with_padding' method
        if icc_profile and original_mode == "CMYK":
            # Create a new image that has both the original colors and the ICC profile
            mask.info['icc_profile'] = icc_profile
    
    temp_image_path = os.path.join(output_folder, f"temp_{filename}")
    
    # Debug processed image color at safe coordinates
    if hasattr(mask, 'width') and hasattr(mask, 'height'):
        safe_x = min(center_x, mask.width - 1)
        safe_y = min(center_y, mask.height - 1)
        debug_color_values(mask, safe_x, safe_y, f"Temp image color ({bg_method})")
    
    # Save temporary file with color profile if available
    if icc_profile:
        try:
            mask.save(temp_image_path, format='PNG', icc_profile=icc_profile)
        except Exception as e:
            print(f"Error saving with ICC profile: {e}, falling back to standard save")
            mask.save(temp_image_path, format='PNG')
    else:
        mask.save(temp_image_path, format='PNG')

    # Add support for zero padding if specified, but NOT when auto_snap is active or when using none_with_padding
    # When auto_snap is active, we want to use manual snap logic through position_logic_old
    # When using none_with_padding, we use the custom position_with_padding function instead
    if zero_padding and hasattr(mask, 'mode') and mask.mode == 'RGBA' and not auto_snap and bg_method != 'none_with_padding':
        print(f"File {filename}: Applying zero padding with background removal: {zero_padding}")
        
        # Get bounding box of actual content
        bbox = get_bounding_box_with_threshold(mask, threshold=10)
        if not bbox:
            bbox = mask.getbbox()  # Fallback to standard bounding box
        
        if bbox:
            print(f"Found bounding box: {bbox}")
            # Crop to content
            cropped_content = mask.crop(bbox)
            
            # SIMPLIFIED ZERO PADDING LOGIC (same as 'none' method)
            aspect_ratio = cropped_content.width / cropped_content.height
            
            print(f"Original content size: {cropped_content.width}x{cropped_content.height}")
            print(f"Canvas size: {canvas_size[0]}x{canvas_size[1]}")
            print(f"Aspect ratio: {aspect_ratio:.3f}")
            
            # Determine which dimensions are constrained by zero padding
            width_constrained = "left" in zero_padding or "right" in zero_padding or "all" in zero_padding
            height_constrained = "top" in zero_padding or "bottom" in zero_padding or "all" in zero_padding
            
            print(f"Width constrained: {width_constrained}, Height constrained: {height_constrained}")
            
            if "all" in zero_padding:
                # Fill entire canvas
                print("Filling entire canvas (all sides)")
                new_width = canvas_size[0]
                new_height = canvas_size[1]
                x, y = 0, 0
            elif width_constrained and height_constrained:
                # Both dimensions constrained - fill entire canvas
                print("Both dimensions constrained - filling entire canvas")
                new_width = canvas_size[0]
                new_height = canvas_size[1]
                x, y = 0, 0
            elif width_constrained:
                # Only width constrained - fit to width, maintain aspect ratio
                print("Only width constrained - fitting to width")
                new_width = canvas_size[0]
                new_height = int(new_width / aspect_ratio)
                
                # Position horizontally (always 0 since width fills canvas)
                x = 0
                
                # Position vertically based on constraints
                if "top" in zero_padding:
                    y = 0
                elif "bottom" in zero_padding:
                    y = canvas_size[1] - new_height
                else:
                    # Center vertically if no vertical constraint
                    y = (canvas_size[1] - new_height) // 2
                    
            elif height_constrained:
                # Only height constrained - fit to height, maintain aspect ratio
                print("Only height constrained - fitting to height")
                new_height = canvas_size[1]
                new_width = int(new_height * aspect_ratio)
                
                # Position vertically (always 0 since height fills canvas)
                y = 0
                
                # Position horizontally based on constraints
                if "left" in zero_padding:
                    x = 0
                elif "right" in zero_padding:
                    x = canvas_size[0] - new_width
                else:
                    # Center horizontally if no horizontal constraint
                    x = (canvas_size[0] - new_width) // 2
                    
            else:
                # No constraints - fit within canvas maintaining aspect ratio
                print("No constraints - fitting within canvas")
                canvas_aspect = canvas_size[0] / canvas_size[1]
                
                if aspect_ratio > canvas_aspect:
                    # Content is wider - fit to width
                    new_width = canvas_size[0]
                    new_height = int(new_width / aspect_ratio)
                else:
                    # Content is taller - fit to height
                    new_height = canvas_size[1]
                    new_width = int(new_height * aspect_ratio)
                
                # Center both dimensions
                x = (canvas_size[0] - new_width) // 2
                y = (canvas_size[1] - new_height) // 2
            
            # Ensure dimensions don't exceed canvas
            if new_width > canvas_size[0]:
                print(f"Width {new_width} exceeds canvas {canvas_size[0]} - cropping")
                new_width = canvas_size[0]
            if new_height > canvas_size[1]:
                print(f"Height {new_height} exceeds canvas {canvas_size[1]} - cropping")
                new_height = canvas_size[1]
            
            # Resize to final dimensions
            cropped_content = cropped_content.resize((new_width, new_height), Image.LANCZOS)
            
            print(f"Final size: {new_width}x{new_height}")
            print(f"Final position: ({x}, {y})")

            # Create a new transparent canvas
            adjusted_mask = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
            
            # Paste content at calculated position
            adjusted_mask.paste(cropped_content, (x, y), cropped_content)
            
            # Replace original mask with the adjusted one
            mask = adjusted_mask
            
            print(f"Zero padding applied: position ({x}, {y}), size {cropped_content.width}x{cropped_content.height}")
            
            # Save the adjusted mask
            mask.save(temp_image_path, format='PNG')
            
            # Use the already positioned image and skip position_logic_old
            logs = [{"action": "zero_padding_applied", "sides": zero_padding, 
                     "position": f"({x}, {y})", "size": f"{cropped_content.width}x{cropped_content.height}"}]
            cropped_img = mask
            x, y = 0, 0  # Position is already in the mask
            
            # Skip to canvas creation - ZERO PADDING PROCESSING COMPLETE
            if bg_choice == 'white':
                canvas = Image.new("RGBA", canvas_size, "WHITE")
            elif bg_choice == 'custom':
                canvas = Image.new("RGBA", canvas_size, custom_color)
            elif bg_choice == 'dominant':
                dom_col = get_dominant_color(original_img)
                canvas = Image.new("RGBA", canvas_size, dom_col)
            else:
                canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))

            canvas.paste(cropped_img, (x, y), cropped_img)
            logs.append({"action": "paste", "x": x, "y": y})

            # Continue with rotation/flip and save logic...
            if flip:
                canvas = flip_image(canvas)
                logs.append({"action": "flip_horizontal"})

            if rotation != "None" and (rotation == "180 Degrees" or direction != "None"):
                if rotation == "90 Degrees":
                    angle = 90 if direction == "Clockwise" else -90
                elif rotation == "180 Degrees":
                    angle = 180
                else:
                    angle = 0
                rotated_subject = cropped_img.rotate(angle, expand=True)
                if bg_choice == 'white':
                    new_canvas = Image.new("RGBA", canvas_size, "WHITE")
                elif bg_choice == 'custom':
                    new_canvas = Image.new("RGBA", canvas_size, custom_color)
                elif bg_choice == 'dominant':
                    dom_col = get_dominant_color(original_img)
                    new_canvas = Image.new("RGBA", canvas_size, dom_col)
                else:
                    new_canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
                available_width = canvas_size[0] - padding_left - padding_right
                target_height = canvas_size[1] - padding_top - padding_bottom
                rs_w, rs_h = rotated_subject.size
                scale_factor = target_height / rs_h
                new_width_h = int(rs_w * scale_factor)
                if new_width_h > available_width:
                    scale_factor = available_width / rs_w
                    new_width = available_width
                    new_height = int(rs_h * scale_factor)
                else:
                    new_width = new_width_h
                    new_height = target_height
                rotated_subject = rotated_subject.resize((new_width, new_height), Image.LANCZOS)
                new_x = padding_left + (available_width - new_width) // 2
                new_y = padding_top + (target_height - new_height) // 2
                new_canvas.paste(rotated_subject, (new_x, new_y), rotated_subject)
                canvas = new_canvas
                logs.append({"action": "rotate_final", "rotation": rotation, "direction": direction})

            # Watermark and twibbon
            if (base_no_ext.endswith("_01") or base_no_ext.endswith("_1") or base_no_ext.endswith("_001")) and watermark_path:
                try:
                    w_img = Image.open(watermark_path).convert("RGBA")
                    canvas.paste(w_img, (0, 0), w_img)
                    logs.append({"action": "add_watermark"})
                except Exception as e:
                    print(f"Error adding watermark: {e}")

            if twibbon_path:
                try:
                    twb = Image.open(twibbon_path).convert("RGBA")
                    canvas.paste(twb, (0, 0), twb)
                    logs.append({"action": "twibbon"})
                except Exception as e:
                    print(f"Error adding twibbon: {e}")

            # Debug final canvas color
            safe_x = min(center_x, canvas.width - 1)
            safe_y = min(center_y, canvas.height - 1)
            debug_color_values(canvas, safe_x, safe_y, "Final image color with zero padding")

            # Save final image - Auto-detect CMYK and force JPG output for CMYK inputs
            if original_mode == "CMYK":
                print(f"Auto-detected CMYK input, forcing JPG output regardless of user selection")
                out_ext = "jpg"
                force_cmyk_output = True
            else:
                out_ext = "jpg" if output_format == "JPG" else "png"
                force_cmyk_output = False
                
            out_filename = f"{os.path.splitext(filename)[0]}.{out_ext}"
            out_path = os.path.join(output_folder, out_filename)

            if output_format == "JPG" or force_cmyk_output:
                if force_cmyk_output and original_mode == "CMYK":
                    # Auto CMYK conversion for CMYK inputs
                    print(f"Auto-converting final image back to CMYK (original mode: {original_mode})")
                    
                    # First convert canvas to RGB if it's not already
                    rgb_canvas = canvas.convert("RGB")
                    
                    # Convert RGB back to CMYK using original profile if available
                    cmyk_canvas = convert_rgb_to_cmyk_with_profile(rgb_canvas, original_cmyk_profile)
                    print(f"Auto-converted to CMYK mode: {cmyk_canvas.mode}")
                    
                    try:
                        # Save as CMYK JPEG
                        if original_cmyk_profile:
                            cmyk_canvas.save(out_path, "JPEG", quality=100, optimize=True, icc_profile=original_cmyk_profile)
                            print(f"Auto-saved CMYK JPG with original ICC profile")
                        else:
                            cmyk_canvas.save(out_path, "JPEG", quality=100, optimize=True)
                            print(f"Auto-saved CMYK JPG without ICC profile")
                    except Exception as e:
                        print(f"Error saving CMYK JPG: {e}, falling back to RGB")
                        rgb_canvas = canvas.convert("RGB")
                        rgb_canvas.save(out_path, "JPEG", quality=100, optimize=True)
                else:
                    # Regular RGB JPG
                    rgb_canvas = canvas.convert("RGB") 
                    
                    # Special handling for CMYK originals with the "none" method (fallback case)
                    if original_mode == "CMYK" and bg_method == "none" and icc_profile:
                        print("Using special CMYK handling for JPG output with 'none' method")
                        try:
                            # Create a custom RGB profile from the original CMYK profile
                            srgb_profile = ImageCms.createProfile("sRGB")
                            rgb_canvas.save(out_path, "JPEG", quality=100, optimize=True, icc_profile=srgb_profile.tobytes())
                        except Exception as e:
                            print(f"Error with custom profile save: {e}, using standard profile")
                            # Fallback to using the original profile
                            try:
                                rgb_canvas.save(out_path, "JPEG", quality=100, optimize=True, icc_profile=icc_profile)
                            except Exception as e2:
                                print(f"Error saving with ICC profile: {e2}, falling back to standard save")
                                rgb_canvas.save(out_path, "JPEG", quality=100, optimize=True)
                    elif icc_profile:
                        try:
                            # Use original ICC profile for best color matching
                            rgb_canvas.save(out_path, "JPEG", quality=100, optimize=True, icc_profile=icc_profile)
                        except Exception as e:
                            print(f"Error saving with ICC profile: {e}, falling back to standard save")
                            rgb_canvas.save(out_path, "JPEG", quality=100, optimize=True)
                        else:
                        # If no profile, use sRGB (standard) conversion
                            rgb_canvas.save(out_path, "JPEG", quality=100, optimize=True)
            else:
                # PNG output
                if icc_profile:
                    try:
                        canvas.save(out_path, "PNG", optimize=True, icc_profile=icc_profile)
                    except Exception as e:
                        print(f"Error saving PNG with ICC profile: {e}, falling back to standard save")
                        canvas.save(out_path, "PNG", optimize=True)
                else:
                    canvas.save(out_path, "PNG", optimize=True)
        
        os.remove(temp_image_path)
        
        # Clean up temporary cropped file if it was created
        if was_cropped and os.path.exists(working_image_path):
            try:
                os.remove(working_image_path)
                print(f"Cleaned up temporary cropped file: {working_image_path}")
            except Exception as e:
                print(f"Error cleaning up temporary file {working_image_path}: {e}")
        
        # Clean up temporary .arw converted file if it was created
        if was_arw_converted and arw_temp_path and os.path.exists(arw_temp_path):
            try:
                os.remove(arw_temp_path)
                print(f"Cleaned up temporary ARW converted file: {arw_temp_path}")
            except Exception as e:
                print(f"Error cleaning up temporary ARW file {arw_temp_path}: {e}")
        
        print(f"Processed with zero padding => {out_path}")
        return [(out_path, image_path)], logs, classification_result

    # If zero padding is used with auto_snap, log the information but continue with normal processing  
    if zero_padding and auto_snap:
        print(f"File {filename}: Zero padding skipped because auto_snap is active. Using manual snap logic instead.")
        print(f"   🎯 Auto snap flags will be processed by position_logic_old(): snap_to_top={snap_to_top}, snap_to_bottom={snap_to_bottom}, snap_to_left={snap_to_left}, snap_to_right={snap_to_right}")
    
    # Log information for none_with_padding method
    if bg_method == 'none_with_padding':
        print(f"File {filename}: Using none_with_padding method - position will be adjusted according to padding settings.")
        print(f"   📐 Padding settings: top={padding_top}, right={padding_right}, bottom={padding_bottom}, left={padding_left}")

    # Special handling for none_with_padding method
    if bg_method == 'none_with_padding':
        print(f"Using none_with_padding method for {filename}")
        
        # Use the new position_with_padding function instead of position_logic_old
        cropped_img, x, y, logs = position_with_padding(
            mask, canvas_size, padding_top, padding_right, padding_bottom, padding_left
        )
        
        print(f"none_with_padding positioning complete: size={cropped_img.size}, position=({x},{y})")
    else:
        # Normal processing logic (for when zero padding is not used or when auto_snap is active)
        logs, cropped_img, x, y = position_logic_old(
            temp_image_path, canvas_size, padding_top, padding_right, padding_bottom, padding_left, 
            use_threshold=True, bg_method=bg_method, is_person=any(term in classification_result.lower() for term in ["human", "person", "model"]) if classification_result else False,
            snap_to_top=snap_to_top, snap_to_bottom=snap_to_bottom,
            snap_to_left=snap_to_left, snap_to_right=snap_to_right
        )

    if bg_choice == 'white':
        canvas = Image.new("RGBA", canvas_size, "WHITE")
    elif bg_choice == 'custom':
        canvas = Image.new("RGBA", canvas_size, custom_color)
    elif bg_choice == 'dominant':
        dom_col = get_dominant_color(original_img)
        canvas = Image.new("RGBA", canvas_size, dom_col)
    else:
        canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))

    canvas.paste(cropped_img, (x, y), cropped_img)
    logs.append({"action": "paste", "x": x, "y": y})

    if flip:
        canvas = flip_image(canvas)
        logs.append({"action": "flip_horizontal"})

    if rotation != "None" and (rotation == "180 Degrees" or direction != "None"):
        if rotation == "90 Degrees":
            angle = 90 if direction == "Clockwise" else -90
        elif rotation == "180 Degrees":
            angle = 180
        else:
            angle = 0
        rotated_subject = cropped_img.rotate(angle, expand=True)
        if bg_choice == 'white':
            new_canvas = Image.new("RGBA", canvas_size, "WHITE")
        elif bg_choice == 'custom':
            new_canvas = Image.new("RGBA", canvas_size, custom_color)
        elif bg_choice == 'dominant':
            dom_col = get_dominant_color(original_img)
            new_canvas = Image.new("RGBA", canvas_size, dom_col)
        else:
            new_canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
        available_width = canvas_size[0] - padding_left - padding_right
        target_height = canvas_size[1] - padding_top - padding_bottom
        rs_w, rs_h = rotated_subject.size
        scale_factor = target_height / rs_h
        new_width_h = int(rs_w * scale_factor)
        if new_width_h > available_width:
            scale_factor = available_width / rs_w
            new_width = available_width
            new_height = int(rs_h * scale_factor)
        else:
            new_width = new_width_h
            new_height = target_height
        rotated_subject = rotated_subject.resize((new_width, new_height), Image.LANCZOS)
        new_x = padding_left + (available_width - new_width) // 2
        new_y = padding_top + (target_height - new_height) // 2
        new_canvas.paste(rotated_subject, (new_x, new_y), rotated_subject)
        canvas = new_canvas
        logs.append({"action": "rotate_final", "rotation": rotation, "direction": direction})

    if stop_event.is_set():
        return None, None

    # Ensure cropped image maintains correct colors if not using 'none' method
    if bg_method != 'none':
        try:
            cropped_img = ensure_color_fidelity(original_img_with_profile, cropped_img)
        except Exception as e:
            print(f"Error ensuring color fidelity for cropped image: {e}")

    # Auto-detect CMYK and force JPG output for CMYK inputs
    if original_mode == "CMYK":
        print(f"Auto-detected CMYK input, forcing JPG output regardless of user selection")
        out_ext = "jpg"
        force_cmyk_output = True
    else:
        out_ext = "jpg" if output_format == "JPG" else "png"
        force_cmyk_output = False
        
    out_filename = f"{os.path.splitext(filename)[0]}.{out_ext}"
    out_path = os.path.join(output_folder, out_filename)

    if (base_no_ext.endswith("_01") or base_no_ext.endswith("_1") or base_no_ext.endswith("_001")) and watermark_path:
        w_img = Image.open(watermark_path).convert("RGBA")
        canvas.paste(w_img, (0, 0), w_img)
        logs.append({"action": "add_watermark"})

    if twibbon_path:
        twb = Image.open(twibbon_path).convert("RGBA")
        canvas.paste(twb, (0, 0), twb)
        logs.append({"action": "twibbon"})

    # Debug final canvas color at safe coordinates
    safe_x = min(center_x, canvas.width - 1)
    safe_y = min(center_y, canvas.height - 1)
    debug_color_values(canvas, safe_x, safe_y, "Final image color")

    # Modified save logic with auto CMYK detection and color profile preservation
    if output_format == "JPG" or force_cmyk_output:
        if force_cmyk_output and original_mode == "CMYK":
            # Auto CMYK conversion for CMYK inputs
            print(f"Auto-converting final image back to CMYK (original mode: {original_mode})")
            
            # First convert canvas to RGB if it's not already
            rgb_canvas = canvas.convert("RGB")
            
            # Convert RGB back to CMYK using original profile if available
            cmyk_canvas = convert_rgb_to_cmyk_with_profile(rgb_canvas, original_cmyk_profile)
            print(f"Auto-converted to CMYK mode: {cmyk_canvas.mode}")
            
            try:
                # Save as CMYK JPEG
                if original_cmyk_profile:
                    cmyk_canvas.save(out_path, "JPEG", quality=100, optimize=True, icc_profile=original_cmyk_profile)
                    print(f"Auto-saved CMYK JPG with original ICC profile")
                else:
                    cmyk_canvas.save(out_path, "JPEG", quality=100, optimize=True)
                    print(f"Auto-saved CMYK JPG without ICC profile")
            except Exception as e:
                print(f"Error saving CMYK JPG: {e}, falling back to RGB")
                rgb_canvas = canvas.convert("RGB")
                rgb_canvas.save(out_path, "JPEG", quality=100, optimize=True)
        else:
            # Regular RGB JPG processing
            rgb_canvas = canvas.convert("RGB") 
            
            # Special handling for CMYK originals with the "none" method (fallback case)
            if original_mode == "CMYK" and bg_method == "none" and icc_profile:
                print("Using special CMYK handling for JPG output with 'none' method")
                try:
                    # Create a custom RGB profile from the original CMYK profile
                    srgb_profile = ImageCms.createProfile("sRGB")
                    rgb_canvas.save(out_path, "JPEG", quality=100, optimize=True, icc_profile=srgb_profile.tobytes())
                except Exception as e:
                    print(f"Error with custom profile save: {e}, using standard profile")
                    # Fallback to using the original profile
                    try:
                        rgb_canvas.save(out_path, "JPEG", quality=100, optimize=True, icc_profile=icc_profile)
                    except Exception as e2:
                        print(f"Error saving with ICC profile: {e2}, falling back to standard save")
                        rgb_canvas.save(out_path, "JPEG", quality=100, optimize=True)
            elif icc_profile:
                try:
                    # Use original ICC profile for best color matching
                    rgb_canvas.save(out_path, "JPEG", quality=100, optimize=True, icc_profile=icc_profile)
                except Exception as e:
                    print(f"Error saving with ICC profile: {e}, falling back to standard save")
                    rgb_canvas.save(out_path, "JPEG", quality=100, optimize=True)
            else:
                # If no profile, use sRGB (standard) conversion
                rgb_canvas.save(out_path, "JPEG", quality=100, optimize=True)
    else:
        # Save as PNG with original color profile
        if original_mode == "CMYK" and bg_method == "none" and icc_profile:
            print("Using special CMYK handling for PNG output with 'none' method")
            try:
                # Create a custom RGB profile from the original CMYK profile
                srgb_profile = ImageCms.createProfile("sRGB")
                canvas.save(out_path, "PNG", optimize=True, icc_profile=srgb_profile.tobytes())
            except Exception as e:
                print(f"Error with custom profile save: {e}, using standard profile")
                try:
                    canvas.save(out_path, "PNG", optimize=True, icc_profile=icc_profile)
                except Exception as e2:
                    print(f"Error saving PNG with ICC profile: {e2}, falling back to standard save")
                    canvas.save(out_path, "PNG", optimize=True)
        elif icc_profile:
            try:
                # Preserve transparent pixels while keeping the ICC profile
                canvas.save(out_path, "PNG", optimize=True, icc_profile=icc_profile)
            except Exception as e:
                print(f"Error saving PNG with ICC profile: {e}, falling back to standard save")
                canvas.save(out_path, "PNG", optimize=True)
        else:
            canvas.save(out_path, "PNG", optimize=True)

    os.remove(temp_image_path)
    
    # Clean up temporary cropped file if it was created
    if was_cropped and os.path.exists(working_image_path):
        try:
            os.remove(working_image_path)
            print(f"Cleaned up temporary cropped file: {working_image_path}")
        except Exception as e:
            print(f"Error cleaning up temporary file {working_image_path}: {e}")
    
    # Clean up temporary .arw converted file if it was created
    if was_arw_converted and arw_temp_path and os.path.exists(arw_temp_path):
        try:
            os.remove(arw_temp_path)
            print(f"Cleaned up temporary ARW converted file: {arw_temp_path}")
        except Exception as e:
            print(f"Error cleaning up temporary ARW file {arw_temp_path}: {e}")
    
    print(f"Processed => {out_path}")
    
    # Prepare classification data in the format 
    classification_data = None
    if classification_result:
        classification_data = {
            "classification": classification_result,
            "padding": padding_used
        }
    
    return [(out_path, image_path)], logs, classification_data

def process_images(
    input_files,
    bg_method='rembg',
    watermark_path=None,
    twibbon_path=None,
    canvas_size='Rox- Columbia & Keen',
    output_format='PNG',
    bg_choice='transparent',
    custom_color="#ffffff",
    num_workers=4,
    rotation=None,
    direction=None,
    flip=False,
    use_old_position=True,
    zero_padding=None,
    auto_detect_padding=False,
    progress=gr.Progress(),
    sheet_data=None,  # New parameter: DataFrame with sheet data
    use_qwen=False,  # New parameter: enable Qwen classification
    snap_to_bottom=False,
    snap_to_top=False,
    snap_to_left=False,
    snap_to_right=False,
    auto_snap=False
):
    stop_event.clear()
    start = time.time()
    if bg_method in ['birefnet', 'birefnet_2']:
        num_workers = 1

    out_folder = "processed_images"
    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    os.makedirs(out_folder)

    procd = []
    origs = []
    all_logs = []
    all_classifications = {}  # Store classification results

    if isinstance(input_files, str) and input_files.lower().endswith(('.zip', '.rar')):
        tmp_in = "temp_input"
        if os.path.exists(tmp_in):
            shutil.rmtree(tmp_in)
        os.makedirs(tmp_in)
        try:
            with zipfile.ZipFile(input_files, 'r') as zf:
                zf.extractall(tmp_in)
        except zipfile.BadZipFile as e:
            print(f"Error extracting zip: {e}")
            return [], None, 0, {}
        images = [os.path.join(tmp_in, f)
                  for f in os.listdir(tmp_in)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp', '.tif', '.tiff', '.avif', '.arw'))]
    elif isinstance(input_files, list):
        images = input_files
    else:
        images = [input_files]

    total = len(images)
    print(f"Total images: {total}")
    avg_time = 0

    with ThreadPoolExecutor(max_workers=num_workers) as exe:
        future_map = {
            exe.submit(
                process_single_image,
                path,
                out_folder,
                bg_method,
                canvas_size,
                output_format,
                bg_choice,
                custom_color,
                watermark_path,
                twibbon_path,
                rotation,
                direction,
                flip,
                use_old_position,
                zero_padding,
                auto_detect_padding,
                sheet_data,  # New parameter: DataFrame with sheet data
                use_qwen,  # New parameter: enable Qwen classification
                snap_to_bottom,
                snap_to_top,
                snap_to_left,
                snap_to_right,
                auto_snap
            ): path for path in images
        }
        for idx, fut in enumerate(future_map):
            if stop_event.is_set():
                print("Stop event triggered.")
                break
            try:
                t0 = time.time()
                result, log, classification_result = fut.result()
                t1 = time.time()
                dur = t1 - t0
                avg_time = (avg_time * idx + dur) / (idx + 1)
                if result:
                    procd.extend(result)
                    origs.append(future_map[fut])
                    all_logs.append({os.path.basename(future_map[fut]): log})
                    
                    # Store classification result
                    if classification_result:
                        filename = os.path.basename(future_map[fut])
                        all_classifications[filename] = classification_result
                        print(f"Stored classification for {filename}: {classification_result}")
                        
                remain = total - (idx + 1)
                eta = remain * avg_time
                progress((idx + 1) / total, f"{idx + 1}/{total} processed, ~{eta:.2f}s left")
            except Exception as e:
                print(f"Error in executor: {e}")
                import traceback
                traceback.print_exc()

    zip_out = "processed_images.zip"
    with zipfile.ZipFile(zip_out, 'w') as zf:
        for outf, _ in procd:
            zf.write(outf, os.path.basename(outf))

    with open(os.path.join(out_folder, "process_log.json"), "w") as lf:
        json.dump(all_logs, lf, indent=2)

    elapsed = time.time() - start
    print(f"Done in {elapsed:.2f}s")
    print(f"Total classifications collected: {len(all_classifications)}")
    return origs, procd, zip_out, elapsed, all_classifications

def gradio_interface(
    input_files,
    bg_method,
    watermark_path=None,
    twibbon_path=None,
    canvas_size='Rox- Columbia & Keen',
    output_format='PNG',
    bg_choice='transparent',
    custom_color="#ffffff",
    num_workers=4,
    rotation=None,
    direction=None,
    flip=False,
    use_old_position=True,
    zero_padding=None,
    auto_detect_padding=False,
    progress=gr.Progress(),
    sheet_data=None,  # New parameter: DataFrame with sheet data
    use_qwen=False,  # New parameter: enable Qwen classification
    snap_to_bottom=False,
    snap_to_top=False,
    snap_to_left=False,
    snap_to_right=False,
    auto_snap=False
):
    if bg_method in ['birefnet', 'birefnet_2', 'birefnet_hr']:
        num_workers = min(num_workers, 2)

    # Make sure watermark_path and twibbon_path are correctly passed
    print(f"gradio_interface: watermark_path: {watermark_path}, twibbon_path: {twibbon_path}")
    
    if isinstance(input_files, str) and input_files.lower().endswith(('.zip', '.rar')):
        return process_images(
            input_files, bg_method, watermark_path, twibbon_path,
            canvas_size, output_format, bg_choice, custom_color, num_workers,
            rotation, direction, flip, use_old_position, zero_padding, auto_detect_padding,
            progress, sheet_data, use_qwen, snap_to_bottom, snap_to_top, snap_to_left, snap_to_right, auto_snap
        )
    elif isinstance(input_files, list):
        return process_images(
            input_files, bg_method, watermark_path, twibbon_path,
            canvas_size, output_format, bg_choice, custom_color, num_workers,
            rotation, direction, flip, use_old_position, zero_padding, auto_detect_padding,
            progress, sheet_data, use_qwen, snap_to_bottom, snap_to_top, snap_to_left, snap_to_right, auto_snap
        )
    else:
        return process_images(
            input_files.name, bg_method, watermark_path, twibbon_path,
            canvas_size, output_format, bg_choice, custom_color, num_workers,
            rotation, direction, flip, use_old_position, zero_padding, auto_detect_padding,
            progress, sheet_data, use_qwen, snap_to_bottom, snap_to_top, snap_to_left, snap_to_right, auto_snap
        )

def show_color_picker(bg_choice):
    if bg_choice == 'custom':
        return gr.update(visible=True)
    return gr.update(visible=False)

def update_compare(evt: gr.SelectData, classifications):
    try:
        print(f"update_compare called with classifications: {classifications}")
        
        if isinstance(evt.value, dict) and 'caption' in evt.value:
            proc_path = evt.value['image']['path']
            caption = evt.value['caption']
            print(f"Processing selection: {caption}, path: {proc_path}")
            
            # Extract original filename from caption
            if caption.startswith("Main: "):
                orig_filename = caption.replace("Main: ", "")
                orig_path = os.path.join("original_images_temp", f"orig_main_{orig_filename}")
                
                # Check if this is a .arw file that was converted to .png for comparison
                if orig_filename.lower().endswith('.arw'):
                    arw_converted_path = orig_path.replace('.ARW', '.png').replace('.arw', '.png')
                    if os.path.exists(arw_converted_path):
                        orig_path = arw_converted_path
                        print(f"Using converted .arw file for comparison: {orig_path}")
                        
            elif caption.startswith("Special: "):
                orig_filename = caption.replace("Special: ", "")
                orig_path = os.path.join("original_images_temp", f"orig_special_{orig_filename}")
                
                # Check if this is a .arw file that was converted to .png for comparison
                if orig_filename.lower().endswith('.arw'):
                    arw_converted_path = orig_path.replace('.ARW', '.png').replace('.arw', '.png')
                    if os.path.exists(arw_converted_path):
                        orig_path = arw_converted_path
                        print(f"Using converted .arw file for comparison: {orig_path}")
                        
            elif caption.startswith("No BG: "):
                orig_filename = caption.replace("No BG: ", "")
                orig_path = os.path.join("original_images_temp", f"orig_nobg_{orig_filename}")
                
                # Check if this is a .arw file that was converted to .png for comparison
                if orig_filename.lower().endswith('.arw'):
                    arw_converted_path = orig_path.replace('.ARW', '.png').replace('.arw', '.png')
                    if os.path.exists(arw_converted_path):
                        orig_path = arw_converted_path
                        print(f"Using converted .arw file for comparison: {orig_path}")
                        
            else:
                # Legacy handling for older captions
                orig_filename = caption
                orig_path = caption.split("Input: ")[-1] if "Input: " in caption else ""
            
            print(f"Extracted filename: {orig_filename}")
            
            # Get classification info for this file
            classification_info = ""
            if classifications and isinstance(classifications, dict):
                print(f"Available classifications keys: {list(classifications.keys())}")
                
                # Try exact match first
                if orig_filename in classifications:
                    cls_data = classifications[orig_filename]
                    if isinstance(cls_data, dict) and 'classification' in cls_data:
                        cls = cls_data["classification"]
                        pad = cls_data["padding"]
                        classification_info = f"Classification: {cls}, Padding - Top: {pad['top']}, Bottom: {pad['bottom']}, Left: {pad['left']}, Right: {pad['right']}"
                    else:
                        # Fallback for simple string format
                        classification_info = f"Classification: {cls_data}"
                    print(f"Found exact match for {orig_filename}")
                else:
                    # Try to find by searching in the classifications dict
                    found = False
                    for key, value in classifications.items():
                        if orig_filename in key or key in orig_filename:
                            if isinstance(value, dict) and 'classification' in value:
                                cls = value["classification"]
                                pad = value["padding"]
                                classification_info = f"Classification: {cls}, Padding - Top: {pad['top']}, Bottom: {pad['bottom']}, Left: {pad['left']}, Right: {pad['right']}"
                            else:
                                classification_info = f"Classification: {value}"
                            print(f"Found partial match: {key} -> {value}")
                            found = True
                            break
                    
                    if not found:
                        classification_info = f"No classification found for: {orig_filename} (Available: {list(classifications.keys())[:3]}...)"
            else:
                classification_info = "No classification data available"
            
            print(f"Final classification info: {classification_info}")
            
            if os.path.exists(orig_path) and os.path.exists(proc_path):
                try:
                    orig = Image.open(orig_path)
                    proc = Image.open(proc_path)
                    ratio_o = f"{orig.width}x{orig.height}"
                    ratio_p = f"{proc.width}x{proc.height}"
                    print(f"Successfully opened images for comparison - Original: {ratio_o}, Processed: {ratio_p}")
                    return (gr.update(value=orig_path),
                            gr.update(value=proc_path),
                            gr.update(value=ratio_o),
                            gr.update(value=ratio_p),
                            gr.update(value=classification_info))
                except Exception as e:
                    print(f"Error opening images for comparison: {e}")
                    return (gr.update(value=None), gr.update(value=proc_path),
                            gr.update(value="Error opening image"), gr.update(value=f"Processed exists: {os.path.exists(proc_path)}"),
                            gr.update(value=f"Classification: {classification_info} | Error: {e}"))
            else:
                print(f"Error in update_compare: File not found - Original: {orig_path}, Processed: {proc_path}")
                if not os.path.exists(orig_path):
                    print(f"Original file missing: {orig_path}")
                if not os.path.exists(proc_path):
                    print(f"Processed file missing: {proc_path}")
                return (gr.update(value=None), gr.update(value=proc_path),
                        gr.update(value="File not found"), gr.update(value=f"Processed exists: {os.path.exists(proc_path)}"),
                        gr.update(value=f"Classification: {classification_info} | File not found"))
        else:
            print("No caption found in selection.")
            return (gr.update(value=None),)*5
    except Exception as e:
        import traceback
        print(f"Error in update_compare: {e}")
        traceback.print_exc()
        return (gr.update(value=None),)*5

def process(
    input_files,
    bg_method,
    watermark,
    twibbon,
    canvas_size,
    output_format,
    bg_choice,
    custom_color,
    num_workers,
    rotation=None,
    direction=None,
    flip=False,
    use_old_position=True,
    zero_padding=None,
    auto_detect_padding=False,
    special_input_files=None,
    special_rotation=None,
    special_direction=None,
    special_flip=False,
    no_bg_removal_files=None,
    sheet_file=None,  # New parameter: CSV/Excel file with classification data
    use_qwen_str="Default (No Vision)",  # New parameter: string instead of boolean
    snap_to_bottom=False,
    snap_to_top=False,
    snap_to_left=False,
    snap_to_right=False,
    auto_snap=False
):
    # Convert watermark and twibbon to paths early to ensure consistent use across all rows
    watermark_path = watermark.name if watermark else None
    twibbon_path = twibbon.name if twibbon else None
    
    # Convert use_qwen string to boolean
    use_qwen = use_qwen_str == "Utilize Vision Model"
    
    # Print debug info for watermark and twibbon
    if watermark_path:
        print(f"Using watermark: {watermark_path}")
        if not os.path.exists(watermark_path):
            print(f"WARNING: Watermark file does not exist: {watermark_path}")
    if twibbon_path:
        print(f"Using twibbon: {twibbon_path}")
        if not os.path.exists(twibbon_path):
            print(f"WARNING: Twibbon file does not exist: {twibbon_path}")
    
    # Load sheet data if provided and Qwen is enabled
    sheet_data = None
    if use_qwen and sheet_file is not None:
        try:
            print(f"Loading sheet data for Qwen classification: {sheet_file.name}")
            if sheet_file.name.endswith('.csv'):
                sheet_data = pd.read_csv(sheet_file.name)
            elif sheet_file.name.endswith(('.xlsx', '.xls')):
                sheet_data = pd.read_excel(sheet_file.name)
            else:
                print(f"Unsupported file format: {sheet_file.name}. Only CSV and Excel files are supported.")
                
            if sheet_data is not None:
                print(f"Sheet data loaded successfully: {len(sheet_data)} rows, columns: {list(sheet_data.columns)}")
                
                # Validate required columns
                if 'Classification' not in sheet_data.columns and 'object' not in sheet_data.columns:
                    print("Warning: Neither 'Classification' nor 'object' column found in sheet data. Classification may not work properly.")
                    print(f"Available columns: {list(sheet_data.columns)}")
                elif 'Classification' in sheet_data.columns:
                    print("Found 'Classification' column in sheet data")
                elif 'object' in sheet_data.columns:
                    print("Found 'object' column in sheet data (using as fallback)")
            else:
                print("Failed to load sheet data.")
        except Exception as e:
            print(f"Error loading sheet data: {e}")
            sheet_data = None
            
    results = []
    total_time = 0
    all_classifications = {}  # Store classification results for each file
    
    # Initialize variables to track processing results
    main_procd = []
    special_procd = []
    no_bg_procd = []
    
    # Ensure at least one type of input file is available
    if not input_files and not special_input_files and not no_bg_removal_files:
        return [], None, "No files uploaded. Please upload at least one file to one of the input fields."
    
    # Create a directory to store all processed files for the combined ZIP
    combined_output_folder = "combined_processed_images_temp"
    if os.path.exists(combined_output_folder):
        shutil.rmtree(combined_output_folder)
    os.makedirs(combined_output_folder)
    
    # Create a directory to store copies of original files for comparison
    originals_folder = "original_images_temp"
    if os.path.exists(originals_folder):
        shutil.rmtree(originals_folder)
    os.makedirs(originals_folder)
    
    # Track filename counters to avoid duplicates
    filename_counters = {}
    
    # Process main images
    if input_files:
        print(f"Processing main input files with zero_padding: {zero_padding}")
        main_origs, main_procd, main_zip, main_tt, main_classifications = gradio_interface(
            input_files, bg_method, watermark_path, twibbon_path,
            canvas_size, output_format, bg_choice, custom_color, num_workers,
            rotation, direction, flip, use_old_position, zero_padding, auto_detect_padding,
            sheet_data=sheet_data, use_qwen=use_qwen, snap_to_bottom=snap_to_bottom, snap_to_top=snap_to_top, snap_to_left=snap_to_left, snap_to_right=snap_to_right, auto_snap=auto_snap
        )
        print(f"Completed processing main input files. Results: {len(main_procd)} files")
        
        if main_procd:
            for outf, inf in main_procd:
                # Copy the processed file to our combined folder
                if os.path.exists(outf):
                    try:
                        # Use original filename without prefix
                        base_filename = os.path.basename(outf)
                        
                        # Handle potential duplicates with a counter suffix
                        if base_filename in filename_counters:
                            filename_counters[base_filename] += 1
                            name, ext = os.path.splitext(base_filename)
                            new_filename = f"{name}_{filename_counters[base_filename]}{ext}"
                        else:
                            filename_counters[base_filename] = 0
                            new_filename = base_filename
                            
                        new_path = os.path.join(combined_output_folder, new_filename)
                        shutil.copy2(outf, new_path)
                        
                        # Copy the original file for comparison
                        orig_filename = f"orig_main_{os.path.basename(inf)}"
                        orig_path = os.path.join(originals_folder, orig_filename)
                        
                        # Special handling for .arw files - convert to viewable format
                        if inf.lower().endswith('.arw'):
                            try:
                                print(f"Converting .arw for comparison: {inf}")
                                # Convert .arw to PIL Image
                                arw_image = convert_arw_to_pil(inf)
                                if arw_image:
                                    # Save as PNG for comparison (change extension)
                                    orig_filename_png = orig_filename.replace('.ARW', '.png').replace('.arw', '.png')
                                    orig_path_png = os.path.join(originals_folder, orig_filename_png)
                                    arw_image.save(orig_path_png, 'PNG')
                                    print(f"Saved converted .arw for comparison: {orig_path_png}")
                                    # Update the path for results
                                    orig_path = orig_path_png
                                else:
                                    print(f"Failed to convert .arw for comparison: {inf}")
                                    # Fallback: just copy the original .arw (won't display but path will exist)
                                    if os.path.exists(inf):
                                        shutil.copy2(inf, orig_path)
                            except Exception as e:
                                print(f"Error converting .arw for comparison: {e}")
                                # Fallback: just copy the original .arw
                                if os.path.exists(inf):
                                    shutil.copy2(inf, orig_path)
                        else:
                            # Regular file handling
                            if os.path.exists(inf):
                                shutil.copy2(inf, orig_path)
                        
                        # Store paths for both processed and original
                        results.append((new_path, orig_path))
                    except Exception as e:
                        print(f"Error copying file {outf}: {e}")
            
            # Collect classification results
            if main_classifications:
                all_classifications.update(main_classifications)
                print(f"Added main classifications: {main_classifications}")
            
            total_time += main_tt
    else:
        print("No main input files provided, skipping this step")
        filename_counters = {}
    
    # Process special rotation images if provided
    if special_input_files:
        print(f"Processing special input files with zero_padding: {zero_padding}")
        special_origs, special_procd, special_zip, special_tt, special_classifications = gradio_interface(
            special_input_files, bg_method, watermark_path, twibbon_path,
            canvas_size, output_format, bg_choice, custom_color, num_workers,
            special_rotation, special_direction, special_flip, use_old_position, zero_padding, auto_detect_padding,
            sheet_data=sheet_data, use_qwen=use_qwen, snap_to_bottom=snap_to_bottom, snap_to_top=snap_to_top, snap_to_left=snap_to_left, snap_to_right=snap_to_right, auto_snap=auto_snap
        )
        print(f"Completed processing special input files. Results: {len(special_procd)} files")
        
        if special_procd:
            for outf, inf in special_procd:
                # Copy the processed file to our combined folder
                if os.path.exists(outf):
                    try:
                        # Use original filename without prefix
                        base_filename = os.path.basename(outf)
                        
                        # Handle potential duplicates with a counter suffix
                        if base_filename in filename_counters:
                            filename_counters[base_filename] += 1
                            name, ext = os.path.splitext(base_filename)
                            new_filename = f"{name}_{filename_counters[base_filename]}{ext}"
                        else:
                            filename_counters[base_filename] = 0
                            new_filename = base_filename
                            
                        new_path = os.path.join(combined_output_folder, new_filename)
                        shutil.copy2(outf, new_path)
                        
                        # Copy the original file for comparison
                        orig_filename = f"orig_special_{os.path.basename(inf)}"
                        orig_path = os.path.join(originals_folder, orig_filename)
                        
                        # Special handling for .arw files - convert to viewable format
                        if inf.lower().endswith('.arw'):
                            try:
                                print(f"Converting .arw for comparison: {inf}")
                                # Convert .arw to PIL Image
                                arw_image = convert_arw_to_pil(inf)
                                if arw_image:
                                    # Save as PNG for comparison (change extension)
                                    orig_filename_png = orig_filename.replace('.ARW', '.png').replace('.arw', '.png')
                                    orig_path_png = os.path.join(originals_folder, orig_filename_png)
                                    arw_image.save(orig_path_png, 'PNG')
                                    print(f"Saved converted .arw for comparison: {orig_path_png}")
                                    # Update the path for results
                                    orig_path = orig_path_png
                                else:
                                    print(f"Failed to convert .arw for comparison: {inf}")
                                    # Fallback: just copy the original .arw (won't display but path will exist)
                                    if os.path.exists(inf):
                                        shutil.copy2(inf, orig_path)
                            except Exception as e:
                                print(f"Error converting .arw for comparison: {e}")
                                # Fallback: just copy the original .arw
                                if os.path.exists(inf):
                                    shutil.copy2(inf, orig_path)
                        else:
                            # Regular file handling
                            if os.path.exists(inf):
                                shutil.copy2(inf, orig_path)
                        
                        # Store paths for both processed and original
                        results.append((new_path, orig_path))
                    except Exception as e:
                        print(f"Error copying file {outf}: {e}")
            
            # Collect classification results for special files
            if special_classifications:
                all_classifications.update(special_classifications)
                print(f"Added special classifications: {special_classifications}")
                
            total_time += special_tt
    else:
        print("No special input files provided, skipping this step")
            
    # Process images without background removal if provided
    if no_bg_removal_files:
        print(f"Processing no-bg input files with zero_padding: {zero_padding}")
        no_bg_origs, no_bg_procd, no_bg_zip, no_bg_tt, no_bg_classifications = gradio_interface(
            no_bg_removal_files, "none", watermark_path, twibbon_path,
            canvas_size, output_format, bg_choice, custom_color, num_workers,
            rotation, direction, flip, use_old_position, zero_padding, auto_detect_padding,
            sheet_data=sheet_data, use_qwen=use_qwen, snap_to_bottom=snap_to_bottom, snap_to_top=snap_to_top, snap_to_left=snap_to_left, snap_to_right=snap_to_right, auto_snap=auto_snap
        )
        print(f"Completed processing no-bg input files. Results: {len(no_bg_procd)} files")
        
        if no_bg_procd:
            for outf, inf in no_bg_procd:
                # Copy the processed file to our combined folder
                if os.path.exists(outf):
                    try:
                        # Use original filename without prefix
                        base_filename = os.path.basename(outf)
                        
                        # Handle potential duplicates with a counter suffix
                        if base_filename in filename_counters:
                            filename_counters[base_filename] += 1
                            name, ext = os.path.splitext(base_filename)
                            new_filename = f"{name}_{filename_counters[base_filename]}{ext}"
                        else:
                            filename_counters[base_filename] = 0
                            new_filename = base_filename
                            
                        new_path = os.path.join(combined_output_folder, new_filename)
                        shutil.copy2(outf, new_path)
                        
                        # Copy the original file for comparison
                        orig_filename = f"orig_nobg_{os.path.basename(inf)}"
                        orig_path = os.path.join(originals_folder, orig_filename)
                        
                        # Special handling for .arw files - convert to viewable format
                        if inf.lower().endswith('.arw'):
                            try:
                                print(f"Converting .arw for comparison: {inf}")
                                # Convert .arw to PIL Image
                                arw_image = convert_arw_to_pil(inf)
                                if arw_image:
                                    # Save as PNG for comparison (change extension)
                                    orig_filename_png = orig_filename.replace('.ARW', '.png').replace('.arw', '.png')
                                    orig_path_png = os.path.join(originals_folder, orig_filename_png)
                                    arw_image.save(orig_path_png, 'PNG')
                                    print(f"Saved converted .arw for comparison: {orig_path_png}")
                                    # Update the path for results
                                    orig_path = orig_path_png
                                else:
                                    print(f"Failed to convert .arw for comparison: {inf}")
                                    # Fallback: just copy the original .arw (won't display but path will exist)
                                    if os.path.exists(inf):
                                        shutil.copy2(inf, orig_path)
                            except Exception as e:
                                print(f"Error converting .arw for comparison: {e}")
                                # Fallback: just copy the original .arw
                                if os.path.exists(inf):
                                    shutil.copy2(inf, orig_path)
                        else:
                            # Regular file handling
                            if os.path.exists(inf):
                                shutil.copy2(inf, orig_path)
                        
                        # Store paths for both processed and original
                        results.append((new_path, orig_path))
                    except Exception as e:
                        print(f"Error copying file {outf}: {e}")
            total_time += no_bg_tt
            
            # Collect classification results for no-bg files
            if no_bg_classifications:
                all_classifications.update(no_bg_classifications)
                print(f"Added no-bg classifications: {no_bg_classifications}")
    else:
        print("No no-bg input files provided, skipping this step")
    
    if not results:
        # Check each input to provide a more specific error message
        missing_inputs = []
        if input_files and not main_procd:
            missing_inputs.append("Main Image Input")
        if special_input_files and not special_procd:
            missing_inputs.append("Rotation/Flip Input")
        if no_bg_removal_files and not no_bg_procd:
            missing_inputs.append("No BG Removal Input")
            
        if missing_inputs:
            error_msg = f"Processing failed for input: {', '.join(missing_inputs)}. Please check input files and try again."
        else:
            error_msg = "No images were successfully processed. Please check input files and try again."
        
        return [], None, error_msg, "No classifications performed.", all_classifications
    
    # Prepare gallery results with proper captions
    gallery_results = []
    for proc_path, orig_path in results:
        # Create caption with type prefix based on filename
        base_filename = os.path.basename(orig_path)
        # Remove the 'orig_' prefix from original filenames
        if base_filename.startswith("orig_main_"):
            caption = f"Main: {base_filename.replace('orig_main_', '')}"
        elif base_filename.startswith("orig_special_"):
            caption = f"Special: {base_filename.replace('orig_special_', '')}"
        elif base_filename.startswith("orig_nobg_"):
            caption = f"No BG: {base_filename.replace('orig_nobg_', '')}"
        else:
            caption = base_filename
            
        gallery_results.append((proc_path, caption))
    
    # Create a combined ZIP file with all processed images
    timestamp = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
    combined_zip_path = f"processed_images_{timestamp}.zip"
    try:
        with zipfile.ZipFile(combined_zip_path, 'w') as zf:
            for proc_path, _ in results:
                if os.path.exists(proc_path):
                    try:
                        zf.write(proc_path, os.path.basename(proc_path).replace("main_", "").replace("special_", "").replace("nobg_", ""))
                    except Exception as e:
                        print(f"Error adding {proc_path} to ZIP: {e}")
        
        # Show processing summary
        print(f"=============== PROCESSING SUMMARY ===============")
        print(f"Main input files processed: {len(main_procd)}")
        print(f"Special input files processed: {len(special_procd)}")
        print(f"No-BG input files processed: {len(no_bg_procd)}")
        print(f"Total files in ZIP: {len(results)}")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"ZIP file created: {combined_zip_path}")
        print(f"================================================")
        
    except Exception as e:
        print(f"Error creating combined ZIP file: {e}")
        # Create classification display text
        class_text = ""
        if all_classifications:
            class_text = "\n".join([
                f"{img}: Classification - {data['classification']}, Padding - Top: {data['padding']['top']}, Bottom: {data['padding']['bottom']}, Left: {data['padding']['left']}, Right: {data['padding']['right']}"
                if isinstance(data, dict) and 'classification' in data 
                else f"{img}: {data}"
                for img, data in all_classifications.items()
            ]) or "No classifications recorded."
        else:
            class_text = "No classifications performed."
        
        print(f"Final classification summary: {all_classifications}")
        print(f"Classification display text: {class_text}")
        
        return gallery_results, None, f"{total_time:.2f} seconds, but ZIP creation failed: {str(e)}", class_text, all_classifications
    
    # Create classification display text
    class_text = ""
    if all_classifications:
        class_text = "\n".join([
            f"{img}: Classification - {data['classification']}, Padding - Top: {data['padding']['top']}, Bottom: {data['padding']['bottom']}, Left: {data['padding']['left']}, Right: {data['padding']['right']}"
            if isinstance(data, dict) and 'classification' in data 
            else f"{img}: {data}"
            for img, data in all_classifications.items()
        ]) or "No classifications recorded."
    else:
        class_text = "No classifications performed."
    
    print(f"Final classification summary: {all_classifications}")
    print(f"Classification display text: {class_text}")
    
    return gallery_results, combined_zip_path, f"{total_time:.2f} seconds", class_text, all_classifications

def stop_processing():
    stop_event.set()

def detect_reference_text(image_path, target_text="*This image is for size reference only"):
    """
    Detect if the target text exists in the image using EasyOCR
    
    Args:
        image_path: Path to the image file
        target_text: Text to search for (default: "*This image is for size reference only")
        
    Returns:
        bool: True if text is found, False otherwise
    """
    try:
        # Get EasyOCR reader instance
        reader = get_easyocr_reader()
        if reader is None:
            print("Warning: EasyOCR not available. Text detection disabled.")
            return False
        
        # Read text from image using EasyOCR
        # EasyOCR returns list of (bbox, text, confidence)
        results = reader.readtext(image_path)
        
        # Extract all text from results
        extracted_texts = []
        for (bbox, text, confidence) in results:
            # Only consider text with reasonable confidence (>0.1)
            if confidence > 0.1:
                extracted_texts.append(text.strip().lower())
        
        # Combine all extracted text
        all_text = " ".join(extracted_texts)
        target_text_clean = target_text.lower()
        
        # Check if target text exists in extracted text
        if target_text_clean in all_text:
            print(f"Reference text detected in {os.path.basename(image_path)} (direct match)")
            return True
        
        # Also check with regex for more flexible matching
        # Handle variations like missing asterisk, extra spaces, etc.
        pattern_variations = [
            r"this\s+image\s+is\s+for\s+size\s+reference\s+only",
            r"\*?\s*this\s+image\s+is\s+for\s+size\s+reference\s+only",
            r"size\s+reference\s+only",
            r"for\s+size\s+reference"
        ]
        
        for pattern in pattern_variations:
            if re.search(pattern, all_text, re.IGNORECASE):
                print(f"Reference text detected in {os.path.basename(image_path)} (pattern match)")
                return True
        
        # Debug: print extracted text for troubleshooting
        if extracted_texts:
            print(f"Extracted text from {os.path.basename(image_path)}: {extracted_texts[:3]}...")  # Show first 3 results
        
        return False
        
    except ImportError:
        print("Warning: EasyOCR not available. Text detection disabled.")
        return False
    except Exception as e:
        print(f"Error in text detection for {image_path}: {e}")
        print("Text detection will be skipped for this image.")
        return False

def crop_bottom_if_reference_text(image_path, crop_pixels=36):
    """
    Crop the bottom of the image if reference text is detected
    
    Args:
        image_path: Path to the image file
        crop_pixels: Number of pixels to crop from bottom (default: 36)
        
    Returns:
        str: Path to the cropped image (temporary file) or original path if no cropping needed
    """
    try:
        if detect_reference_text(image_path):
            print(f"Cropping bottom {crop_pixels}px from {os.path.basename(image_path)}")
            
            # Open the image
            image = Image.open(image_path)
            
            # Calculate new height after cropping
            original_width, original_height = image.size
            new_height = max(original_height - crop_pixels, 1)  # Ensure height is at least 1
            
            # Crop the image (remove bottom portion)
            cropped_image = image.crop((0, 0, original_width, new_height))
            
            # Save to temporary file
            temp_path = image_path.rsplit('.', 1)[0] + '_cropped_temp.' + image_path.rsplit('.', 1)[1]
            cropped_image.save(temp_path)
            
            print(f"Image cropped: {original_width}x{original_height} -> {original_width}x{new_height}")
            return temp_path
        else:
            return image_path
            
    except Exception as e:
        print(f"Error in cropping image {image_path}: {e}")
        print("Image will be processed without cropping.")
        return image_path

with gr.Blocks(theme='allenai/gradio-theme') as iface:
    gr.Markdown("## Image BG Removal with Rotation, Watermark, Twibbon & Qwen Classifications")
    
    with gr.Row():
        input_files = gr.File(label="Upload (Image(s)/ZIP/RAR)", file_types=[".zip", ".rar", "image", ".arw"], interactive=True)
        watermark = gr.File(label="Watermark (Optional)", file_types=[".png"])
        twibbon = gr.File(label="Twibbon (Optional)", file_types=[".png"])
        special_input_files = gr.File(
            label="Upload for Rotation/Direction/Flip (Optional)", 
            file_types=[".zip", ".rar", "image", ".arw"], 
            interactive=True
        )
        no_bg_removal_files = gr.File(
            label="Upload Images to Process Without Background Removal", 
            file_types=[".zip", ".rar", "image", ".arw"], 
            interactive=True
        )
        sheet_file = gr.File(
            label="Upload Classification Sheet (CSV/Excel) - Must have 'Classification' column", 
            file_types=[".csv", ".xlsx", ".xls"],
            interactive=True
        )

    with gr.Row():
        bg_method = gr.Radio(["bria", "rembg", "photoroom", "birefnet", "birefnet_2", "birefnet_hr", "none", "none_with_padding"],
                             label="Background Removal", value="bria")
        bg_choice = gr.Radio(["transparent", "white", "custom"], label="BG Choice", value="white")
        custom_color = gr.ColorPicker(label="Custom BG", value="#ffffff", visible=False)
        output_format = gr.Radio(["PNG", "JPG"], label="Output Format", value="JPG")
        num_workers = gr.Slider(1, 16, 1, label="Number of Workers", value=5)
        use_qwen = gr.Dropdown(
            ["Default (No Vision)", "Utilize Vision Model"],
            label="Classification",
            value="Default (No Vision)"  # Default is off
        )

    with gr.Row():
        canvas_size = gr.Radio(
            choices=[
                "Aetrex-L/S", "Allbirds-L/S", "Backjoy-L/S", "Beecho-L/S", "Billabong-L/S", "Birkenstock-L/S", "Bratpack-L/S", "Ccilu-L/S", "Columbia-L/S", "DC-L/S", "Delsey-L/S", "Drmartens-L/S", "Fitflop-L/S", "Fjallraven-L/S", "Fox-L/S", "FreedomMoses-L/S", "Gregory-L/S", "Hedgren-L/S", "Hellolulu-L/S", "Herschel-L/S", "Heydude-L/S", "Hydroflask-L/S", "Ipanema-L/S", "Jansport-L/S", "Keen-L/S", "Livall-L/S", "Lixil-American Standard-L/S", "Lojel-L/S", "Moleskine-L/S", "Native-L/S", "Osprey-L/S", "Parkland-L/S", "Poler-L/S", "Quiksilver-L/S", "Reef-L/S", "RTR-L/S", "Rider-L/S", "Rockport-L/S", "Roxy-L/S", "RVCA-L/S", "Sakroots-L/S", "Salomon-L/S", "SeaToSummit-L/S", "Sledgers-L/S", "Theragun-L/S", "Timbuk2-L/S", "TNF-L/S", "TomsSG-L/S", "TopoDesigns-L/S", "Travelon-L/S", "Tretorn-L/S", "WorldTraveller-L/S", "Zaxy-L/S", "Niko-L/S", "ACE-L/S", "Coghlans-L/S", "Inochi-L/S",
                "Allbirds-Zalora", "Backjoy-Zalora", "Columbia-Zalora", "DC-Zalora", "Delsey-Zalora", "Drmartens-Zalora", "FreedomMoses-Zalora", "Hedgren-Zalora", "Herschel-Zalora", "Hydroflask-Zalora", "Ipanema-Zalora", "Jansport-Zalora", "LCS (LeCoqSportif)-Zalora", "Rider-Zalora", "Roxy-Zalora", "Sakroots-Zalora", "Salomon-Zalora", "SeaToSummit-Zalora", "TNF-Zalora", "Travelon-Zalora", "Tretorn-Zalora", "Zaxy-Zalora",
                "Aetrex-DOTCOM", "Allbirds-DOTCOM", "Backjoy-DOTCOM", "Bratpack-DOTCOM", "Bratpack Hydroflask-DOTCOM", "Bratpack Parkland-DOTCOM", "Bratpack TNF-DOTCOM", "Columbia-DOTCOM", "DC-DOTCOM", "EYS-DOTCOM", "Grind Bags-DOTCOM", "Grind Hydroflask-DOTCOM", "Grind Shoes-DOTCOM", "Hedgren-DOTCOM"
            ],
            label="Canvas Size", value="Aetrex-L/S"
        )

    with gr.Row():
        # Main image controls
        rotation = gr.Radio(["None", "90 Degrees", "180 Degrees"], label="Rotation Angle (Main)", value="None", visible=False)
        direction = gr.Radio(["None", "Clockwise", "Anticlockwise"], label="Direction (Main)", value="None", visible=False)
        flip_option = gr.Checkbox(label="Flip Horizontal (Main)", value=False, visible=False)
        
        # Special image controls
        special_rotation = gr.Radio(["None", "90 Degrees", "180 Degrees"], label="Rotation (Special)", value="None")
        special_direction = gr.Radio(["None", "Clockwise", "Anticlockwise"], label="Direction (Special)", value="None")
        special_flip_option = gr.Checkbox(label="Flip Horizontal (Special)", value=False)

    with gr.Row():
        # Zero padding controls
        snap_to_top = gr.Checkbox(label="Snap to Top (Zero Top Padding)", value=False)
        snap_to_bottom = gr.Checkbox(label="Snap to Bottom (Zero Bottom Padding)", value=False)
        snap_to_left = gr.Checkbox(label="Snap to Left (Zero Left Padding)", value=False)
        snap_to_right = gr.Checkbox(label="Snap to Right (Zero Right Padding)", value=False)
        auto_snap = gr.Checkbox(
            label="Auto Snap Detection", 
            value=False
        )
        
        # Hidden compatibility variables
        use_old_position = gr.Checkbox(label="Use Old Position Logic", value=True, visible=False)
        zero_padding = gr.CheckboxGroup(choices=["top", "right", "bottom", "left", "all"], value=[], visible=False)
        auto_detect_padding = gr.Checkbox(label="Auto-Detect Zero Padding", value=False, visible=False)
        
    proc_btn = gr.Button("Process Images")
    stop_btn = gr.Button("Stop")
    
    with gr.Row():
        gallery_processed = gr.Gallery(label="Processed Images")
        
    with gr.Row():
        selected_info = gr.Textbox(label="Selected Image Classification and Processing Info", lines=2, interactive=False)
        
    with gr.Row():
        img_orig = gr.Image(label="Original", interactive=False)
        img_proc = gr.Image(label="Processed", interactive=False)
        
    with gr.Row():
        ratio_orig = gr.Textbox(label="Original Ratio")
        ratio_proc = gr.Textbox(label="Processed Ratio")
        
    with gr.Row():
        out_zip = gr.File(label="Download as ZIP")
        time_box = gr.Textbox(label="Processing Time (seconds)")
        classifications_state = gr.State()
        
    with gr.Row():
        class_display = gr.Textbox(label="All Classification and Processing Results", lines=5, interactive=False)

    bg_choice.change(show_color_picker, inputs=bg_choice, outputs=custom_color)

    proc_btn.click(fn=process,
                   inputs=[input_files, bg_method, watermark, twibbon, canvas_size, output_format,
                           bg_choice, custom_color, num_workers, rotation, direction, flip_option,
                           use_old_position, zero_padding, auto_detect_padding, special_input_files, special_rotation, 
                           special_direction, special_flip_option, no_bg_removal_files, sheet_file, use_qwen,
                           snap_to_bottom, snap_to_top, snap_to_left, snap_to_right, auto_snap],
                   outputs=[gallery_processed, out_zip, time_box, class_display, classifications_state])

    gallery_processed.select(update_compare, 
                            inputs=[classifications_state],
                            outputs=[img_orig, img_proc, ratio_orig, ratio_proc, selected_info])

    stop_btn.click(fn=stop_processing, outputs=[])

iface.launch(share=True)