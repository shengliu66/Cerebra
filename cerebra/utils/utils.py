import os
import pickle
import base64
import io
from PIL import Image
import numpy as np
import nibabel as nib

def save_to_pickle(data, name, cache_dir, check_exist=False):
    os.makedirs(cache_dir, exist_ok=True)
    save_path = os.path.join(cache_dir, f"{name}.pkl")
    if check_exist:
        if not os.path.exists(save_path): 
            with open(save_path, 'wb') as f:
                pickle.dump(data, f)
    else:
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
    return save_path


def dict_to_xml_str(key, value, indent=0):
    """Convert a dictionary or value to a pretty-printed XML-like string."""
    space = '  ' * indent
    if isinstance(value, dict):
        lines = [f"{space}<{key}>"]
        for k, v in value.items():
            lines.append(dict_to_xml_str(k, v, indent + 1))
        lines.append(f"{space}</{key}>")
        return '\n'.join(lines)
    elif isinstance(value, list):
        lines = [f"{space}<{key}>"]
        for item in value:
            lines.append(dict_to_xml_str('item', item, indent + 1))
        lines.append(f"{space}</{key}>")
        return '\n'.join(lines)
    else:
        value_str = "None" if value is None else str(value)
        return f"{space}<{key}>{value_str}</{key}>"
    

def encode_image_to_raw_bytes(image_array):
    """
    Converts a 2D NumPy array (grayscale image) into a PNG image and returns its raw bytes.

    Parameters:
    - image_array: 2D NumPy array representing a grayscale image.

    Returns:
    - image_grid: Raw bytes of the PNG-encoded image.
    """
    # Convert NumPy array to PIL image (normalize first)
    image = Image.fromarray((image_array * 255).astype('uint8'))  # Scale to [0,255] for saving

    # Save image to a buffer
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")  # Save as PNG (lossless)
    buffer.seek(0)

    image_grid = buffer.getvalue()

    return image_grid

def create_montage(image_data, grid_shape=(8, 4)):
    """
    Convert 3D NIfTI scan into a 2D montage of slices.
    
    Parameters:
    - image_data: 3D NumPy array of the NIfTI scan
    - grid_shape: Tuple (rows, cols) defining the montage shape
    
    Returns:
    - montage: 2D NumPy array of the montage
    """
    num_slices = grid_shape[0] * grid_shape[1]  # Total number of slices needed
    total_slices = image_data.shape[2]  # Number of slices in the scan
    
    # Get evenly spaced slice indices
    slice_indices = np.linspace(0, total_slices - 1, num_slices, dtype=int)

    # Extract slices
    slices = [image_data[:, :, i] for i in slice_indices]
    
    # Normalize slices for better contrast
#     slices = [(s - np.min(s)) / (np.max(s) - np.min(s) + 1e-8) for s in slices]

    # Arrange slices into a grid
    rows, cols = grid_shape
    slice_h, slice_w = slices[0].shape  # Shape of a single slice
    montage = np.zeros((rows * slice_h, cols * slice_w))

    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            montage[i * slice_h:(i + 1) * slice_h, j * slice_w:(j + 1) * slice_w] = slices[idx]

    return montage
def create_montage_from_file(file_name, grid_shape=(6, 4)):
    # Check file extension to determine how to handle the file

    if '.nii' in file_name or '.nii.gz' in file_name:
        # Handle NIfTI files
        img_data = nib.load(file_name).get_fdata()
        return encode_image_to_raw_bytes(create_montage(img_data, grid_shape=grid_shape))
    elif '.png' in file_name or '.jpg' in file_name or '.jpeg' in file_name or '.bmp' in file_name or '.tiff' in file_name or '.tif' in file_name:
        # Handle regular image files (PNG, JPG, etc.)
        image = Image.open(file_name)
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert PIL image to bytes
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return buffer.getvalue()
    else:
        raise ValueError(f"Unsupported file format: {file_name}. Supported formats are NIfTI (.nii, .nii.gz) and common image formats (.png, .jpg, .jpeg, .bmp, .tiff, .tif)")