import pydicom
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import measure
from PIL import Image

# --- DICOM Analysis & Conversion ---
# This function's core logic is unchanged.
def load_dicom_as_nifti(dcm_path):
    ds = pydicom.dcmread(dcm_path)
    pixel_array = ds.pixel_array
    if pixel_array.ndim == 3:
        dicom_volume = pixel_array
    else:
        dicom_volume = np.expand_dims(pixel_array, axis=0)
    dicom_volume = np.transpose(dicom_volume, (2, 1, 0))
    dicom_volume = np.flip(dicom_volume, axis=1)
    print(f"[DICOM] Loaded volume shape for conversion: {dicom_volume.shape}")
    depth_elem = ds.get((0x0022, 0x0035), None)
    spacing_z = float(depth_elem.value) if depth_elem else 1.0
    along_elem = ds.get((0x0022, 0x0037), None)
    spacing_x = float(along_elem.value) if along_elem else 1.0
    across_elem = ds.get((0x0022, 0x0048), None)
    spacing_y = float(across_elem.value) if across_elem else 1.0
    affine = np.diag([spacing_y, spacing_x, spacing_z, 1])
    nifti_img = nib.Nifti1Image(dicom_volume, affine)
    return dicom_volume, nifti_img

# --- NIfTI Analysis ---
# This function's core logic is unchanged.
def load_nifti_mask(nifti_path):
    nii_img = nib.load(nifti_path)
    nii_data = nii_img.get_fdata()
    print(f"[NIfTI] ROI mask shape: {nii_data.shape}")
    return nii_data

# --- Save overlay images with 90° CCW rotation ---
# MODIFIED: Accepts a 'base_output_dir' to create a specific subdirectory for overlays.
def save_overlay_slices(dicom_vol, roi_mask, base_output_dir, start_slice=0, end_slice=5, alpha=0.3):
    output_dir = os.path.join(base_output_dir, "Overlays")
    os.makedirs(output_dir, exist_ok=True)
    num_slices = dicom_vol.shape[2]
    end_slice = min(end_slice, num_slices)
    for i in range(start_slice, end_slice):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        img_rot = np.rot90(dicom_vol[:, :, i])
        mask_rot = np.rot90(roi_mask[:, :, i] > 0.5)
        axs[0].imshow(img_rot, cmap='gray')
        axs[0].set_title(f"Original Slice {i}")
        axs[0].axis('off')
        axs[1].imshow(img_rot, cmap='gray')
        axs[1].imshow(mask_rot, cmap='Reds', alpha=alpha)
        contours = measure.find_contours(mask_rot, 0.5)
        for contour in contours:
            axs[1].plot(contour[:, 1], contour[:, 0], linewidth=1.5, color='red')
        axs[1].set_title(f"Overlay Slice {i}")
        axs[1].axis('off')
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"slice_{i:03d}_overlay.png")
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
    print(f"[INFO] Saved overlay images to: {output_dir}")

# --- Save separate DICOM and mask slices ---
# MODIFIED: Accepts a 'base_output_dir' to create specific subdirectories for slices and masks.
def save_individual_slices(dicom_vol, roi_mask, base_output_dir, start_slice=0, end_slice=5):
    img_dir = os.path.join(base_output_dir, "DICOM_Slices")
    mask_dir = os.path.join(base_output_dir, "Binary_Masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    num_slices = dicom_vol.shape[2]
    end_slice = min(end_slice, num_slices)
    for i in range(start_slice, end_slice):
        img_rot = np.rot90(dicom_vol[:, :, i])
        mask_bin = (np.rot90(roi_mask[:, :, i]) > 0.5).astype(np.uint8) * 255 # Multiply by 255 to make mask visible
        img_out = Image.fromarray(img_rot.astype(np.uint8))
        img_out.save(os.path.join(img_dir, f"slice_{i:03d}.png"))
        mask_out = Image.fromarray(mask_bin)
        mask_out.save(os.path.join(mask_dir, f"mask_{i:03d}.png"))
    print(f"[INFO] Saved DICOM slices to: {img_dir}")
    print(f"[INFO] Saved binary masks to: {mask_dir}")

# --- Main Processing Logic ---
def main():
    # --- Define Folder Paths ---
    DICOM_DIR = "/home/shared/projects/HRF_Reeva/input_downloads_testset"
    SEG_DIR = "/home/shared/projects/HRF_Reeva/output_downloads_testset"
    OUTPUT_DIR = "/home/shared/projects/HRF_Reeva/pre_processed_testset"

    # Create the main output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get lists of all files to match
    dicom_files = [f for f in os.listdir(DICOM_DIR) if f.endswith('.dcm.zip')]
    print(f"[INFO] Found {len(dicom_files)} DICOM files in {DICOM_DIR}.")
    seg_files = os.listdir(SEG_DIR)

    # Loop through each DICOM file and find its corresponding segmentation
    for dcm_filename in dicom_files:
        print(f"\n{'='*50}\nProcessing DICOM file: {dcm_filename}\n{'='*50}")

        # MODIFIED LOGIC: Use the filename *without the extension* for matching
        dcm_stem = os.path.splitext(dcm_filename)[0]
        dcm_stem = dcm_stem.replace('.dcm', '')
        dcm_stem = dcm_stem.replace(' + ', '_')
        

        # Find the matching segmentation file
        matching_seg_file = None
        for seg_filename in seg_files:
            # Strip .nii.gz or .nii before comparing
            seg_stem = seg_filename.replace('.nii.gz', '').replace('.nii', '')
            print(f"[DEBUG] Comparing DICOM stem '{dcm_stem}' with segmentation stem '{seg_stem}'")
            if dcm_stem in seg_stem:
                matching_seg_file = seg_filename
                break
        
        if not matching_seg_file:
            print(f"[WARNING] No matching segmentation found for {dcm_filename}. Skipping.")
            continue
        
        print(f"[INFO] Found matching segmentation: {matching_seg_file}")

        # --- Define Full File and Directory Paths for this pair ---
        dcm_path = os.path.join(DICOM_DIR, dcm_filename)
        nifti_path = os.path.join(SEG_DIR, matching_seg_file)
        
        # Create a unique sub-directory for this case's output
        case_output_dir = os.path.join(OUTPUT_DIR, dcm_stem) # Use the stem for the folder name
        os.makedirs(case_output_dir, exist_ok=True)
        
        # --- Run the processing pipeline ---
        try:
            dicom_vol, _ = load_dicom_as_nifti(dcm_path)
            roi_mask = load_nifti_mask(nifti_path)
            
            # Process all slices found in the DICOM volume
            num_slices = dicom_vol.shape[2]
            
            # Call the saving functions, passing the unique output directory
            save_overlay_slices(dicom_vol, roi_mask, case_output_dir, start_slice=0, end_slice=num_slices)
            print(f"[INFO] Overlay images saved for {dcm_filename} and {matching_seg_file}.")
            save_individual_slices(dicom_vol, roi_mask, case_output_dir, start_slice=0, end_slice=num_slices)
            
        except Exception as e:
            print(f"[ERROR] Failed to process the pair {dcm_filename} and {matching_seg_file}. Reason: {e}")

    print(f"\n{'='*50}\n Batch processing complete. Check the '{OUTPUT_DIR}' folder.\n{'='*50}")

# --- Run the script ---
if __name__ == "__main__":
    main()