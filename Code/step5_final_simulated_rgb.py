"""
STEP 5: Generation of the final simulated RGB images from the cleaned HSI.

Same procedure as STEP 1, but applied to the cleaned cubes produced by
STEP 4 (background and white illumination already set to 0). For every
cleaned hyperspectral cube (ENVI format: .hdr + data file), this script:
    1. loads the cube and checks that it is a valid 3D array (H x W x bands),
    2. extracts three spectral bands used as Red, Green and Blue channels,
    3. masks invalid pixels (NaN/Inf or empty background),
    4. normalizes the three channels with ONE shared intensity range,
       so that the relative color balance between bands is preserved,
    5. saves the result as an 8-bit PNG, keeping the input folder structure,
    6. displays a few examples per category for visual quality control.
"""

from pathlib import Path
import spectral
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def run():
    """
    Convert all cleaned HSI cubes of both categories into final simulated
    RGB images.

    Returns
    -------
    bool
        True if every image was processed without error, False otherwise
        (also False if the input directory is missing).
    """

    print("=" * 65)
    print("STEP 5: Generating and Displaying Final Simulated RGB Images")
    print("=" * 65)

    # --------------------------------------------------
    # Directories
    # --------------------------------------------------

    # Root folder of the cleaned HSI dataset (output of STEP 4).
    # Expected layout: HDR_ROOT/<category>/**/*.hdr
    HDR_ROOT = Path("/results/Cleaned_HSI")

    # Root folder where the final simulated RGB images will be written.
    # The same <category>/<subfolder> structure is reproduced here.
    OUTPUT_ROOT = Path("/results/Simulated_Cleaned_RGB")

    # Class folders to process (one sub-directory per class)
    categories = ["healthy", "unhealthy"]

    # --------------------------------------------------
    # RGB band selection (0-based indexing)
    # --------------------------------------------------

    # Indices of the spectral bands that approximate the three color channels
    # (same bands as in STEP 1, so both RGB versions are comparable).
    # The wavelength of each band is given for reference.
    R_BAND = 90    # 660.84 nm - Red
    G_BAND = 52    # 548.55 nm - Green
    B_BAND = 18    # 449.35 nm - Blue

    # Maximum number of images shown on screen per category
    # (visual check only; all images are saved regardless of this limit)
    MAX_DISPLAY_PER_CATEGORY = 5

    # --------------------------------------------------
    # Check input directory
    # --------------------------------------------------

    # Stop immediately if the dataset root is missing: nothing can be processed
    if not HDR_ROOT.exists():
        print(f"ERROR: HSI directory not found: {HDR_ROOT}")
        return False

    # Global counters used in the final report
    total_processed = 0
    total_errors = 0

    # --------------------------------------------------
    # Process both categories
    # --------------------------------------------------

    for category in categories:

        print("\n" + "=" * 65)
        print(f"CATEGORY: {category.upper()}")
        print("=" * 65)

        hdr_dir = HDR_ROOT / category
        output_dir = OUTPUT_ROOT / category

        # A missing category folder is not fatal: skip it and continue
        if not hdr_dir.exists():
            print(f"WARNING: Folder not found: {hdr_dir}")
            continue

        output_dir.mkdir(parents=True, exist_ok=True)

        # ----------------------------------------------
        # Find HDR files
        # ----------------------------------------------

        # Recursive search of all ENVI header files (case-insensitive
        # extension). Sorting guarantees a reproducible processing order.
        hdr_files = sorted(
            p for p in hdr_dir.rglob("*")
            if p.is_file() and p.suffix.lower() == ".hdr"
        )

        print(f"HDR images found: {len(hdr_files)}")

        # Per-category counters
        category_processed = 0
        category_errors = 0
        displayed = 0

        # ----------------------------------------------
        # Process each HSI image
        # ----------------------------------------------

        for index, hdr_path in enumerate(hdr_files, start=1):

            # Each image is processed in its own try/except block, so one
            # corrupted cube does not interrupt the whole batch.
            try:

                base_name = hdr_path.stem

                print(
                    f"\n[{category}] "
                    f"Processing {index}/{len(hdr_files)}: {base_name}"
                )

                # --------------------------------------
                # Load cleaned HSI cube (.hdr + .dat)
                # --------------------------------------

                # The .hdr file describes the cube; the spectral library
                # locates the associated binary data file automatically.
                cube = spectral.open_image(str(hdr_path))

                # Load the full cube into memory as a NumPy array
                img = np.asarray(cube.load())

                print(f"Original HSI shape: {img.shape}")

                # Remove extra singleton dimensions (e.g. shape (1, H, W, B))
                if img.ndim > 3:
                    img = np.squeeze(img)

                # After cleanup, the cube must be (height, width, bands)
                if img.ndim != 3:

                    raise ValueError(
                        f"Expected a 3D HSI cube, "
                        f"received shape: {img.shape}"
                    )

                height, width, bands = img.shape

                print(f"Corrected HSI shape: {img.shape}")

                # --------------------------------------
                # Check selected spectral bands
                # --------------------------------------

                # Make sure the chosen band indices exist in this cube
                if max(R_BAND, G_BAND, B_BAND) >= bands:

                    raise ValueError(
                        f"Selected RGB bands exceed available "
                        f"bands: {bands}"
                    )

                # --------------------------------------
                # Extract RGB channels
                # --------------------------------------

                # Each channel is a 2D (height, width) float32 image taken
                # from one spectral band of the cube.
                red = np.squeeze(
                    np.asarray(img[:, :, R_BAND], dtype=np.float32)
                )

                green = np.squeeze(
                    np.asarray(img[:, :, G_BAND], dtype=np.float32)
                )

                blue = np.squeeze(
                    np.asarray(img[:, :, B_BAND], dtype=np.float32)
                )

                # Sanity check: the three channels must have identical
                # spatial dimensions, otherwise they cannot be stacked
                if not (
                    red.shape == green.shape == blue.shape
                    == (height, width)
                ):

                    raise ValueError(
                        f"RGB channel dimensions do not match: "
                        f"R={red.shape}, G={green.shape}, "
                        f"B={blue.shape}"
                    )

                print(f"RGB channel shape: {red.shape}")

                # --------------------------------------
                # Create valid pixel mask
                # --------------------------------------

                # A pixel is valid if all three values are finite (no NaN/Inf)
                # AND at least one of them is non-zero. Pixels that are zero
                # in all three bands (background and illumination removed in
                # STEP 4) are treated as empty and excluded from the
                # normalization statistics.
                valid_mask = (
                    np.isfinite(red) &
                    np.isfinite(green) &
                    np.isfinite(blue) &
                    (
                        (red != 0) |
                        (green != 0) |
                        (blue != 0)
                    )
                )

                valid_pixels = np.sum(valid_mask)

                print(f"Valid pixels: {valid_pixels}")

                # An image without any valid pixel cannot be normalized
                if valid_pixels == 0:
                    raise ValueError("No valid RGB pixels found.")

                # --------------------------------------
                # COMMON NORMALIZATION FOR RGB CHANNELS
                # --------------------------------------

                # Pool the valid values of the three channels together so
                # that a single intensity range can be estimated. Using one
                # shared range (instead of one per channel) keeps the
                # relative reflectance between R, G and B, and therefore
                # the natural color balance of the simulated image.
                rgb_values = np.concatenate([
                    red[valid_mask],
                    green[valid_mask],
                    blue[valid_mask]
                ])

                # Extra safety: discard any remaining non-finite value
                rgb_values = rgb_values[
                    np.isfinite(rgb_values)
                ]

                if rgb_values.size == 0:
                    raise ValueError(
                        "No finite RGB values available."
                    )

                # Robust intensity limits: the 2nd and 98th percentiles
                # ignore extreme outliers (e.g. saturated or dead pixels)
                # that would otherwise compress the contrast.
                low, high = np.percentile(
                    rgb_values, [2, 98]
                )

                # Fallback to the absolute min/max if the percentile range
                # is degenerate (e.g. almost constant image)
                if high <= low:

                    low = float(rgb_values.min())
                    high = float(rgb_values.max())

                # If the range is still degenerate, normalization is
                # impossible (constant image)
                if high <= low:
                    raise ValueError(
                        "Invalid RGB normalization range."
                    )

                print(
                    f"Common normalization limits: "
                    f"low={low:.6f}, high={high:.6f}"
                )

                # --------------------------------------
                # Apply the same normalization to all
                # three RGB channels
                # --------------------------------------

                def normalize_common(band):
                    """
                    Linearly rescale a band to [0, 1] using the shared
                    (low, high) limits. Values outside the limits are
                    clipped; invalid pixels are left at 0 (black).
                    """

                    normalized = np.zeros(
                        band.shape, dtype=np.float32
                    )

                    normalized[valid_mask] = np.clip(
                        (band[valid_mask] - low) / (high - low),
                        0,
                        1
                    )

                    return normalized

                R = normalize_common(red)
                G = normalize_common(green)
                B = normalize_common(blue)

                # --------------------------------------
                # Generate simulated RGB image
                # --------------------------------------

                # Stack the channels along the last axis -> (H, W, 3)
                rgb_image = np.stack(
                    [R, G, B],
                    axis=-1
                )

                # Convert from float [0, 1] to 8-bit integers [0, 255]
                # (standard format for PNG images)
                rgb_uint8 = (
                    rgb_image * 255
                ).astype(np.uint8)

                print(
                    f"Final RGB shape: {rgb_uint8.shape}"
                )

                # --------------------------------------
                # Preserve output subdirectories
                # --------------------------------------

                # Reproduce the input sub-folder structure (relative to the
                # category folder) in the output directory
                relative_path = hdr_path.relative_to(hdr_dir)

                output_subdir = (
                    output_dir / relative_path.parent
                )

                output_subdir.mkdir(
                    parents=True,
                    exist_ok=True
                )

                # --------------------------------------
                # Save simulated RGB image
                # --------------------------------------

                # PNG is lossless, so no compression artifacts are added
                # to the final images
                output_path = (
                    output_subdir / f"{base_name}.png"
                )

                Image.fromarray(rgb_uint8).save(output_path)

                print(f"Saved RGB: {output_path}")

                # --------------------------------------
                # Display RGB image
                # --------------------------------------

                # Show only the first few images of each category as a
                # quick visual quality check
                if displayed < MAX_DISPLAY_PER_CATEGORY:

                    plt.figure(figsize=(8, 6))

                    plt.imshow(rgb_uint8)

                    plt.title(
                        f"{category.capitalize()} - {base_name}\n"
                        f"Simulated RGB "
                        f"(R=660.84 nm, "
                        f"G=548.55 nm, "
                        f"B=449.35 nm)"
                    )

                    plt.axis("off")

                    plt.tight_layout()

                    plt.show()

                    # Release the figure to avoid memory build-up
                    plt.close()

                    displayed += 1

                # --------------------------------------
                # Update statistics
                # --------------------------------------

                category_processed += 1
                total_processed += 1

                print(f"SUCCESS: {base_name}")

            except Exception as e:

                # Log the failure and move on to the next image
                category_errors += 1
                total_errors += 1

                print(
                    f"ERROR processing {hdr_path.name}: {e}"
                )

        # ----------------------------------------------
        # Category summary
        # ----------------------------------------------

        print("\n" + "-" * 55)
        print(f"{category.upper()} SUMMARY")
        print("-" * 55)

        print(f"Images processed: {category_processed}")
        print(f"Images displayed: {displayed}")
        print(f"Errors: {category_errors}")

    # --------------------------------------------------
    # Final report
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("STEP 5 COMPLETED")
    print("=" * 65)

    print(f"Total images processed: {total_processed}")
    print(f"Total errors: {total_errors}")

    print(f"\nOutput directory: {OUTPUT_ROOT}")

    print("=" * 65)

    # The step is considered successful only if no image failed
    return total_errors == 0

if __name__ == "__main__":
    run()