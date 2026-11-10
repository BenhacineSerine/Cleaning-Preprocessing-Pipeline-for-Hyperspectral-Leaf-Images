"""
STEP 4: Applying the RGB-derived masks to the hyperspectral images (HSI).

The RGB images cleaned in STEP 3 (background removed, white illumination
blanked) are used as binary masks. For every HSI cube, this script:
    1. finds the RGB image with the same file name,
    2. builds a binary mask (non-black pixel = leaf pixel to keep),
    3. checks that the mask and the cube have the same spatial size,
    4. sets all spectral bands to 0 outside the mask,
    5. saves the cleaned cube as ENVI (.hdr + .dat) and as NumPy (.npy),
       keeping the input folder structure.
"""

from pathlib import Path
import spectral
import spectral.io.envi as envi
import numpy as np
from PIL import Image


def run():
    """
    Apply the cleaned RGB masks to all HSI cubes of both categories.

    HSI cubes without a matching RGB image are counted and skipped; images
    that fail are logged and skipped, so one problematic file does not
    stop the whole batch. Nothing is returned; results are reported
    through the printed summary.
    """

    print("=" * 65)
    print("STEP 4: Applying RGB Masks to Hyperspectral Images")
    print("=" * 65)

    # --------------------------------------------------
    # Directories
    # --------------------------------------------------

    # Root directory of the original HSI cubes (ENVI .hdr + data files).
    # Expected layout: HDR_ROOT/<category>/**/*.hdr
    HDR_ROOT = Path("/data/Original_Dataset")

    # RGB images cleaned in STEP 3, used as masks.
    # Expected layout: PNG_ROOT/<category>/**/*.(png|jpg|jpeg)
    PNG_ROOT = Path("/results/RGB_Without_White_Illumination")

    # Output directory: cleaned HSI cubes, same <category>/<subfolder> layout
    OUTPUT_ROOT = Path("/results/Cleaned_HSI")

    # Class folders to process (one sub-directory per class)
    categories = ["healthy", "unhealthy"]

    # Accepted RGB image formats (compared in lowercase)
    extensions = {".png", ".jpg", ".jpeg"}

    # --------------------------------------------------
    # Check main directories
    # --------------------------------------------------

    # Stop immediately if an input root is missing: nothing can be processed
    if not HDR_ROOT.exists():
        print(f"ERROR: HSI directory not found: {HDR_ROOT}")
        return

    if not PNG_ROOT.exists():
        print(f"ERROR: RGB directory not found: {PNG_ROOT}")
        return

    # --------------------------------------------------
    # Global statistics
    # --------------------------------------------------

    # Global counters used in the final report
    total_processed = 0
    total_missing = 0
    total_errors = 0

    # --------------------------------------------------
    # Process both categories
    # --------------------------------------------------

    for category in categories:

        print("\n" + "=" * 65)
        print(f"CATEGORY: {category.upper()}")
        print("=" * 65)

        # Input directories (HSI cubes and their RGB masks)
        hdr_dir = HDR_ROOT / category
        png_dir = PNG_ROOT / category

        # Output directory for this category
        output_dir = OUTPUT_ROOT / category

        # A missing category folder is not fatal: skip it and continue
        if not hdr_dir.exists():
            print(f"WARNING: HSI folder not found: {hdr_dir}")
            continue

        # Without RGB masks the whole category is skipped
        # (counted once in the missing statistics)
        if not png_dir.exists():
            print(f"WARNING: RGB folder not found: {png_dir}")
            total_missing += 1
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

        # ----------------------------------------------
        # Index RGB images by filename
        # ----------------------------------------------

        # Recursive search of all RGB images of the category
        png_files = sorted(
            p for p in png_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in extensions
        )

        # Dictionary {lowercase file name without extension: path}, so that
        # each HSI cube finds its RGB mask instantly. The pairing relies on
        # identical base names between HSI and RGB files.
        png_index = {
            p.stem.lower(): p for p in png_files
        }

        print(f"HSI images found: {len(hdr_files)}")
        print(f"RGB images found: {len(png_files)}")

        # Per-category counters
        category_processed = 0
        category_missing = 0
        category_errors = 0

        # ----------------------------------------------
        # Process each HSI image
        # ----------------------------------------------

        for index, hdr_path in enumerate(hdr_files, start=1):

            # Each cube has its own try/except block, so a corrupted
            # file is logged and skipped instead of stopping the batch.
            try:

                base_name = hdr_path.stem

                print(
                    f"\n[{category}] "
                    f"Processing {index}/{len(hdr_files)}: {base_name}"
                )

                # --------------------------------------
                # Find corresponding RGB image
                # --------------------------------------

                # Case-insensitive lookup of the matching RGB image
                png_path = png_index.get(base_name.lower())

                # No matching RGB image: skip this cube (not an error)
                if png_path is None:

                    category_missing += 1
                    total_missing += 1

                    print(
                        f"WARNING: RGB image not found for {base_name}"
                    )
                    continue

                # --------------------------------------
                # Load hyperspectral cube
                # --------------------------------------

                # The .hdr file describes the cube; the spectral library
                # locates the associated binary data file automatically.
                cube = spectral.open_image(str(hdr_path))

                # Load the full cube into memory (height, width, bands)
                img = cube.load()

                print(f"HSI cube shape: {img.shape}")

                # --------------------------------------
                # Load RGB image and generate mask
                # --------------------------------------

                # Convert to grayscale (single channel) so that a pixel
                # can be tested with one value
                with Image.open(png_path) as rgb_img:

                    mask_img = np.array(
                        rgb_img.convert("L")
                    )

                # Boolean mask: True = pixel kept (non-black in the cleaned
                # RGB image), False = background or illumination removed
                mask = mask_img > 0

                print(f"RGB mask shape: {mask.shape}")

                # --------------------------------------
                # Check dimensions
                # --------------------------------------

                # The mask must be pixel-aligned with the cube: both must
                # have the same (height, width)
                if mask.shape != img.shape[:2]:

                    raise ValueError(
                        f"Dimension mismatch: "
                        f"HSI={img.shape[:2]}, "
                        f"RGB={mask.shape}"
                    )

                # --------------------------------------
                # Apply mask to all spectral bands
                # --------------------------------------

                # Work on a float32 copy so the original cube is untouched
                cube_clean = np.asarray(
                    img, dtype=np.float32
                ).copy()

                # Set every band to 0 for pixels outside the mask
                cube_clean[~mask, :] = 0

                print(
                    f"Mask applied to {cube_clean.shape[2]} bands."
                )

                # --------------------------------------
                # Create output subdirectory
                # --------------------------------------

                # Reproduce the input sub-folder structure (relative to the
                # category folder) in the output directory
                relative_path = hdr_path.relative_to(hdr_dir)

                output_subdir = output_dir / relative_path.parent

                output_subdir.mkdir(
                    parents=True,
                    exist_ok=True
                )

                # --------------------------------------
                # Save ENVI HDR + DAT
                # --------------------------------------

                save_hdr_path = (
                    output_subdir / f"{base_name}.hdr"
                )

                save_dat_path = (
                    output_subdir / f"{base_name}.dat"
                )

                # Reuse the original metadata (wavelengths, sensor info...)
                # so the cleaned cube stays fully described
                metadata = cube.metadata.copy()

                # Write the header (.hdr) and the binary data (.dat).
                # The original interleave is kept (default: bip).
                envi.save_image(
                    str(save_hdr_path),
                    cube_clean,
                    dtype=np.float32,
                    ext="dat",
                    force=True,
                    metadata=metadata,
                    interleave=metadata.get("interleave", "bip")
                )

                # --------------------------------------
                # Save NumPy array
                # --------------------------------------

                # Additional copy in NumPy format for faster loading in the
                # next steps (no ENVI parsing needed)
                save_npy_path = (
                    output_subdir / f"{base_name}.npy"
                )

                np.save(save_npy_path, cube_clean)

                # --------------------------------------
                # Update statistics
                # --------------------------------------

                category_processed += 1
                total_processed += 1

                print(f"SUCCESS: {base_name}")
                print(f"Saved HDR: {save_hdr_path}")
                print(f"Saved DAT: {save_dat_path}")
                print(f"Saved NPY: {save_npy_path}")

            except Exception as e:

                # Log the failure and move on to the next cube
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

        print(f"Processed successfully: {category_processed}")
        print(f"Missing RGB images: {category_missing}")
        print(f"Errors: {category_errors}")

    # --------------------------------------------------
    # Final global report
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("STEP 4 COMPLETED")
    print("=" * 65)

    print(f"Total images processed: {total_processed}")
    print(f"Missing RGB images: {total_missing}")
    print(f"Images with errors: {total_errors}")

    print("\nOutput directory:", OUTPUT_ROOT)

    print("=" * 65)

if __name__ == "__main__":
    run()