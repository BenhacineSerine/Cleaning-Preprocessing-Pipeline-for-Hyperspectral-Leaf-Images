"""
STEP 2: Background removal on the simulated RGB dataset.

For every RGB image (PNG/JPG/JPEG) found in the "healthy" and "unhealthy"
folders produced by STEP 1, this script:
    1. loads the image and converts it to 3-channel RGB,
    2. removes the background with the U2-Net segmentation model (rembg),
    3. saves the result as a transparent PNG (RGBA), keeping the input
       folder structure and file names.
"""

from pathlib import Path
from rembg import remove, new_session
from PIL import Image


def run():
    """
    Remove the background of all RGB images of both categories.

    Images that fail are logged and skipped, so one problematic file
    does not stop the whole batch. Nothing is returned; results are
    reported through the printed summary.
    """

    print("STEP 2: Removing backgrounds from the RGB dataset")

    # Input dataset: simulated RGB images generated in STEP 1.
    # Expected layout: data_dir/<category>/**/*.(png|jpg|jpeg)
    data_dir = Path("/results/RGB_Dataset")

    # Output dataset: transparent PNGs, same <category>/<subfolder> layout
    output_dir = Path("/results/RGB_BackGround_Removed")

    # Class folders to process (one sub-directory per class)
    categories = ["healthy", "unhealthy"]

    # Accepted image formats (compared in lowercase)
    extensions = {".png", ".jpg", ".jpeg"}

    # Stop immediately if the dataset root is missing: nothing to process
    if not data_dir.exists():
        print(f"ERROR: Dataset directory not found: {data_dir}")
        return

    # Initialize the segmentation model only once: loading it is costly,
    # so the same session is reused for every image.
    print("Loading U2-Net model...")
    session = new_session("u2net")

    # Global counters used in the final report
    total_processed = 0
    total_errors = 0

    for category in categories:
        input_dir = data_dir / category
        category_output_dir = output_dir / category

        # A missing category folder is not fatal: skip it and continue
        if not input_dir.exists():
            print(f"WARNING: Folder not found: {input_dir}")
            continue

        category_output_dir.mkdir(parents=True, exist_ok=True)

        # Recursive search of all supported images. Sorting guarantees
        # a reproducible processing order.
        image_files = sorted(
            p for p in input_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in extensions
        )

        print(f"\nCategory: {category} | Images: {len(image_files)}")

        for index, input_path in enumerate(image_files, start=1):
            # Each image has its own try/except block, so a corrupted
            # file is logged and skipped instead of stopping the batch.
            try:
                # Load image and force 3-channel RGB (drops any alpha or
                # palette mode) so the model always receives the same format
                with Image.open(input_path) as img:
                    input_image = img.convert("RGB")

                # Remove background: the model predicts a foreground mask
                # and applies it as the alpha channel of the output
                output_image = remove(input_image, session=session)
                output_image = output_image.convert("RGBA")

                # Preserve relative subfolders and file name; the extension
                # is forced to .png because transparency needs a PNG file
                relative_path = input_path.relative_to(input_dir)
                output_path = category_output_dir / relative_path.with_suffix(".png")
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Save as lossless transparent PNG
                output_image.save(output_path, format="PNG")

                total_processed += 1

                # Progress report every 10 images and on the last one
                if index % 10 == 0 or index == len(image_files):
                    print(
                        f"{category}: {index}/{len(image_files)} processed"
                    )

            except Exception as e:
                # Log the failure and move on to the next image
                total_errors += 1
                print(f"ERROR processing {input_path}: {e}")

    # Final report
    print("\nSTEP 2 completed.")
    print(f"Images processed successfully: {total_processed}")
    print(f"Images with errors: {total_errors}")
    print(f"Output directory: {output_dir}")


# Run the step when the file is executed directly (not when imported)
if __name__ == "__main__":
    run()