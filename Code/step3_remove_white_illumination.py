"""
STEP 3: Removal of the white illumination reflection (template matching).

For every background-removed image produced by STEP 2, this script:
    1. converts the image to grayscale,
    2. searches for the white illumination spot using a set of grayscale
       templates (normalized cross-correlation, OpenCV template matching),
    3. keeps the best match across all templates,
    4. if the best score reaches THRESHOLD, paints the matched rectangle
       black so that the reflection does not disturb later analysis,
    5. saves the processed image, keeping the input folder structure.
"""

from pathlib import Path
import cv2
import numpy as np


def run():
    """
    Detect and blank out the white illumination in all images of both
    categories.

    Images that fail are logged and skipped, so one problematic file
    does not stop the whole batch. Nothing is returned; results are
    reported through the printed summary.
    """

    print("=" * 60)
    print("STEP 3: Removing White Illumination")
    print("=" * 60)

    # --------------------------------------------------
    # Directories
    # --------------------------------------------------

    # Input images: background-removed RGB images generated in STEP 2.
    # Expected layout: data_dir/<category>/**/*.(png|jpg|jpeg)
    data_dir = Path("/results/RGB_BackGround_Removed")

    # Folder containing the reference templates of the white illumination
    # spot (attached to the project as a separate dataset)
    template_dir = Path("/data/Templates_Matching")

    # Output directory: processed images, same <category>/<subfolder> layout
    output_dir = Path("/results/RGB_Without_White_Illumination")

    # Class folders to process (one sub-directory per class)
    categories = ["healthy", "unhealthy"]

    # Accepted image formats (compared in lowercase)
    extensions = {".png", ".jpg", ".jpeg"}

    # Minimum matching score (TM_CCOEFF_NORMED, range [-1, 1]) required to
    # accept a detection. Below this value the image is left untouched, which
    # avoids erasing leaf regions by mistake.
    THRESHOLD = 0.75

    # --------------------------------------------------
    # Check directories
    # --------------------------------------------------

    # Stop immediately if the inputs are missing: nothing can be processed
    if not data_dir.exists():
        print(f"ERROR: Input directory not found: {data_dir}")
        return

    if not template_dir.exists():
        print(f"ERROR: Template directory not found: {template_dir}")
        return

    # --------------------------------------------------
    # Load templates only once
    # --------------------------------------------------

    # Templates are read once and reused for every image, which is much
    # faster than reloading them inside the image loop.
    print("\nLoading illumination templates...")

    templates = []

    for tpl_path in sorted(template_dir.rglob("*")):

        if tpl_path.is_file() and tpl_path.suffix.lower() in extensions:

            # Templates are loaded in grayscale because the matching is
            # done on grayscale images (intensity pattern only)
            tpl = cv2.imread(str(tpl_path), cv2.IMREAD_GRAYSCALE)

            # An unreadable template is skipped, not fatal
            if tpl is None:
                print(f"WARNING: Cannot load template: {tpl_path}")
                continue

            templates.append((tpl_path.name, tpl))

            print(f"Template loaded: {tpl_path.name} {tpl.shape}")

    # Without any valid template, detection is impossible
    if not templates:
        print("ERROR: No valid templates found.")
        return

    print(f"\nTotal templates loaded: {len(templates)}")

    # --------------------------------------------------
    # Processing statistics
    # --------------------------------------------------

    # Global counters used in the final report
    total_processed = 0
    total_detected = 0
    total_not_detected = 0
    total_errors = 0

    # --------------------------------------------------
    # Process healthy and unhealthy images
    # --------------------------------------------------

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

        print(f"\n{'-' * 50}")
        print(f"Category: {category}")
        print(f"Images found: {len(image_files)}")
        print(f"{'-' * 50}")

        for index, input_path in enumerate(image_files, start=1):

            # Each image has its own try/except block, so a corrupted
            # file is logged and skipped instead of stopping the batch.
            try:

                # --------------------------------------
                # Load image
                # --------------------------------------

                # Color image (BGR order, OpenCV convention): this is the
                # image that will be modified and saved
                img_color = cv2.imread(str(input_path))

                if img_color is None:
                    raise ValueError("Unable to read image.")

                # Grayscale copy used only for template matching
                img_gray = cv2.cvtColor(
                    img_color, cv2.COLOR_BGR2GRAY
                )

                # Best match found so far across all templates
                # (-1 is the lowest possible correlation score)
                best_val = -1
                best_tpl = None
                best_loc = None
                best_size = None

                # --------------------------------------
                # Template matching
                # --------------------------------------

                # Every template is tested on the image and only the one
                # with the highest correlation score is kept.
                for tpl_name, tpl in templates:

                    h, w = tpl.shape[:2]

                    # Skip templates larger than the image (OpenCV cannot
                    # slide a template bigger than the image)
                    if h > img_gray.shape[0] or w > img_gray.shape[1]:
                        continue

                    # Normalized correlation map: one score per position
                    # of the template over the image
                    res = cv2.matchTemplate(
                        img_gray,
                        tpl,
                        cv2.TM_CCOEFF_NORMED
                    )

                    # Keep the highest score and its position
                    # (top-left corner of the best-matching region)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)

                    # Ignore invalid matching scores (NaN/Inf can occur on
                    # flat regions) and keep the best valid score
                    if np.isfinite(max_val) and max_val > best_val:

                        best_val = max_val
                        best_tpl = tpl_name
                        best_loc = max_loc
                        best_size = (w, h)

                # --------------------------------------
                # Remove detected illumination
                # --------------------------------------

                # Remove the region only if the match is reliable enough
                if best_val >= THRESHOLD and best_loc is not None:

                    w, h = best_size
                    x, y = best_loc

                    # Replace detected region with black (BGR = 0, 0, 0)
                    img_color[y:y+h, x:x+w] = (0, 0, 0)

                    total_detected += 1

                    print(
                        f"{category} | {input_path.name} -> "
                        f"Detected: {best_tpl} "
                        f"(confidence={best_val:.2f})"
                    )

                else:

                    # Below threshold: the image is saved unchanged
                    total_not_detected += 1

                    print(
                        f"{category} | {input_path.name} -> "
                        f"No illumination detected "
                        f"(best score={best_val:.2f})"
                    )

                # --------------------------------------
                # Save processed image
                # --------------------------------------

                # Reproduce the input sub-folder structure (relative to the
                # category folder) and keep the original file name
                relative_path = input_path.relative_to(input_dir)

                output_path = (
                    category_output_dir / relative_path
                )

                output_path.parent.mkdir(
                    parents=True, exist_ok=True
                )

                # imwrite returns False (without raising) if saving fails
                success = cv2.imwrite(str(output_path), img_color)

                if not success:
                    raise ValueError("Unable to save output image.")

                total_processed += 1

                # Progress report every 10 images and on the last one
                if index % 10 == 0 or index == len(image_files):

                    print(
                        f"{category}: {index}/{len(image_files)} "
                        f"images processed"
                    )

            except Exception as e:

                # Log the failure and move on to the next image
                total_errors += 1

                print(
                    f"ERROR processing {input_path}: {e}"
                )

    # --------------------------------------------------
    # Final report
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("STEP 3 COMPLETED")
    print("=" * 60)

    print(f"Images processed successfully: {total_processed}")
    print(f"Illumination detected and removed: {total_detected}")
    print(f"No illumination detected: {total_not_detected}")
    print(f"Images with errors: {total_errors}")

    print(f"\nOutput directory: {output_dir}")

    print("=" * 60)

if __name__ == "__main__":
    run()