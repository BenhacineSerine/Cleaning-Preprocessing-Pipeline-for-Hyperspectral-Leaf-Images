"""
STEP 6: Analysis of the impact of image reduction (cleaning) on the HSI data.

The cleaning pipeline (STEP 2 to STEP 4) removes background and illumination
pixels. This script checks that this reduction does not alter the spectral
information of the leaf. For every pair (original cube, cleaned cube), it:
    1. builds a leaf mask from the cleaned cube (pixels that are non-zero),
    2. counts original vs. retained pixels and the percentage of reduction,
    3. accumulates, per band, the mean and standard deviation of the leaf
       pixels of BOTH cubes using the SAME mask (fair comparison),
    4. produces one figure per category: (a) mean spectra before/after
       cleaning, (b) pixel counts before/after cleaning,
    5. writes a CSV report with the pixel statistics of each image.
"""

from pathlib import Path
import spectral
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import csv


def run():
    """
    Compare original and cleaned HSI cubes for both categories.

    Returns
    -------
    bool
        True if at least one image pair was processed and there was no
        missing cleaned file and no error, False otherwise.
    """

    print("=" * 70)
    print("STEP 6: ANALYZING THE IMPACT OF IMAGE REDUCTION")
    print("=" * 70)

    # --------------------------------------------------
    # Directories
    # --------------------------------------------------

    # Original HSI cubes (before any cleaning)
    ORIGINAL_ROOT = Path("/data/Original_Dataset")

    # Cleaned HSI cubes (output of STEP 4), same layout as the originals
    CLEANED_ROOT = Path("/results/Cleaned_HSI")

    # Output directory: figures and CSV report
    OUTPUT_ROOT = Path("/results/Reduction_Impact")

    # Class folders to process (one sub-directory per class)
    categories = ["healthy", "unhealthy"]

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Extract wavelength information from HDR metadata
    # --------------------------------------------------

    def get_wavelengths(metadata, bands):
        """
        Return the wavelength of each band and the matching x-axis label.

        The wavelengths are read from the ENVI header. If they are missing,
        malformed or do not match the number of bands, the band index
        (1..bands) is used instead so the plots can still be produced.
        """

        raw = metadata.get("wavelength")

        if raw is not None:

            try:

                # ENVI headers may store the list as a string "{a, b, c}"
                if isinstance(raw, str):
                    raw = raw.strip().strip("{}")
                    raw = raw.split(",")

                wavelengths = np.asarray(
                    raw, dtype=np.float64
                ).ravel()

                # Accept the wavelengths only if there is exactly one valid
                # value per band
                if (
                    len(wavelengths) == bands
                    and np.all(np.isfinite(wavelengths))
                ):

                    units = metadata.get(
                        "wavelength units", "nm"
                    )

                    return wavelengths, f"Wavelength ({units})"

            except (ValueError, TypeError):
                # Non-numeric content: fall back to band indices below
                pass

        # Fallback: no usable wavelength information
        print("WARNING: Wavelength metadata unavailable.")
        return np.arange(1, bands + 1), "Spectral Band Index"

    # --------------------------------------------------
    # Accumulate spectral statistics
    # --------------------------------------------------

    def accumulate_statistics(
        cube,
        mask,
        sum_values,
        sum_squares,
        counts
    ):
        """
        Add the leaf pixels of one cube to the running per-band statistics.

        For each band, the arrays sum_values, sum_squares and counts are
        updated IN PLACE with the sum, the sum of squares and the number of
        valid values selected by the mask. Keeping these running sums
        allows the mean and standard deviation of a whole category to be
        computed at the end without keeping all pixels in memory.
        """

        bands = cube.shape[2]

        for b in range(bands):

            # Values of band b for the pixels selected by the mask,
            # in float64 to limit rounding errors in the sums
            values = np.asarray(
                cube[:, :, b][mask],
                dtype=np.float64
            )

            # Ignore NaN/Inf values
            values = values[np.isfinite(values)]

            if values.size == 0:
                continue

            sum_values[b] += np.sum(values)
            sum_squares[b] += np.sum(values ** 2)
            counts[b] += values.size

    # --------------------------------------------------
    # Global statistics
    # --------------------------------------------------

    # Global counters used in the final report
    total_processed = 0
    total_missing = 0
    total_errors = 0

    # One row per image, written to the CSV report at the end
    summary_rows = []

    # --------------------------------------------------
    # Process healthy and unhealthy categories
    # --------------------------------------------------

    for category in categories:

        print("\n" + "=" * 65)
        print(f"CATEGORY: {category.upper()}")
        print("=" * 65)

        original_dir = ORIGINAL_ROOT / category
        cleaned_dir = CLEANED_ROOT / category

        # A missing folder is not fatal: skip the category and continue
        if not original_dir.exists():
            print(f"WARNING: Original folder not found: {original_dir}")
            continue

        if not cleaned_dir.exists():
            print(f"WARNING: Cleaned folder not found: {cleaned_dir}")
            continue

        # Find original HDR files (recursive, case-insensitive extension).
        # Sorting guarantees a reproducible processing order.
        hdr_files = sorted(
            p for p in original_dir.rglob("*")
            if p.is_file() and p.suffix.lower() == ".hdr"
        )

        print(f"Original HDR images found: {len(hdr_files)}")

        # Per-category counters
        category_processed = 0
        category_missing = 0
        category_errors = 0

        # Class-level accumulated statistics (one value per band).
        # They are created when the first valid image pair is read,
        # because the number of bands is only known at that point.
        sum_original = None
        sum_cleaned = None

        squares_original = None
        squares_cleaned = None

        counts_original = None
        counts_cleaned = None

        wavelengths = None
        xlabel = None
        number_of_bands = None

        # Pixel totals of the whole category (used for the bar chart)
        total_original_pixels = 0
        total_retained_pixels = 0

        # ----------------------------------------------
        # Process each image pair
        # ----------------------------------------------

        for index, original_path in enumerate(hdr_files, start=1):

            # Each pair has its own try/except block, so a corrupted
            # file is logged and skipped instead of stopping the batch.
            try:

                base_name = original_path.stem

                print(
                    f"\n[{category}] "
                    f"Processing {index}/{len(hdr_files)}: {base_name}"
                )

                # --------------------------------------
                # Find corresponding cleaned HDR
                # --------------------------------------

                # The cleaned cube has the same relative path as the
                # original one (STEP 4 preserves the folder structure)
                relative_path = original_path.relative_to(
                    original_dir
                )

                cleaned_path = cleaned_dir / relative_path

                # No cleaned counterpart: skip this image (not an error)
                if not cleaned_path.exists():

                    category_missing += 1
                    total_missing += 1

                    print(
                        f"WARNING: Cleaned HDR not found: "
                        f"{cleaned_path}"
                    )
                    continue

                # --------------------------------------
                # Load original and cleaned HSI cubes
                # --------------------------------------

                original_cube = spectral.open_image(
                    str(original_path)
                )

                cleaned_cube = spectral.open_image(
                    str(cleaned_path)
                )

                # Load both cubes in memory as float32 NumPy arrays
                original = np.asarray(
                    original_cube.load(),
                    dtype=np.float32
                )

                cleaned = np.asarray(
                    cleaned_cube.load(),
                    dtype=np.float32
                )

                print(f"Original shape: {original.shape}")
                print(f"Cleaned shape: {cleaned.shape}")

                # --------------------------------------
                # Verify dimensions
                # --------------------------------------

                # Both cubes must be pixel-aligned to share the same mask
                if original.shape != cleaned.shape:

                    raise ValueError(
                        f"Cube dimensions do not match: "
                        f"Original={original.shape}, "
                        f"Cleaned={cleaned.shape}"
                    )

                height, width, bands = original.shape

                # --------------------------------------
                # Initialize class statistics
                # --------------------------------------

                # First valid image of the category: allocate the per-band
                # accumulators and read the wavelengths
                if sum_original is None:

                    number_of_bands = bands

                    sum_original = np.zeros(
                        bands, dtype=np.float64
                    )

                    sum_cleaned = np.zeros(
                        bands, dtype=np.float64
                    )

                    squares_original = np.zeros(
                        bands, dtype=np.float64
                    )

                    squares_cleaned = np.zeros(
                        bands, dtype=np.float64
                    )

                    counts_original = np.zeros(
                        bands, dtype=np.int64
                    )

                    counts_cleaned = np.zeros(
                        bands, dtype=np.int64
                    )

                    wavelengths, xlabel = get_wavelengths(
                        original_cube.metadata,
                        bands
                    )

                # Following images: all cubes of a category must have the
                # same number of bands to be accumulated together
                elif bands != number_of_bands:

                    raise ValueError(
                        f"Band count differs from previous images: "
                        f"{bands} vs {number_of_bands}"
                    )

                # --------------------------------------
                # Generate valid leaf mask from cleaned HSI
                # --------------------------------------

                # A pixel belongs to the leaf if at least one band of the
                # cleaned cube is finite and non-zero (removed pixels were
                # set to 0 in all bands in STEP 4)
                valid_mask = np.any(
                    np.isfinite(cleaned) & (cleaned != 0),
                    axis=2
                )

                # Pixel counts and reduction rate for this image
                retained_pixels = int(
                    np.count_nonzero(valid_mask)
                )

                original_pixels = height * width

                removed_pixels = (
                    original_pixels - retained_pixels
                )

                # Protection against division by zero
                reduction_percent = (
                    100 * removed_pixels / original_pixels
                    if original_pixels > 0 else 0
                )

                print(f"Original pixels: {original_pixels}")
                print(f"Retained leaf pixels: {retained_pixels}")
                print(f"Removed pixels: {removed_pixels}")
                print(f"Reduction: {reduction_percent:.2f}%")

                # A completely empty cleaned cube cannot be analyzed
                if retained_pixels == 0:

                    raise ValueError(
                        "No valid pixels found in cleaned cube."
                    )

                # --------------------------------------
                # Compare spectra using the SAME mask
                # --------------------------------------

                # Both cubes are evaluated on the same leaf pixels, so any
                # difference between the two spectra comes from the cleaning
                # itself and not from a different pixel selection.
                accumulate_statistics(
                    original,
                    valid_mask,
                    sum_original,
                    squares_original,
                    counts_original
                )

                accumulate_statistics(
                    cleaned,
                    valid_mask,
                    sum_cleaned,
                    squares_cleaned,
                    counts_cleaned
                )

                # --------------------------------------
                # Accumulate pixel counts
                # --------------------------------------

                total_original_pixels += original_pixels

                total_retained_pixels += retained_pixels

                # --------------------------------------
                # Save image-level statistics
                # --------------------------------------

                # Row of the CSV report for this image
                summary_rows.append({

                    "Category": category,
                    "Image": base_name,
                    "Original_Pixels": original_pixels,
                    "Retained_Pixels": retained_pixels,
                    "Removed_Pixels": removed_pixels,
                    "Reduction_Percent": round(
                        reduction_percent, 3
                    )

                })

                category_processed += 1
                total_processed += 1

                print(f"SUCCESS: {base_name}")

            except Exception as e:

                # Log the failure and move on to the next image
                category_errors += 1
                total_errors += 1

                print(
                    f"ERROR processing {original_path.name}: {e}"
                )

        # ----------------------------------------------
        # Generate comparison figure for each category
        # ----------------------------------------------

        # No valid pair: nothing to plot for this category
        if category_processed == 0:

            print(f"No valid image pairs for {category}.")
            continue

        # Calculate mean and standard deviation
        # using all retained pixels across the category.
        # Bands without any valid value stay at NaN.

        mean_original = np.full(
            number_of_bands, np.nan
        )

        mean_cleaned = np.full(
            number_of_bands, np.nan
        )

        std_original = np.full(
            number_of_bands, np.nan
        )

        std_cleaned = np.full(
            number_of_bands, np.nan
        )

        for b in range(number_of_bands):

            if counts_original[b] > 0:

                # Mean = sum / count
                mean_original[b] = (
                    sum_original[b] / counts_original[b]
                )

                # Variance = E[x^2] - (E[x])^2, computed from the
                # running sums
                variance = (
                    squares_original[b] / counts_original[b]
                    - mean_original[b] ** 2
                )

                # max(..., 0) protects against tiny negative values
                # caused by floating-point rounding
                std_original[b] = np.sqrt(
                    max(variance, 0)
                )

            if counts_cleaned[b] > 0:

                mean_cleaned[b] = (
                    sum_cleaned[b] / counts_cleaned[b]
                )

                variance = (
                    squares_cleaned[b] / counts_cleaned[b]
                    - mean_cleaned[b] ** 2
                )

                std_cleaned[b] = np.sqrt(
                    max(variance, 0)
                )

        # Smooth mean curves for visualization only (light Gaussian filter
        # along the spectral axis); the statistics above are not modified
        smooth_original = gaussian_filter1d(
            mean_original, sigma=1
        )

        smooth_cleaned = gaussian_filter1d(
            mean_cleaned, sigma=1
        )

        # ----------------------------------------------
        # Pixel reduction calculation
        # ----------------------------------------------

        # Overall reduction of the category (all images together)
        removed_total = (
            total_original_pixels - total_retained_pixels
        )

        reduction_percent = (
            100 * removed_total / total_original_pixels
            if total_original_pixels > 0 else 0
        )

        # ----------------------------------------------
        # Create combined figure
        # ----------------------------------------------

        # Two panels: spectra (wide, left) and pixel counts (narrow, right)
        fig, (ax1, ax2) = plt.subplots(
            1,
            2,
            figsize=(15, 6),
            gridspec_kw={"width_ratios": [3, 1]}
        )

        fig.suptitle(
            f"{category.capitalize()} - Impact of Image Cleaning",
            fontsize=15,
            fontweight="bold"
        )

        # ----------------------------------------------
        # LEFT: Mean spectral reflectance
        # ----------------------------------------------

        # Original spectrum: solid line, shaded band = +/- 1 std
        ax1.plot(
            wavelengths,
            smooth_original,
            color="dodgerblue",
            linewidth=2,
            label="Original (retained leaf pixels)"
        )

        ax1.fill_between(
            wavelengths,
            smooth_original - std_original,
            smooth_original + std_original,
            color="dodgerblue",
            alpha=0.15
        )

        # Cleaned spectrum: dashed line, so it stays visible when it
        # overlaps the original one
        ax1.plot(
            wavelengths,
            smooth_cleaned,
            color="mediumvioletred",
            linewidth=2,
            linestyle="--",
            label="Cleaned (retained leaf pixels)"
        )

        ax1.fill_between(
            wavelengths,
            smooth_cleaned - std_cleaned,
            smooth_cleaned + std_cleaned,
            color="mediumvioletred",
            alpha=0.15
        )

        ax1.set_xlabel(xlabel, fontsize=11)

        ax1.set_ylabel(
            "Mean Reflectance",
            fontsize=11
        )

        ax1.set_title(
            "(a) Spectral Information Preservation"
        )

        ax1.legend(fontsize=9)

        ax1.grid(
            True,
            linestyle="--",
            alpha=0.4
        )

        # ----------------------------------------------
        # RIGHT: Pixel count comparison
        # ----------------------------------------------

        pixel_counts = [
            total_original_pixels,
            total_retained_pixels
        ]

        bars = ax2.bar(
            ["Original", "Cleaned"],
            pixel_counts,
            color=["dodgerblue", "mediumvioletred"],
            alpha=0.8
        )

        ax2.set_ylabel(
            "Number of Pixels",
            fontsize=11
        )

        ax2.set_title(
            "(b) Uninformative Pixel Reduction"
        )

        ax2.grid(
            axis="y",
            linestyle="--",
            alpha=0.4
        )

        # Write the exact pixel count above each bar
        for bar, value in zip(bars, pixel_counts):

            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:,}",
                ha="center",
                va="bottom",
                fontsize=9
            )

        # Overall reduction percentage, placed in axes coordinates
        # (0-1) so its position does not depend on the data values
        ax2.text(
            0.5,
            0.92,
            f"{reduction_percent:.2f}% reduction",
            transform=ax2.transAxes,
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold"
        )

        # Leave 18% of free space above the tallest bar for the labels
        ax2.set_ylim(
            0,
            max(pixel_counts) * 1.18
        )

        # ----------------------------------------------
        # Save figure
        # ----------------------------------------------

        plt.tight_layout()

        figure_path = (
            OUTPUT_ROOT / f"{category}_impact.png"
        )

        # 300 dpi: publication quality
        plt.savefig(
            figure_path,
            dpi=300,
            bbox_inches="tight"
        )

        print(f"\nFigure saved: {figure_path}")

        plt.show()

        # Release the figure to avoid memory build-up
        plt.close(fig)

        # ----------------------------------------------
        # Category report
        # ----------------------------------------------

        print("\n" + "-" * 60)
        print(f"{category.upper()} IMPACT SUMMARY")
        print("-" * 60)

        print(f"Images processed: {category_processed}")
        print(f"Original pixels: {total_original_pixels:,}")
        print(f"Retained pixels: {total_retained_pixels:,}")
        print(f"Removed pixels: {removed_total:,}")
        print(f"Overall reduction: {reduction_percent:.2f}%")
        print(f"Errors: {category_errors}")

    # --------------------------------------------------
    # Save detailed CSV report
    # --------------------------------------------------

    csv_path = OUTPUT_ROOT / "pixel_reduction_summary.csv"

    # Write the report only if at least one image was processed
    if summary_rows:

        # newline="" avoids blank lines between rows on Windows;
        # utf-8 keeps special characters intact
        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as csv_file:

            # Column names are taken from the keys of the first row
            writer = csv.DictWriter(
                csv_file,
                fieldnames=summary_rows[0].keys()
            )

            writer.writeheader()
            writer.writerows(summary_rows)

        print(f"\nDetailed CSV report saved: {csv_path}")

    # --------------------------------------------------
    # Final report
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 6 COMPLETED")
    print("=" * 70)

    print(f"Total images processed: {total_processed}")
    print(f"Missing cleaned images: {total_missing}")
    print(f"Total errors: {total_errors}")

    print(f"\nOutput directory: {OUTPUT_ROOT}")

    print("=" * 70)

    # Success only if something was processed and nothing went wrong
    return total_processed > 0 and total_errors == 0 and total_missing == 0


if __name__ == "__main__":
    run()