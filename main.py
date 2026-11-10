
"""
Main script of the hyperspectral image preprocessing pipeline.

This script orchestrates the six processing stages in their
predefined execution order. Each stage uses the outputs generated
by the preceding stage.

The pipeline stops immediately if a stage raises an exception
or explicitly returns False.
"""

# ------------------------------------------------------------
# Import processing modules
# ------------------------------------------------------------

from step1_generate_rgb_from_hsi import run as step1
from step2_background_removal import run as step2
from step3_remove_white_illumination import run as step3
from step4_apply_masks_to_hsi import run as step4
from step5_final_simulated_rgb import run as step5
from step6_reduction_impact_analysis import run as step6


def main():
    """
    Execute all six processing stages sequentially.

    A stage is considered unsuccessful if it raises an exception
    or explicitly returns False. Functions returning None are
    considered successful.
    """

    print("\n" + "=" * 70)
    print("       HYPERSPECTRAL IMAGE PREPROCESSING PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # Define the processing workflow
    # --------------------------------------------------------

    steps = [
        ("STEP 1: Simulated RGB Image Generation from HSI Data", step1),

        ("STEP 2: Background Removal", step2),

        ("STEP 3: Illumination Artifact Detection and Suppression", step3),

        ("STEP 4: Spatial Mask Transfer to Hyperspectral Data", step4),

        ("STEP 5: RGB Reconstruction from Cleaned HSI Cubes", step5),

        ("STEP 6: Quantitative Evaluation of Preprocessing Impact", step6)
    ]

    # --------------------------------------------------------
    # Execute processing stages sequentially
    # --------------------------------------------------------

    for step_name, step_function in steps:

        print("\n\n" + "#" * 70)
        print(step_name)
        print("#" * 70 + "\n")

        try:

            result = step_function()

            if result is False:
                raise RuntimeError(
                    f"{step_name} reported failure."
                )

            print("\n" + "-" * 70)
            print(f"{step_name} COMPLETED SUCCESSFULLY")
            print("-" * 70)

        except Exception as e:

            print("\n" + "!" * 70)
            print(f"PIPELINE TERMINATED AT: {step_name}")
            print(f"ERROR DETAILS: {e}")
            print("!" * 70)

            raise

    # --------------------------------------------------------
    # Final execution report
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ALL SIX PROCESSING STAGES COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
