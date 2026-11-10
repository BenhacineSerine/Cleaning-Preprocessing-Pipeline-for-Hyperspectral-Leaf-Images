# RGB-Guided Hyperspectral Preprocessing

A Python pipeline for preprocessing hyperspectral images (HSI) of
grapevine leaves using simulated RGB images as spatial guides.

The workflow removes non-informative background pixels and white
illumination artifacts, then transfers the resulting masks to the
original hyperspectral cubes. **All original spectral bands and the
spectral values of retained leaf pixels are preserved.** The method
performs spatial data reduction, not spectral band reduction.

## Workflow

The pipeline consists of six steps:

1.  **Generate simulated RGB images** from three selected HSI bands:
    -   Red: 660.84 nm (band index 90)
    -   Green: 548.55 nm (band index 52)
    -   Blue: 449.35 nm (band index 18)
2.  **Remove the background** using the U2-Net model through `rembg`.
3.  **Detect and suppress white illumination artifacts** using OpenCV
    template matching.
4.  **Transfer the RGB-derived masks** to the original HSI cubes and set
    pixels outside the leaf region to zero.
5.  **Reconstruct RGB images** from the cleaned HSI cubes for visual
    inspection.
6.  **Evaluate preprocessing impact**, including spatial pixel reduction
    and spectral-value preservation.

The six steps are orchestrated by `main.py`.

## Repository structure

``` text
RGB-Guided-Hyperspectral-Preprocessing/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── main.py
├── step1_generate_rgb_from_hsi.py
├── step2_background_removal.py
├── step3_remove_white_illumination.py
├── step4_apply_masks_to_hsi.py
├── step5_final_simulated_rgb.py
└── step6_reduction_impact_analysis.py
```

## Dataset

The pipeline was developed using the publicly available grapevine-leaf
hyperspectral dataset:

-   **Original dataset:** [Hyperspectral images of grape leaves
    including healthy leaves and leaves with biotic and abiotic
    symptoms](https://doi.org/10.57745/WW7TY7)
-   **Associated article:** [Ryckewaert et al. (2023), Scientific
    Data](https://doi.org/10.1038/s41597-023-02642-w)
-   **Cleaned dataset:** [Cleaned Dataset after
    processing](https://doi.org/10.5281/zenodo.24118345)

The demonstration configuration processes 10 samples (five healthy and
five unhealthy). All unhealthy samples are grouped into one class;
individual diseases and stress conditions are not distinguished.

The cleaned dataset is derived from the original dataset. Please consult
the original and cleaned dataset records for licence terms and
attribution requirements. When using the cleaned data, cite both the
original source and the cleaned dataset.

## Requirements

Install the required Python packages with:

``` bash
pip install -r requirements.txt
```

Main dependencies include `numpy`, `spectral`, `Pillow`,
`opencv-python-headless`, `rembg`, `matplotlib`, and `scipy`.

## Input data organization

Place the original HSI data and illumination templates in the following
structure:

``` text
/data/
├── Original_Dataset/
│   ├── healthy/
│   └── unhealthy/
└── Templates_Matching/
```

The input paths in the scripts should match the actual location of your
data.

## Run the pipeline

Run all six preprocessing steps in sequence:

``` bash
python main.py
```

The pipeline reads input data from the configured input directories and
writes generated files to `/results` (for the Code Ocean configuration).
Outputs include simulated RGB images, background-removed images,
illumination-corrected images, cleaned HSI cubes, reconstructed RGB
images, and pixel-reduction reports.

## Citation

If you use this pipeline or the cleaned dataset in your research, please
cite the original dataset and the cleaned dataset:

**Original dataset**

Ryckewaert, M. (2023). *Hyperspectral images of grape leaves including
healthy leaves and leaves with biotic and abiotic symptoms* \[Data
set\]. Recherche Data Gouv. https://doi.org/10.57745/WW7TY7

**Cleaned dataset**

Please use the citation exported by the [Zenodo
record](https://doi.org/10.5281/zenodo.24118345).

## Contact

**Serine Amina Benhacine**\
serine.benhacine@univ-constantine2.dz
