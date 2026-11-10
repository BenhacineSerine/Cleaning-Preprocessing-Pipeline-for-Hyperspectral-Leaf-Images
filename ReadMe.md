# Automated Preprocessing Pipeline for Hyperspectral Leaf Images: Background and Illumination Removal Using Simulated RGB Masks

## 1. Overview

This capsule provides a reproducible, RGB-guided preprocessing pipeline for hyperspectral images (HSI) of grapevine leaves. It is designed to reduce non-informative spatial regions, isolate leaf regions of interest (ROIs), and retain the original spectral values of the pixels selected by the spatial masks.

The key idea is to perform segmentation and illumination-artifact detection in the simulated RGB domain, where standard 2D image-processing methods can be applied, and then transfer the resulting spatial masks to the corresponding original hyperspectral cubes.

The complete workflow consists of six sequential stages and is orchestrated by `main.py`.

## 2. Dataset source and subset used in this capsule

### 2.1 Original dataset

The original data are derived from the publicly available grapevine-leaf hyperspectral dataset published by Ryckewaert et al. The source repository contains 204 hyperspectral image acquisitions. Each acquisition is provided in ENVI format as a paired header file (`.hdr`) and binary data file (`.dat`). The dataset covers healthy leaves and leaves exhibiting foliar symptoms associated with biotic or abiotic stress.

- **Original dataset:** [Hyperspectral images of grape leaves including healthy leaves and leaves with biotic and abiotic symptoms](https://doi.org/10.57745/WW7TY7)
- **Associated scientific article:** [Ryckewaert et al. (2023), Scientific Data](https://doi.org/10.1038/s41597-023-02642-w)

For this capsule, the samples are organized into two classes: `healthy` and `unhealthy`. The `unhealthy` category groups the samples designated as symptomatic in the source data and in this project’s organization.

### 2.2 Demonstration subset used for the reproducible run

To limit storage requirements and execution time, the capsule is configured to process a subset of **10 samples: five healthy and five unhealthy samples**. This subset represents only a portion of the original dataset and is provided for demonstration and reproducibility purposes. It is intended to illustrate and reproduce the complete preprocessing workflow and does not include the full original dataset.

The complete original dataset remains available through its official repository (see the link above). In this preprocessed version, all unhealthy grapevine leaf samples, originally associated with different biotic and abiotic conditions, have been grouped into a single class labeled "Unhealthy", while healthy samples remain in a separate class labeled "Healthy". No additional samples have been introduced.

### 2.3 Complete cleaned dataset

The complete cleaned HSI dataset will be made available separately for users who want to inspect or reuse the full preprocessing results without running the entire pipeline.

- **Complete cleaned dataset:** [Insert the Zenodo record URL or DOI after publication]

The cleaned dataset is derived from the original source data. It is not an independently acquired dataset. Its reuse must follow the applicable source-data licence and attribution requirements. The original dataset record identifies the **Etalab Open License 2.0** as its data-use licence. The article’s separate CC BY 4.0 notice applies to the publication and to material covered by that article licence; for redistribution of the underlying hyperspectral data, follow the licence stated in the original dataset record. Users should also consult the licence and attribution information displayed in the cleaned dataset’s own Zenodo record.

## 3. Processing workflow

The pipeline is implemented as six independent Python modules. Each module exposes a `run()` function, and `main.py` executes the modules in the required order.

### Step 1 – Simulated RGB Image Generation from Hyperspectral Data

Three spectral bands are extracted from each original HSI cube and assigned to the red, green, and blue channels:

| RGB channel | Wavelength | Python band index (zero-based) |
|---|---:|---:|
| Red (R) | 660.84 nm | 90 |
| Green (G) | 548.55 nm | 52 |
| Blue (B) | 449.35 nm | 18 |

A common intensity range is used to normalize the three channels and generate simulated RGB images. The images are saved in PNG format.

### Step 2 – Background Removal

The U2-Net model, accessed through the `rembg` library, is applied to the simulated RGB images to segment the leaf from the surrounding background. The resulting images are saved as transparent PNG files.

### Step 3 – Illumination Artifact Detection and Suppression

White illumination reflections are detected in the background-removed RGB images using OpenCV template matching. Multiple reference templates represent different possible positions of the illumination artifact. For each image, the best match is retained and compared with the configured confidence threshold. A detected region is replaced with black pixels and the processed image is saved.

### Step 4 – Spatial Mask Transfer to Hyperspectral Data

The processed RGB images from Step 3 are converted into binary spatial masks. Each mask is matched to the corresponding original HSI cube using the image filename. After verifying that the spatial dimensions match, the mask is applied to every spectral band of the original cube.

Pixels outside the mask are set to zero across all spectral bands. The spectral values of retained pixels are copied from the original HSI cube without intentional spectral modification. The cleaned cubes are saved in:

- ENVI format: `.hdr` and `.dat`;
- NumPy format: `.npy`.

### Step 5 – RGB Reconstruction from Cleaned Hyperspectral Cubes

Simulated RGB images are reconstructed from the cleaned HSI cubes using the same three spectral bands as in Step 1. This stage provides a visual quality check of the spatially masked hyperspectral data. The reconstructed PNG images are saved for both categories.

### Step 6 – Quantitative Evaluation of Preprocessing Impact

The original and cleaned HSI cubes are compared for each class. A spatial mask is derived from the cleaned cube and applied to both the original and cleaned cubes when calculating the spectral statistics. This ensures that the mean spectra are compared over the same retained leaf pixels.

The analysis produces:

- mean spectral profiles and standard-deviation ranges before and after cleaning;
- original, retained, and removed pixel counts;
- the percentage of spatial pixels removed;
- a CSV report containing image-level pixel-reduction statistics.

The pixel-reduction percentage quantifies the proportion of spatial pixels excluded by the mask. It does **not** represent a reduction in the number of spectral bands or necessarily a reduction in the stored dimensions of the HSI cube.

## 4. Repository and capsule structure

The capsule code is organized as follows:

```text
/code/
├── main.py
├── run
├── step1_generate_rgb_from_hsi.py
├── step2_background_removal.py
├── step3_remove_white_illumination.py
├── step4_apply_masks_to_hsi.py
├── step5_final_simulated_rgb.py
├── step6_reduction_impact_analysis.py
└── README.md
```

The main input datasets and reference templates are attached under `/data`. The expected input organization is:

```text
/data/
├── Original_Dataset/
│   ├── healthy/
│   └── unhealthy/
└── Templates_Matching/
```

Additional copies of selected intermediate datasets may also be attached under `/data` to support independent testing of individual steps.

## 5. Running the complete pipeline

The recommended way to reproduce the complete workflow is to execute the capsule’s configured `run` entry point. The entry point calls `main.py`, which runs all six stages sequentially.

The pipeline can also be launched from a terminal with:

```bash
python -u /code/main.py
```

During a complete run, intermediate files are written to `/results` and consumed by the subsequent stages in the **same execution**. The pipeline therefore does not require intermediate results from a previous run.

**Important Code Ocean note:** `/results` is initialized for a new Reproducible Run. Do not expect files generated during an earlier run to remain there. For a complete execution through `main.py`, each stage must read the `/results/...` directory produced by the preceding stage, while original inputs and reference templates are read from `/data`.

For the complete-pipeline configuration, Step 6 must read the cleaned HSI cubes from:

```python
CLEANED_ROOT = Path("/results/Cleaned_HSI")
```

The original cubes remain under `/data/Original_Dataset`.

## 6. Testing an individual processing step

Each processing stage is implemented as an independent Python module and can be executed separately. This is useful for debugging or inspecting one operation without rerunning all preceding stages.

However, a new Code Ocean execution starts with an initialized `/results` directory. Therefore, when testing a step independently, its required intermediate input must be available as a persistent Data Asset under `/data`.

Copies of the intermediate results have been attached to this capsule for this purpose. When testing an individual step, change only the relevant input path so that it points to the corresponding copy under `/data`. The output of the individual test is still written to `/results`.

### 6.1 Input-path changes for individual testing

The following table shows the expected paths. The `/data` paths must match the names of the Data Assets actually attached to the capsule.

| Step tested | Input used in the complete pipeline | Input for an independent test |
|---|---|---|
| Step 1 – RGB generation | `/data/Original_Dataset` | `/data/Original_Dataset` (no change) |
| Step 2 – Background removal | `/results/RGB_Dataset` | `/data/RGB_Dataset` |
| Step 3 – Illumination suppression | `/results/RGB_BackGround_Removed` | `/data/RGB_BackGround_Removed` |
| Step 3 – Reference templates | `/data/Templates_Matching` | `/data/Templates_Matching` (no change) |
| Step 4 – Original HSI cubes | `/data/Original_Dataset` | `/data/Original_Dataset` (no change) |
| Step 4 – RGB masks | `/results/RGB_Without_White_Illumination` | `/data/RGB_Without_White_Illumination` |
| Step 5 – Cleaned HSI cubes | `/results/Cleaned_HSI` | `/data/Cleaned_HSI` |
| Step 6 – Original HSI cubes | `/data/Original_Dataset` | `/data/Original_Dataset` (no change) |
| Step 6 – Cleaned HSI cubes | `/results/Cleaned_HSI` | `/data/Cleaned_HSI` |

For example, to test Step 2 independently, change its input directory from:

```python
data_dir = Path("/results/RGB_Dataset")
```

to:

```python
data_dir = Path("/data/RGB_Dataset")
```

Use `/data/RGB_Dataset` with no space in the path.

### 6.2 Individual-step testing procedure

1. Attach the required saved intermediate dataset as a Data Asset under `/data`.
2. Change the relevant input path in the selected step to the corresponding `/data/...` directory, as shown in the table above.
3. Execute the selected Python module independently.
4. Inspect the execution logs and the generated files under `/results`.
5. Before launching the complete pipeline again, restore the input path to the `/results/...` location expected by the full workflow. For Step 1 and the original HSI inputs in Steps 4 and 6, the source remains under `/data`.

In summary, `/data` stores persistent input datasets and saved intermediate copies, whereas `/results` stores outputs generated during the current execution.

## 7. Expected output structure

A complete run is expected to generate the following principal output directories:

```text
/results/
├── RGB_Dataset/
│   ├── healthy/
│   └── unhealthy/
├── RGB_BackGround_Removed/
│   ├── healthy/
│   └── unhealthy/
├── RGB_Without_White_Illumination/
│   ├── healthy/
│   └── unhealthy/
├── Cleaned_HSI/
│   ├── healthy/
│   └── unhealthy/
├── Simulated_Cleaned_RGB/
│   ├── healthy/
│   └── unhealthy/
└── Reduction_Impact/
    ├── healthy_impact.png
    ├── unhealthy_impact.png
    └── pixel_reduction_summary.csv
```

The exact number of output files depends on the samples successfully processed during the run. The original input files under `/data` are not modified.

## 8. Software and dependencies

The pipeline is implemented in Python and uses the following main packages:

- `spectral` – reading and writing ENVI hyperspectral data;
- `numpy` – numerical operations and array manipulation;
- `Pillow` – RGB image loading and saving;
- `opencv-python-headless` – grayscale conversion and template matching;
- `rembg` – U2-Net-based background removal;
- `matplotlib` – image visualization and impact-analysis figures;
- `scipy` – Gaussian smoothing of spectral curves.

The required packages are listed in `requirements.txt` and installed through the capsule’s computational environment.

## 9. Data provenance, licensing, and citation

### 9.1 Original data source

The original HSI data were created and published by Ryckewaert and colleagues. Users of the cleaned dataset should acknowledge the original source and cite the original data record and associated article:

**Original dataset citation**

Ryckewaert, M. (2023). *Hyperspectral images of grape leaves including healthy leaves and leaves with biotic and abiotic symptoms* [Data set]. Recherche Data Gouv. https://doi.org/10.57745/WW7TY7

**Associated article**

Ryckewaert, M., et al. (2023). *Hyperspectral images of grapevine leaves including healthy leaves and leaves with biotic and abiotic symptoms*. Scientific Data, 10, 743. https://doi.org/10.1038/s41597-023-02642-w

The original data repository identifies the Etalab Open License 2.0 as the data-use licence. Please consult the original record for the applicable terms and attribution requirements.

### 9.2 Citation of the cleaned dataset

The cleaned HSI dataset is a derived resource produced through the preprocessing workflow documented in this capsule. **If you use, analyse, redistribute, or publish results based on the cleaned dataset, please cite both the original dataset and the cleaned dataset record.** This gives credit to the original data creator and to the work involved in producing and documenting the cleaned version.

Once the cleaned dataset has been published on Zenodo, use the following citation for the derived dataset (replace the placeholders with the final record information):

**Cleaned dataset record:** [Insert the Zenodo URL/DOI]

If additional contributors are formally listed as creators of the cleaned dataset in its Zenodo metadata, use the full creator list and the citation exported by Zenodo.


## 10. Contact and further information

For further details about the preprocessing methodology, input data, folder organization, dependencies, and execution procedure, please refer to this README and the individual documentation included in the six Python modules.
or contact the corresponding author at : serine.benhacine@univ-constantine2.dz
