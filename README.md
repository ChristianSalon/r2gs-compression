# r2gs-compression

## 3DGS Compression Techniques

### Vector Quantization

Partition gaussians into clusters. Then for each cluster pick one representative gaussian. This forms a codebook (a set of selected gaussians). There can be one codebook for the entire multi-dimensional gaussian vector (position, rotation, scale, color, opacity, SH) or a codebook for the individual dimensions of the gaussian vector.

### Structuring and Dimension Reduction

Organizes gaussians hierarchically or around representative points, enabling efficient reuse of similar properties across regions.

### Octree

Spatial partitioning of gaussians.

### Anchor Based Representations

Anchors are initialized by voxelizing the 3D scene and are assigned position, context features, scaling, and learnable offsets to other gaussians. Similar to vector quantization, but here anchor points are stored.

### Multi-Resolution Hash Grids

Hierarchical set of hash maps starting from a coarse resolution to finer resolutions. The key to the hash map is a 3D voxel position, The values in the hash map are feature vectors. Each hash map gives a different value based on key. This ensures that a collision on one resolution does not produce an error since the chance that two different positions in all resolutions produce collisions is practically zero. A MLP is used to decode the final gaussian attributes from the concatenated feature vectors from all resolutions.

### Z-order (Morton) Curve

Creates a one dimensional index mapping multi-dimensional data into a linear sequence while preserving spatial locality.

### Tri-planes and K-planes

Projects 3D positions onto three orthogonal 2D planes. This creates a natural 2D organization, where gaussians that are spatially close in 3D tend to occupy proximate coordinates on each plane, allowing 2D feature maps to be saved or edited like images while preserving some local neighborhood relationships. For 3DGS compression, the query can be the position of the primitive. The planes can store features, which are decoded into the other attributes (e.g. color, opacity) with a small MLP.

### Self-Organizing Gaussians

Projects high-dimensional data onto a lower-dimensional grid, preserving the topological relationships between data points through competitive learning. Each scene is optimized separately. Enables the use of standard image compression algorithms.

### Region Adaptive Hierarchical Transform

RAHT is a bottom-up transform that reorganizes color attributes along an octree.

## 3DGS Attribute Pruning

Selective reduction of attributes associated with each gaussian, such as spherical harmonics, to optimize memory usage and computational efficiency.

## 3DGS Compaction Techniques

The key tool for achieving compaction in 3DGS is Adaptive Density Control (ADC). ADC adds or removes gaussians depending on their contribution to the scene, ensuring that only the most essential elements are retained. This ensures that additional gaussians are allocated where they are most needed, such as in high-frequency regions, while redundant or less impactful gaussians are pruned. ADC can be broadly categorized into two key approaches: 

- **Densification**
- **Pruning**

### Pruning

Many methods compute a sensitivity score for each gaussian, which expresses how much will the quality of the rendered scene drop, when the given gaussian is pruned. Based on this score, algorithms remove gaussians with the lowest sensitivity in iterations. 

## R2 Gaussian Compression

R2 gaussians consists of mean (position), covariance (scale and rotation) and density (attenuation coefficient). Compared to original 3DGS, R2 gaussians exclude color and spherical harmonics and instead of opacity, density is present. Because of this, compression techniques based on those attributes can not be simply used for compression and compaction of R2 gaussians. 

Opacity of 3DGS can be viewed as density of R2 gaussians. In 3DGS, gaussians are alpha blended using their color and opacity. In R2 gaussians, density is integrated along the ray path. Because of this, I think density can be also viewed as color from 3DGS. So possibly with some modifications, algorithms that depend on color can be useful for R2 gaussians.

## Dataset

Most available datasets can be categorized as:

- Real: X-ray projections given, sometimes camera positions are also given, need to compute GT volume (with FBP, FDK, ...).
    - [42 Walnuts](https://www.nature.com/articles/s41597-019-0235-y)
        - **Scanned Object:** 42 real biological walnuts (selected for natural structural variability and irregular boundaries).
        - **CT Type:** Laboratory X-ray cone-beam (CB) CT setup.
        - **Projections & Scans:** 42 distinct physical scans. Cone-beam projections were acquired across **three different source orbits** per walnut to provide multi-angle data and varying cone angles.
        - **File Size:** Approximately **6 GB per walnut** (ZIP archive), totaling around **254 GB** for the complete collection.
- Synthetic: GT volume given, need to compute synthetic X-ray projections and camera positions.
    - [CT-ORG](https://www.cancerimagingarchive.net/collection/ct-org/)
        - **Scanned Object:** Human anatomical structures (focusing on torso, lungs, liver, and kidneys).
        - **CT Type:** Clinical multi-detector CT (forward projections simulated via numerical ray-tracing).
        - **Projections & Scans:** Contains 140 multi-organ clinical CT volumes converted into synthetic projection datasets.
        - **File Size:** Varies by subset, typically ranging in the **tens of GBs** for full volumetric archives.
    - [MORE](https://huggingface.co/datasets/WSKINGDOM/MORE)
        - **Scanned Object:** Comprehensive multi-organ human bodies (chest, abdomen, pelvis).
        - **CT Type:** Standard clinical spiral/helical CT scans used to generate synthetic limited-angle or sparse-view projections.
        - **Projections & Scans:** Large-scale multi-center collection providing numerous 3D high-resolution reference volumes.
        - **File Size:** Large-scale repository spanning **several hundred GBs** across compressed data splits.
    - [Medical Segmentation Decathlon](https://huggingface.co/datasets/Novel-BioMedAI/Medical_Segmentation_Decathlon)
        - **Scanned Object:** Diverse human anatomical targets (liver, brain, heart, pancreas, colon, lung, hepatic vessels, etc.).
        - **CT & MRI Type:** Mixed clinical imaging modalities (subsets contain high-resolution clinical CT volumes used for simulation frameworks).
        - **Projections & Scans:** Tens to hundreds of 3D multi-class segmented volumes per specialized organ task.
        - **File Size:** Ranging from **10 GB to 50 GB** depending on the specific organ task subset.
    - [ToothFairy](https://ditto.ing.unimore.it/toothfairy/)
        - **Scanned Object:** Mandibular bone structures, teeth, and dental canals.
        - **CT Type:** Cone-beam dental computed tomography (CBCT).
        - **Projections & Scans:** Dozens to hundreds of high-resolution dental volumetric scans with expert annotations, optimized for bone-structure simulation.
        - **File Size:** Around **15 GB to 40 GB**.
    - [CT COLONOGRAPHY](https://www.cancerimagingarchive.net/collection/ct-colonography/)
        - **Scanned Object:** Human abdomen and pelvis, specifically targeting the colon for polyp detection and virtual colonoscopy.
        - **CT Type:** Clinical multi-detector row CT scanners.
        - **Projections & Scans:** Over 800 patient CT examinations consisting of thousands of high-resolution 2D axial slices.
        - **File Size:** Massive archive totaling **over 300 GB** across all patient cases.
- Full: X-ray projections, camera positions and GT volume given
    - [LDCT-and-Projection-data](https://www.cancerimagingarchive.net/collection/ldct-and-projection-data/)
        - **Scanned Object:** Human patients (abdomen, chest, and head scanning protocols).
        - **CT Type:** Clinical Siemens medical CT scanners (capturing both routine full-dose and simulated low-dose protocols).
        - **Projections & Scans:** Data from **30 patient cases**, including raw projection sinograms, routine full-dose CT volumes, and corresponding low-dose CT volumes.
        - **File Size:** Extremely large due to raw sinogram files, spanning **1.3 TB** in total.
    - [Cobra2026](https://zenodo.org/records/21322350)
        - **Scanned Object:** Human pelvic region (pelvic malignancies undergoing radiation therapy).
        - **CT Type:** Multi-center clinical Cone-Beam CT (CBCT) acquired via Elekta and Varian linear accelerator systems, paired with planning CTs.
        - **Projections & Scans:** **867 clinical cases** sourced from 6 international hospital centers. Includes raw measured CBCT projections, acquisition geometries, calibration data, clinical CBCT volumes, and synthetic projections derived from planning CTs.
        - **File Size:** Heavy multi-center dataset spanning **hundreds of GBs to 1 TB** across archived project packages.

## Metrics

**Quality:**

- PSNR (Peak Signal-to-Noise Ratio): X-ray projections
- SSIM (Structural Similarity Index Measure): X-ray projections
- RMSE/MSE (Root Mean Square Error): 3D volume

**Compression:**

- Number of gaussians
- Model file size

**Rendering (Novel View Synthesis):**

- FPS

## Experiments

TODO
