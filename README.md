<<<<<<< HEAD
# Tamil-Letter-Clustering
clustering modern tamil letters
=======
# Unsupervised Clustering of Handwritten Tamil Images

This project groups handwritten Tamil character images by visual similarity. It uses a **CRNN autoencoder (CNN + bidirectional GRU) with IDEC** (Improved Deep Embedded Clustering) to learn image embeddings and assign them to clusters.

> **This is an image-clustering project, not an OCR system.** It does not transcribe an image or return a Tamil character label.

## What makes this project distinctive

- **Unsupervised representation learning for Tamil handwriting:** the model learns 32-dimensional image embeddings from reconstruction rather than relying on character labels during training.
- **Sequence-aware image encoder:** convolutional layers extract visual features, then a bidirectional GRU models those features across the image width. Mean and max pooling combine the sequence features before producing the embedding.
- **Joint clustering and representation refinement:** IDEC refines the learned embeddings and cluster assignments together, initialized with 31 clusters.
- **Robust, consistent image preprocessing:** input images are converted to grayscale, median-blurred, binarized with Otsu thresholding, cropped to the ink, resized proportionally to a 32 × 128 canvas, and normalized.
- **Useful ways to inspect a prediction:** the app shows the preprocessed input and its reconstruction, soft cluster scores, a PCA position against the dataset, and the eight nearest dataset examples.
- **Interactive exploration:** browse sample images from any of the 31 clusters directly in the app.

## Dataset

The included `Vowels&Consonants/` dataset contains **14,260 handwritten images** in **31 folders**. Folder names are not used to train the clustering model; they are used only for external evaluation metrics such as ARI and NMI.

Images are standardized to 32 × 128 pixels for the model. The trained model and inference data used by the app are stored in `artifacts/`.

## Results

The reported comparison on the project dataset shows the CRNN autoencoder + IDEC results alongside two K-Means baselines:

| Method | Silhouette ↑ | Davies–Bouldin ↓ | Calinski–Harabasz ↑ | ARI | NMI |
|---|---:|---:|---:|---:|---:|
| **CRNN AE + IDEC (proposed)** | **0.8097** | **0.2735** | **27809.3890** | **0.8601** | **0.9162** |
| CRNN AE + K-Means | 0.1551 | 2.0275 | 441.5318 | 0.7164 | 0.8102 |
| CNN-only AE + K-Means | 0.1521 | 2.0939 | 464.1242 | 0.6376 | 0.7774 |

These are results from this project's evaluation run, not a guarantee for other datasets or handwriting samples. The complete table is in [`clustering_results.csv`](./clustering_results.csv).

## Visual results

Example images from the learned clusters:

![Samples from handwritten Tamil image clusters](./cluster_samples.png)

PCA and UMAP views of the IDEC embeddings:

![PCA and UMAP visualizations of IDEC clusters](./pca_umap_idec.png)

## Run the demo

Run the following commands from the project root with Python installed.

```bash
python -m venv .venv
```

Activate the environment, then install the dependencies and launch the app:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux, activate it with:

```bash
source .venv/bin/activate
```

Then install dependencies and launch the app:

```bash
python -m pip install -r requirements.txt
python app.py
```

Open the local URL printed by Gradio in the terminal (typically `http://127.0.0.1:7860`).

The app has two tabs:

1. **Assign an image to a cluster** — upload an image to see its predicted cluster, soft assignments, reconstruction, nearest examples, and PCA position.
2. **Explore clusters** — select a cluster to view sample images assigned to it.

## Project files

| Path | Purpose |
|---|---|
| `app.py` | Gradio interface, preprocessing, inference, nearest-neighbor search, and cluster exploration |
| `artifacts/crnn_ae.pth` | Trained CRNN autoencoder weights |
| `artifacts/centers.npy` | IDEC cluster centers |
| `artifacts/embeddings.npy` | Dataset embeddings used for nearest-neighbor search and PCA |
| `artifacts/labels.npy` | Saved cluster assignments |
| `artifacts/images.npy` | Preprocessed dataset images shown in the app |
| `artifacts/config.json` | Image dimensions, latent size, and cluster count |
| `clustering_results.csv` | Evaluation metrics and baseline comparison |
| `cluster_samples.png`, `pca_umap_idec.png` | Saved visualizations of the clustering results |

The app loads the trained model and saved artifacts at startup; it does not train the model when launched.
>>>>>>> origin/master
