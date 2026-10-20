import json
import numpy as np
import cv2
import torch
import torch.nn as nn
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

ART = "artifacts"
cfg = json.load(open(f"{ART}/config.json"))
H, W, LATENT, K = cfg["H"], cfg["W"], cfg["LATENT"], cfg["N_CLUSTERS"]


# ---------- Model (same architecture as training) ----------
class CRNNAE(nn.Module):
    def __init__(self, latent=LATENT):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                                 nn.ReLU(), nn.MaxPool2d(2))
        self.cnn = nn.Sequential(blk(1, 32), blk(32, 64), blk(64, 128))
        self.gru = nn.GRU(128 * 4, 128, batch_first=True, bidirectional=True)
        self.to_z = nn.Linear(256 * 2, latent)
        self.from_z = nn.Linear(latent, 128 * 4 * 16)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid())

    def encode(self, x):
        f = self.cnn(x)
        B, C, h, w = f.shape
        seq = f.permute(0, 3, 1, 2).reshape(B, w, C * h)
        out, _ = self.gru(seq)
        feat = torch.cat([out.mean(1), out.max(1)[0]], 1)
        return self.to_z(feat)

    def forward(self, x):
        z = self.encode(x)
        xr = self.dec(self.from_z(z).view(-1, 128, 4, 16))
        return xr, z


# ---------- Load artifacts once ----------
model = CRNNAE()
model.load_state_dict(torch.load(f"{ART}/crnn_ae.pth", map_location="cpu"))
model.eval()
centers = np.load(f"{ART}/centers.npy")
Z = np.load(f"{ART}/embeddings.npy")
labels = np.load(f"{ART}/labels.npy")
images = np.load(f"{ART}/images.npy")
pca = PCA(2, random_state=42).fit(Z)
P_all = pca.transform(Z)
sizes = np.bincount(labels, minlength=K)


# ---------- Preprocessing (same as training) ----------
def to_gray(arr):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr.astype(np.uint8)
    if arr.shape[2] == 4:  # RGBA: composite on white background
        a = arr[:, :, 3:4] / 255.0
        arr = (arr[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
    return cv2.cvtColor(arr[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)


def preprocess(arr):
    img = to_gray(arr)
    img = cv2.medianBlur(img, 3)
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if bw.mean() > 127:
        bw = 255 - bw
    ys, xs = np.where(bw > 0)
    if len(ys) > 0:
        bw = bw[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = bw.shape
    new_w = max(1, min(W, int(w * H / h)))
    bw = cv2.resize(bw, (new_w, H), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((H, W), np.uint8)
    off = (W - new_w) // 2
    canvas[:, off:off + new_w] = bw
    return canvas.astype(np.float32) / 255.0


def upscale(img_u8, f=4):
    return cv2.resize(img_u8, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)


# ---------- Prediction ----------
def predict(arr):
    if arr is None:
        raise gr.Error("Please upload an image first.")
    x = preprocess(arr)
    with torch.no_grad():
        xr, z = model(torch.tensor(x)[None, None])
    z = z[0].numpy()
    xr = xr[0, 0].numpy()

    d = ((z[None] - centers) ** 2).sum(1)
    q = 1.0 / (1.0 + d)              # Student-t soft assignment (alpha = 1)
    q = q / q.sum()
    cluster = int(q.argmax())

    # nearest neighbours in the dataset
    nn_idx = np.argsort(np.linalg.norm(Z - z, axis=1))[:8]
    gallery = [(upscale(images[i], 3), f"Cluster {labels[i]}") for i in nn_idx]

    # PCA plot
    p = pca.transform(z[None])[0]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(P_all[:, 0], P_all[:, 1], c=labels, cmap="tab20", s=5, alpha=0.6)
    ax.scatter(p[0], p[1], c="black", marker="*", s=300, label="Your image")
    ax.set_title("Position in embedding space (PCA)")
    ax.legend()
    plt.tight_layout()

    probs = {f"Cluster {i}": float(q[i]) for i in range(K)}
    summary = f"**Predicted cluster: {cluster}**  (confidence {q[cluster] * 100:.1f}%)"
    return (upscale((x * 255).astype(np.uint8)),
            upscale((xr * 255).astype(np.uint8)),
            summary, probs, gallery, fig)


def show_cluster(c):
    idx = np.where(labels == int(c))[0]
    if len(idx) == 0:
        return []
    sel = np.random.default_rng(0).choice(idx, min(30, len(idx)), replace=False)
    return [upscale(images[i], 3) for i in sel]


# ---------- UI ----------
with gr.Blocks(title="Tamil Word Image Clustering") as demo:
    gr.Markdown(
        "# Unsupervised Clustering of Handwritten Tamil Word Images\n"
        "CRNN Autoencoder (CNN + BiGRU) + IDEC. The app groups visually similar images; "
        "it does not recognize Tamil text.")

    with gr.Tab("Assign an image to a cluster"):
        with gr.Row():
            inp = gr.Image(label="Upload a handwritten Tamil image", type="numpy",
                           image_mode=None, height=250)
            with gr.Column():
                pre = gr.Image(label="Preprocessed", height=130, interactive=False)
                rec = gr.Image(label="Reconstruction", height=130, interactive=False)
        btn = gr.Button("Cluster this image", variant="primary")
        summary = gr.Markdown()
        with gr.Row():
            lab = gr.Label(label="Soft assignment (top 5)", num_top_classes=5)
            plot = gr.Plot(label="PCA position")
        gal = gr.Gallery(label="Most similar images in the dataset", columns=8, height=200)

        btn.click(predict, inp, [pre, rec, summary, lab, gal, plot])

    with gr.Tab("Explore clusters"):
        dd = gr.Dropdown(choices=[(f"Cluster {i} ({sizes[i]} images)", i) for i in range(K)],
                         value=0, label="Choose a cluster")
        explore = gr.Gallery(label="Sample images", columns=10, height=400)
        dd.change(show_cluster, dd, explore)
        demo.load(show_cluster, dd, explore)

if __name__ == "__main__":
    demo.launch()