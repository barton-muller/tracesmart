"""SAM models: SAM 2.1 (automatic grid masks and point prompts) and SAM 3 (text prompts)."""
import cv2
import numpy as np
import torch
from PIL import Image

SAM2 = "facebook/sam2.1-hiera-large"
SAM3 = "facebook/sam3"


def best_device() -> str:
    """The Apple GPU (MPS) or CUDA if available, otherwise the CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def auto_masks(image: Image.Image, model: str = SAM2, device: str | None = None, grid: int = 32,
               crops: bool = True) -> list[np.ndarray]:
    """SAM 2.1 automatic masks: a ``grid`` x ``grid`` point grid over the whole image, plus (``crops``)
    2x2 overlapping zoomed crops as test-time augmentation. Masks cut by an artificial crop edge are dropped."""
    from transformers import pipeline

    gen = pipeline("mask-generation", model=model, device=device or best_device())

    def run(img: Image.Image) -> list[np.ndarray]:
        out = gen(img, points_per_batch=32, points_per_crop=grid, pred_iou_thresh=0.8, stability_score_thresh=0.85)
        return [np.asarray(m, bool) for m in out["masks"]]

    w, h = image.size
    masks = run(image)
    if crops:
        cw, ch = int(w * 0.6), int(h * 0.6)
        for x0 in (0, w - cw):
            for y0 in (0, h - ch):
                tile = image.crop((x0, y0, x0 + cw, y0 + ch)).resize((1024, round(1024 * ch / cw)))
                for m in run(tile):
                    m = cv2.resize(m.astype(np.uint8), (cw, ch), interpolation=cv2.INTER_NEAREST).astype(bool)
                    edge = np.zeros_like(m)
                    for side, artificial in ((np.s_[0, :], y0 > 0), (np.s_[-1, :], y0 == 0),
                                             (np.s_[:, 0], x0 > 0), (np.s_[:, -1], x0 == 0)):
                        if artificial:
                            edge[side] = True
                    if (m & edge).any():
                        continue
                    full = np.zeros((h, w), bool)
                    full[y0:y0 + ch, x0:x0 + cw] = m
                    masks.append(full)
    return masks


class Prompter:
    """SAM 2.1 with point prompts. The image is encoded once; each batch of points is then cheap."""

    def __init__(self, image: Image.Image, model: str = SAM2, device: str | None = None):
        from transformers import Sam2Model, Sam2Processor

        self.image, self.device = image, device or best_device()
        self.processor = Sam2Processor.from_pretrained(model)
        self.model = Sam2Model.from_pretrained(model).to(self.device).eval()
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            self.embeddings = self.model.get_image_embeddings(inputs["pixel_values"])

    def masks(self, points: list[tuple[int, int]], batch: int = 16) -> list[np.ndarray]:
        """One mask per (x, y) point."""
        out = []
        for i in range(0, len(points), batch):
            chunk = points[i:i + batch]
            inputs = self.processor(images=self.image, input_points=[[[[x, y]] for x, y in chunk]],
                                    input_labels=[[[1] for _ in chunk]], return_tensors="pt").to(self.device)
            inputs.pop("pixel_values")
            inputs["image_embeddings"] = self.embeddings
            with torch.no_grad():
                res = self.model(**inputs, multimask_output=False)
            masks = self.processor.post_process_masks(res.pred_masks.cpu(), inputs["original_sizes"], binarize=True)[0]
            out += [m[0].numpy().astype(bool) for m in masks]
        return out


_SAM3: dict = {}


def sam3_text(image: Image.Image, prompt: str, model: str = SAM3, device: str | None = None,
              threshold: float = 0.5) -> list[np.ndarray]:
    """Every instance of a text concept ("window", "red shutters", "dog") found by SAM 3.

    The weights are gated on Hugging Face: accept the licence on the model page and run ``hf auth login``.
    """
    dev = device or best_device()
    if (model, dev) not in _SAM3:
        from transformers import Sam3Model, Sam3Processor

        _SAM3[(model, dev)] = (Sam3Processor.from_pretrained(model), Sam3Model.from_pretrained(model).to(dev).eval())
    processor, net = _SAM3[(model, dev)]
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(dev)
    with torch.no_grad():
        outputs = net(**inputs)
    res = processor.post_process_instance_segmentation(
        outputs, threshold=threshold, mask_threshold=0.5, target_sizes=inputs.get("original_sizes").tolist())[0]
    return [m.astype(bool) for m in res["masks"].cpu().numpy()]
