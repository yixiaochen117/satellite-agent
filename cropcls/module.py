"""LightningModule: freeze-then-finetune classifier with GPU-side preprocessing."""

from __future__ import annotations

import math
from collections.abc import Sequence

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
)

from .bands import S2_ALL_BANDS
from .model import backbone_parameters, build_model, head_parameters


def random_dihedral(x: torch.Tensor) -> torch.Tensor:
    """Per-sample random flips / 90 degree rotations (all 8 dihedral transforms).

    Satellite patches have no canonical "up", so this is a free, label-safe augment.
    """
    b = x.shape[0]
    dev = x.device
    m = torch.rand(b, device=dev) < 0.5
    x = torch.where(m[:, None, None, None], x.flip(-1), x)
    m = torch.rand(b, device=dev) < 0.5
    x = torch.where(m[:, None, None, None], x.flip(-2), x)
    if x.shape[-1] == x.shape[-2]:
        m = torch.rand(b, device=dev) < 0.5
        x = torch.where(m[:, None, None, None], x.transpose(-1, -2), x)
    return x


class CropClassifier(L.LightningModule):
    def __init__(
        self,
        num_classes: int,
        class_names: Sequence[str],
        bands: Sequence[str] = S2_ALL_BANDS,
        arch: str = "resnet18",
        weights: str | None = "SENTINEL2_ALL_MOCO",
        img_size: int = 224,
        dn_scale: float = 10000.0,
        lr: float = 1e-3,
        backbone_lr_mult: float = 0.1,
        weight_decay: float = 1e-4,
        freeze_epochs: int = 5,
        warmup_epochs: float = 1.0,
        label_smoothing: float = 0.1,
        drop_rate: float = 0.2,
        class_weights: Sequence[float] | None = None,
        timesteps: int = 1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = build_model(num_classes, bands, arch, weights, drop_rate, timesteps)
        self.model = self.model.to(memory_format=torch.channels_last)
        cw = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.register_buffer("class_weights", cw, persistent=False)

        def metrics(prefix: str) -> MetricCollection:
            return MetricCollection({
                "acc": MulticlassAccuracy(num_classes, average="micro"),
                "bal_acc": MulticlassAccuracy(num_classes, average="macro"),
                "f1": MulticlassF1Score(num_classes, average="macro"),
            }, prefix=prefix)

        self.train_metrics = metrics("train/")
        self.val_metrics = metrics("val/")
        self.test_metrics = metrics("test/")
        self.test_per_class = MulticlassAccuracy(num_classes, average="none")
        self.test_confmat = MulticlassConfusionMatrix(num_classes)
        self._frozen: bool | None = None

    # ---------------------------------------------------------------- preprocessing
    def preprocess(self, x: torch.Tensor, train: bool) -> torch.Tensor:
        x = x.float() / self.hparams.dn_scale
        if train:
            x = random_dihedral(x)
        s = self.hparams.img_size
        if x.shape[-2:] != (s, s):
            x = F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False)
        if self.hparams.timesteps > 1:
            return x  # TemporalResNet reshapes per date and sets the memory format
        return x.contiguous(memory_format=torch.channels_last)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raw DN input -> logits (preprocessing included, no augmentation)."""
        return self.model(self.preprocess(x, train=False))

    # ------------------------------------------------------------- freeze schedule
    def _set_frozen(self, frozen: bool) -> None:
        if frozen == self._frozen:
            return
        for p in backbone_parameters(self.model):
            p.requires_grad_(not frozen)
        self._frozen = frozen
        self.print(f"[epoch {self.current_epoch}] backbone "
                   f"{'frozen (training head only)' if frozen else 'unfrozen (full finetune)'}")

    def on_train_epoch_start(self) -> None:
        self._set_frozen(self.current_epoch < self.hparams.freeze_epochs)

    def on_train_batch_start(self, batch, batch_idx: int) -> None:
        # While frozen keep BatchNorm running stats fixed at their pretrained values.
        if self._frozen:
            head = set(self.model.get_classifier().modules())
            for m in self.model.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm) and m not in head:
                    m.eval()

    # ------------------------------------------------------------------ steps
    def _loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits.float(), y, weight=self.class_weights,
                               label_smoothing=self.hparams.label_smoothing)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self.model(self.preprocess(x, train=True))
        loss = self._loss(logits, y)
        self.train_metrics.update(logits, y)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def on_train_epoch_end(self) -> None:
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        self.log("val/loss", self._loss(logits, y), prog_bar=True)
        self.val_metrics.update(logits, y)

    def on_validation_epoch_end(self) -> None:
        m = self.val_metrics.compute()
        self.log_dict(m, prog_bar=True)
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        self.log("test/loss", self._loss(logits, y))
        self.test_metrics.update(logits, y)
        self.test_per_class.update(logits, y)
        self.test_confmat.update(logits, y)

    def on_test_epoch_end(self) -> None:
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()
        # kept for evaluate.py / train.py to write reports
        self.last_test_per_class = self.test_per_class.compute().cpu()
        self.last_test_confmat = self.test_confmat.compute().cpu()
        self.test_per_class.reset()
        self.test_confmat.reset()

    def predict_step(self, batch, batch_idx: int):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        return self(x).softmax(-1)

    # --------------------------------------------------------------- optimiser
    def configure_optimizers(self):
        hp = self.hparams
        head = head_parameters(self.model)
        backbone = backbone_parameters(self.model)
        opt = torch.optim.AdamW([
            {"params": head, "lr": hp.lr},
            {"params": backbone, "lr": hp.lr * hp.backbone_lr_mult},
        ], weight_decay=hp.weight_decay)

        total = max(int(self.trainer.estimated_stepping_batches), 1)
        max_epochs = max(self.trainer.max_epochs or 1, 1)
        per_epoch = max(total // max_epochs, 1)
        warm = max(int(hp.warmup_epochs * per_epoch), 1)
        unfreeze_at = hp.freeze_epochs * per_epoch

        def cosine(step: int, start: int) -> float:
            span = max(total - start, 1)
            prog = min(max(step - start, 0) / span, 1.0)
            return 0.5 * (1 + math.cos(math.pi * prog))

        def head_lambda(step: int) -> float:
            if step < warm:
                return (step + 1) / warm
            return cosine(step, warm)

        def backbone_lambda(step: int) -> float:
            # zero while frozen, then its own warmup + cosine to the end
            if step < unfreeze_at:
                return 0.0
            s = step - unfreeze_at
            if s < warm:
                return (s + 1) / warm
            return cosine(step, unfreeze_at + warm)

        sched = torch.optim.lr_scheduler.LambdaLR(opt, [head_lambda, backbone_lambda])
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
