import numpy as np
import pytorch_lightning as pl
import torch.nn
import torch.optim
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

from src.dataset.preprocessing import denormalize_data_dict
from src.metrics.sevir_skill_score import SEVIRSkillScore
from src.models.unet.unet import UNet


class UNetPLModule(pl.LightningModule):
    def __init__(self, in_channels, out_channels, device, ndim=4, denormalization_method="sevir"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ndim = ndim
        self.denormalize = denormalization_method

        self.unet = UNet(in_channels, out_channels, bilinear=False)

        self.skill_metrics = ("csi", "pod", "sucr", "bias")
        self.thresholds = (16, 74, 133, 160, 181, 219)
        self.metrics = {
            "SSIM": SSIM(data_range=255).to(device),
            "MSE": MeanSquaredError().to(device),
            "MAE": MeanAbsoluteError().to(device),
            "SkillScore": SEVIRSkillScore(denormalization_method).to(device),
        }

        self.loss = torch.nn.MSELoss()

    def forward(self, x, y=None, compute_loss=False):
        unet_output = self.unet(x)

        if compute_loss:
            assert y is not None
            loss = self.loss(unet_output, y)
            return unet_output, loss

        return unet_output

    def training_step(self, batch, batch_idx):
        input, target = batch
        output, loss = self(input, target, compute_loss=True)

        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input, target = batch
        output, loss = self(input, target, compute_loss=True)
        self.log("validation_loss", loss, on_step=False, on_epoch=True, logger=True)

        output = denormalize_data_dict(data_dict={"vil": output.detach().float()}, method=self.denormalize)["vil"]
        target = denormalize_data_dict(data_dict={"vil": target.detach().float()}, method=self.denormalize)["vil"]

        for name, metric in self.metrics.items():
            metric.update(output, target)

        return loss

    def on_validation_epoch_end(self):
        self.compute_metrics("validation")

    def test_step(self, batch, batch_idx):
        input, target = batch
        output = self(input, target, compute_loss=False)

        output = denormalize_data_dict(data_dict={"vil": output.detach().float()}, method=self.denormalize)["vil"]
        target = denormalize_data_dict(data_dict={"vil": target.detach().float()}, method=self.denormalize)["vil"]

        for name, metric in self.metrics.items():
            metric.update(output, target)

    def on_test_epoch_end(self):
        self.compute_metrics("test")

    def compute_metrics(self, mode: str):
        for name, metric in self.metrics.items():
            metric_res = metric.compute()

            if isinstance(metric_res, dict):  # log dict from skill score metric
                self.log_score_epoch_end(metric_res, "val")
            else:
                self.log(f"{mode}_{name}_epoch", metric_res, on_epoch=True, logger=True)

            metric.reset()

    def log_score_epoch_end(self, score_dict: dict, mode: str = "val"):
        if mode == "val":
            log_mode_prefix = "valid"
        elif mode == "test":
            log_mode_prefix = "test"
        else:
            raise ValueError(f"Wrong mode {mode}. Must be 'val' or 'test'.")
        for metrics in self.skill_metrics:
            for thresh in self.thresholds:
                score_mean = np.mean(score_dict[thresh][metrics]).item()
                self.log(f"{log_mode_prefix}_{metrics}_{thresh}_epoch", score_mean)
            score_avg_mean = score_dict.get("avg", None)
            if score_avg_mean is not None:
                score_avg_mean = np.mean(score_avg_mean[metrics]).item()
                self.log(f"{log_mode_prefix}_{metrics}_avg_epoch", score_avg_mean)

    def infer(self, x, ret_np=True):
        """
        Run inference on input data.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor with a sequence of images.
        ret_np: bool
            Should numpy ndarray be returned

        Returns
        -------
        Output images (either as a Tenor or np.ndarray).
        """

        # change type
        if isinstance(x, torch.Tensor):
            _x = x.to(self.device).type(self.dtype)
        elif isinstance(x, np.ndarray):
            _x = torch.from_numpy(x).to(self.device).type(self.dtype)
        else:
            raise NotImplementedError(f"Type {type(x)} not supported.")

        with torch.no_grad():
            # build missing leading dimensions
            _count = 0
            while _x.ndim < self.ndim:
                _x = _x[None, ...]
                _count += 1

            # compute prediction using forward method
            pred = self(_x)

            # remove leading dimensions missing in the input
            for _ in range(_count):
                pred = pred.squeeze(dim=0)

        return pred.cpu().detach().numpy() if ret_np else pred.to(self.device)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
