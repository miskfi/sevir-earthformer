"""Code adapted from https://github.com/amazon-science/earth-forecasting-transformer/blob/main/scripts/cuboid_transformer/sevir/train_cuboid_sevir.py"""

from typing import Dict

import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
from einops import rearrange
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from src.dataset.preprocessing import denormalize_data_dict
from src.metrics.sevir_skill_score import SEVIRSkillScore
from src.models.earthformer.cuboid_transformer import CuboidTransformerModel
from src.utils.optim import SequentialLR, warmup_lambda
from src.utils.utils import get_parameter_names


class EarthformerPLModule(pl.LightningModule):
    def __init__(
        self,
        total_num_steps: int,
        oc_file: str = None,
    ):
        super(EarthformerPLModule, self).__init__()
        if oc_file is not None:
            oc_from_file = OmegaConf.load(open(oc_file, "r"))
        else:
            oc_from_file = None
        oc = self.get_base_config(oc_from_file=oc_from_file)
        model_cfg = OmegaConf.to_object(oc.model)
        num_blocks = len(model_cfg["enc_depth"])
        if isinstance(model_cfg["self_pattern"], str):
            enc_attn_patterns = [model_cfg["self_pattern"]] * num_blocks
        else:
            enc_attn_patterns = OmegaConf.to_container(model_cfg["self_pattern"])
        if isinstance(model_cfg["cross_self_pattern"], str):
            dec_self_attn_patterns = [model_cfg["cross_self_pattern"]] * num_blocks
        else:
            dec_self_attn_patterns = OmegaConf.to_container(model_cfg["cross_self_pattern"])
        if isinstance(model_cfg["cross_pattern"], str):
            dec_cross_attn_patterns = [model_cfg["cross_pattern"]] * num_blocks
        else:
            dec_cross_attn_patterns = OmegaConf.to_container(model_cfg["cross_pattern"])

        self.torch_nn_module = CuboidTransformerModel(
            input_shape=model_cfg["input_shape"],
            target_shape=model_cfg["target_shape"],
            base_units=model_cfg["base_units"],
            block_units=model_cfg["block_units"],
            scale_alpha=model_cfg["scale_alpha"],
            enc_depth=model_cfg["enc_depth"],
            dec_depth=model_cfg["dec_depth"],
            enc_use_inter_ffn=model_cfg["enc_use_inter_ffn"],
            dec_use_inter_ffn=model_cfg["dec_use_inter_ffn"],
            dec_hierarchical_pos_embed=model_cfg["dec_hierarchical_pos_embed"],
            downsample=model_cfg["downsample"],
            downsample_type=model_cfg["downsample_type"],
            enc_attn_patterns=enc_attn_patterns,
            dec_self_attn_patterns=dec_self_attn_patterns,
            dec_cross_attn_patterns=dec_cross_attn_patterns,
            dec_cross_last_n_frames=model_cfg["dec_cross_last_n_frames"],
            dec_use_first_self_attn=model_cfg["dec_use_first_self_attn"],
            num_heads=model_cfg["num_heads"],
            attn_drop=model_cfg["attn_drop"],
            proj_drop=model_cfg["proj_drop"],
            ffn_drop=model_cfg["ffn_drop"],
            upsample_type=model_cfg["upsample_type"],
            ffn_activation=model_cfg["ffn_activation"],
            gated_ffn=model_cfg["gated_ffn"],
            norm_layer=model_cfg["norm_layer"],
            # global vectors
            num_global_vectors=model_cfg["num_global_vectors"],
            use_dec_self_global=model_cfg["use_dec_self_global"],
            dec_self_update_global=model_cfg["dec_self_update_global"],
            use_dec_cross_global=model_cfg["use_dec_cross_global"],
            use_global_vector_ffn=model_cfg["use_global_vector_ffn"],
            use_global_self_attn=model_cfg["use_global_self_attn"],
            separate_global_qkv=model_cfg["separate_global_qkv"],
            global_dim_ratio=model_cfg["global_dim_ratio"],
            # initial_downsample
            initial_downsample_type=model_cfg["initial_downsample_type"],
            initial_downsample_activation=model_cfg["initial_downsample_activation"],
            # initial_downsample_type=="stack_conv"
            initial_downsample_stack_conv_num_layers=model_cfg["initial_downsample_stack_conv_num_layers"],
            initial_downsample_stack_conv_dim_list=model_cfg["initial_downsample_stack_conv_dim_list"],
            initial_downsample_stack_conv_downscale_list=model_cfg["initial_downsample_stack_conv_downscale_list"],
            initial_downsample_stack_conv_num_conv_list=model_cfg["initial_downsample_stack_conv_num_conv_list"],
            # misc
            padding_type=model_cfg["padding_type"],
            z_init_method=model_cfg["z_init_method"],
            checkpoint_level=model_cfg["checkpoint_level"],
            pos_embed_type=model_cfg["pos_embed_type"],
            use_relative_pos=model_cfg["use_relative_pos"],
            self_attn_use_final_proj=model_cfg["self_attn_use_final_proj"],
            # initialization
            attn_linear_init_mode=model_cfg["attn_linear_init_mode"],
            ffn_linear_init_mode=model_cfg["ffn_linear_init_mode"],
            conv_init_mode=model_cfg["conv_init_mode"],
            down_up_linear_init_mode=model_cfg["down_up_linear_init_mode"],
            norm_init_mode=model_cfg["norm_init_mode"],
        )

        self.total_num_steps = total_num_steps
        if oc_file is not None:
            oc_from_file = OmegaConf.load(open(oc_file, "r"))
        else:
            oc_from_file = None
        oc = self.get_base_config(oc_from_file=oc_from_file)
        self.save_hyperparameters(oc)
        self.oc = oc
        # layout
        self.in_len = oc.layout.in_len
        self.out_len = oc.layout.out_len
        self.layout = oc.layout.layout
        # optimization
        self.max_epochs = oc.optim.max_epochs
        self.optim_method = oc.optim.method
        self.lr = oc.optim.lr
        self.wd = oc.optim.wd
        # lr_scheduler
        self.total_num_steps = total_num_steps
        self.lr_scheduler_mode = oc.optim.lr_scheduler_mode
        self.warmup_percentage = oc.optim.warmup_percentage
        self.min_lr_ratio = oc.optim.min_lr_ratio
        # logging
        self.logging_prefix = oc.logging.logging_prefix
        # visualization
        self.train_example_data_idx_list = list(oc.vis.train_example_data_idx_list)
        self.val_example_data_idx_list = list(oc.vis.val_example_data_idx_list)
        self.test_example_data_idx_list = list(oc.vis.test_example_data_idx_list)
        self.eval_example_only = oc.vis.eval_example_only

        # evaluation
        self.metrics_list = oc.dataset.metrics_list
        self.threshold_list = oc.dataset.threshold_list
        self.metrics_mode = oc.dataset.metrics_mode
        self.denormalize = oc.dataset.normalization_method
        self.valid_mse = torchmetrics.MeanSquaredError()
        self.valid_mae = torchmetrics.MeanAbsoluteError()
        self.valid_score = SEVIRSkillScore(
            denormalize_method=self.denormalize,
            threshold_list=self.threshold_list,
            metrics_list=self.metrics_list,
            eps=1e-4,
        )
        self.valid_ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=255)
        self.test_mse = torchmetrics.MeanSquaredError()
        self.test_mae = torchmetrics.MeanAbsoluteError()
        self.test_score = SEVIRSkillScore(
            denormalize_method=self.denormalize,
            threshold_list=self.threshold_list,
            metrics_list=self.metrics_list,
            eps=1e-4,
        )
        self.test_ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=255)

        self.precision = 32
        self.ndim = 5

    def get_base_config(self, oc_from_file=None):
        oc = OmegaConf.create()
        oc.dataset = self.get_dataset_config()
        oc.layout = self.get_layout_config()
        oc.optim = self.get_optim_config()
        oc.logging = self.get_logging_config()
        oc.trainer = self.get_trainer_config()
        oc.vis = self.get_vis_config()
        oc.model = self.get_model_config()
        if oc_from_file is not None:
            oc = OmegaConf.merge(oc, oc_from_file)
        return oc

    @staticmethod
    def get_dataset_config():
        oc = OmegaConf.create()
        oc.dataset_name = "sevir"
        oc.img_height = 384
        oc.img_width = 384
        oc.in_len = 13
        oc.out_len = 12
        oc.seq_len = 25
        oc.plot_stride = 2
        oc.interval_real_time = 5
        oc.sample_mode = "sequent"
        oc.stride = oc.out_len
        oc.layout = "NTHWC"
        oc.start_date = None
        oc.train_val_split_date = (2019, 1, 1)
        oc.train_test_split_date = (2019, 6, 1)
        oc.end_date = None
        oc.metrics_mode = "0"
        oc.metrics_list = ("csi", "pod", "sucr", "bias")
        oc.threshold_list = (16, 74, 133, 160, 181, 219)
        oc.normalization_method = "sevir"
        return oc

    @classmethod
    def get_model_config(cls):
        cfg = OmegaConf.create()
        dataset_oc = cls.get_dataset_config()
        height = dataset_oc.img_height
        width = dataset_oc.img_width
        in_len = dataset_oc.in_len
        out_len = dataset_oc.out_len
        data_channels = 1
        cfg.input_shape = (in_len, height, width, data_channels)
        cfg.target_shape = (out_len, height, width, data_channels)

        cfg.base_units = 64
        cfg.block_units = None  # multiply by 2 when downsampling in each layer
        cfg.scale_alpha = 1.0

        cfg.enc_depth = [1, 1]
        cfg.dec_depth = [1, 1]
        cfg.enc_use_inter_ffn = True
        cfg.dec_use_inter_ffn = True
        cfg.dec_hierarchical_pos_embed = True

        cfg.downsample = 2
        cfg.downsample_type = "patch_merge"
        cfg.upsample_type = "upsample"

        cfg.num_global_vectors = 8
        cfg.use_dec_self_global = True
        cfg.dec_self_update_global = True
        cfg.use_dec_cross_global = True
        cfg.use_global_vector_ffn = True
        cfg.use_global_self_attn = False
        cfg.separate_global_qkv = False
        cfg.global_dim_ratio = 1

        cfg.self_pattern = "axial"
        cfg.cross_self_pattern = "axial"
        cfg.cross_pattern = "cross_1x1"
        cfg.dec_cross_last_n_frames = None

        cfg.attn_drop = 0.1
        cfg.proj_drop = 0.1
        cfg.ffn_drop = 0.1
        cfg.num_heads = 4

        cfg.ffn_activation = "gelu"
        cfg.gated_ffn = False
        cfg.norm_layer = "layer_norm"
        cfg.padding_type = "zeros"
        cfg.pos_embed_type = "t+hw"
        cfg.use_relative_pos = True
        cfg.self_attn_use_final_proj = True
        cfg.dec_use_first_self_attn = False

        cfg.z_init_method = "zeros"
        cfg.checkpoint_level = 2
        # initial downsample and final upsample
        cfg.initial_downsample_type = "stack_conv"
        cfg.initial_downsample_activation = "leaky"
        cfg.initial_downsample_stack_conv_num_layers = 3
        cfg.initial_downsample_stack_conv_dim_list = [4, 16, cfg.base_units]
        cfg.initial_downsample_stack_conv_downscale_list = [3, 2, 2]
        cfg.initial_downsample_stack_conv_num_conv_list = [2, 2, 2]
        # initialization
        cfg.attn_linear_init_mode = "0"
        cfg.ffn_linear_init_mode = "0"
        cfg.conv_init_mode = "0"
        cfg.down_up_linear_init_mode = "0"
        cfg.norm_init_mode = "0"
        return cfg

    @classmethod
    def get_layout_config(cls):
        oc = OmegaConf.create()
        dataset_oc = cls.get_dataset_config()
        oc.in_len = dataset_oc.in_len
        oc.out_len = dataset_oc.out_len
        oc.layout = dataset_oc.layout
        return oc

    @staticmethod
    def get_optim_config():
        oc = OmegaConf.create()
        oc.seed = None
        oc.total_batch_size = 32
        oc.micro_batch_size = 8

        oc.method = "adamw"
        oc.lr = 1e-3
        oc.wd = 1e-5
        oc.gradient_clip_val = 1.0
        oc.max_epochs = 100
        # scheduler
        oc.warmup_percentage = 0.2
        oc.lr_scheduler_mode = "cosine"  # Can be strings like 'linear', 'cosine', 'platue'
        oc.min_lr_ratio = 0.1
        oc.warmup_min_lr_ratio = 0.1
        # early stopping
        oc.early_stop = False
        oc.early_stop_mode = "min"
        oc.early_stop_patience = 20
        oc.save_top_k = 1
        return oc

    @staticmethod
    def get_logging_config():
        oc = OmegaConf.create()
        oc.logging_prefix = "SEVIR"
        oc.monitor_lr = True
        oc.monitor_device = False
        oc.track_grad_norm = -1
        oc.use_wandb = True
        return oc

    @staticmethod
    def get_trainer_config():
        oc = OmegaConf.create()
        oc.check_val_every_n_epoch = 1
        oc.log_step_ratio = 0.001  # Logging every 1% of the total training steps per epoch
        oc.precision = 32
        return oc

    @classmethod
    def get_vis_config(cls):
        oc = OmegaConf.create()
        dataset_oc = cls.get_dataset_config()
        oc.train_example_data_idx_list = [
            0,
        ]
        oc.val_example_data_idx_list = [
            80,
        ]
        oc.test_example_data_idx_list = [0, 80, 160, 240, 320, 400]
        oc.eval_example_only = False
        oc.plot_stride = dataset_oc.plot_stride
        return oc

    def configure_optimizers(self):
        # Configure the optimizer. Disable the weight decay for layer norm weights and all bias terms.
        decay_parameters = get_parameter_names(self.torch_nn_module, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.torch_nn_module.named_parameters() if n in decay_parameters],
                "weight_decay": self.oc.optim.wd,
            },
            {
                "params": [p for n, p in self.torch_nn_module.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]

        if self.oc.optim.method == "adamw":
            optimizer = torch.optim.AdamW(
                params=optimizer_grouped_parameters, lr=self.oc.optim.lr, weight_decay=self.oc.optim.wd
            )
        else:
            raise NotImplementedError

        warmup_iter = int(np.round(self.oc.optim.warmup_percentage * self.total_num_steps))

        if self.oc.optim.lr_scheduler_mode == "cosine":
            warmup_scheduler = LambdaLR(
                optimizer,
                lr_lambda=warmup_lambda(warmup_steps=warmup_iter, min_lr_ratio=self.oc.optim.warmup_min_lr_ratio),
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=(self.total_num_steps - warmup_iter),
                eta_min=self.oc.optim.min_lr_ratio * self.oc.optim.lr,
            )
            lr_scheduler = SequentialLR(
                optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_iter]
            )
            lr_scheduler_config = {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            }
        else:
            raise NotImplementedError
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    @classmethod
    def get_total_num_steps(cls, num_samples: int, total_batch_size: int, epoch: int = None):
        r"""
        Parameters
        ----------
        num_samples:    int
            The number of samples of the datasets. `num_samples / micro_batch_size` is the number of steps per epoch.
        total_batch_size:   int
            `total_batch_size == micro_batch_size * world_size * grad_accum`
        """
        if epoch is None:
            epoch = cls.get_optim_config().max_epochs
        return int(epoch * num_samples / total_batch_size)

    def forward(self, in_seq, out_seq=None, compute_loss=True):
        output = self.torch_nn_module(in_seq)

        if compute_loss:
            loss = F.mse_loss(output, out_seq)
        else:
            loss = None

        return output, loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = rearrange(x.unsqueeze(1), "b t c h w -> b t h w c")
        y = rearrange(y.unsqueeze(1), "b t c h w -> b t h w c")

        y_hat, loss = self(x, y)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = rearrange(x.unsqueeze(1), "b t c h w -> b t h w c")
        y = rearrange(y.unsqueeze(1), "b t c h w -> b t h w c")

        micro_batch_size = x.shape[0]
        data_idx = int(batch_idx * micro_batch_size)
        if not self.eval_example_only or data_idx in self.val_example_data_idx_list:
            y_hat, _ = self(x, y)
            if self.precision == 16:
                y_hat = y_hat.float()

            y = denormalize_data_dict(data_dict={"vil": y.detach().float()}, method=self.denormalize)["vil"]
            y_hat = denormalize_data_dict(data_dict={"vil": y_hat.detach().float()}, method=self.denormalize)["vil"]

            step_mse = self.valid_mse(y_hat, y)
            step_mae = self.valid_mae(y_hat, y)
            self.valid_score.update(y_hat, y)
            self.log("validation_mse_step", step_mse, prog_bar=True, on_step=True, on_epoch=False)
            self.log("validation_mae_step", step_mae, prog_bar=True, on_step=True, on_epoch=False)
        return None

    def on_validation_epoch_end(self):
        valid_mse = self.valid_mse.compute()
        valid_mae = self.valid_mae.compute()
        self.log("validation_mse_epoch", valid_mse, prog_bar=True, on_step=False, on_epoch=True)
        self.log("validation_mae_epoch", valid_mae, prog_bar=True, on_step=False, on_epoch=True)
        self.valid_mse.reset()
        self.valid_mae.reset()
        valid_score = self.valid_score.compute()
        # self.log("validation_loss", -valid_score["avg"]["csi"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("validation_loss", valid_mse, prog_bar=True, on_step=False, on_epoch=True)
        self.log_score_epoch_end(score_dict=valid_score, mode="val")
        self.valid_score.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = rearrange(x.unsqueeze(1), "b t c h w -> b t h w c")
        y = rearrange(y.unsqueeze(1), "b t c h w -> b t h w c")

        micro_batch_size = x.shape[0]
        data_idx = int(batch_idx * micro_batch_size)
        if not self.eval_example_only or data_idx in self.test_example_data_idx_list:
            y_hat, _ = self(x, y)
            if self.precision == 16:
                y_hat = y_hat.float()

            y = denormalize_data_dict(data_dict={"vil": y.detach().float()}, method=self.denormalize)["vil"]
            y_hat = denormalize_data_dict(data_dict={"vil": y_hat.detach().float()}, method=self.denormalize)["vil"]

            step_mse = self.test_mse(y_hat, y)
            step_mae = self.test_mae(y_hat, y)
            step_ssim = self.test_ssim(y_hat.squeeze(4), y.squeeze(4))
            self.test_score.update(y_hat, y)
            self.log("test_mse_step", step_mse, prog_bar=True, on_step=True, on_epoch=False)
            self.log("test_mae_step", step_mae, prog_bar=True, on_step=True, on_epoch=False)
            self.log("test_ssim_step", step_ssim, prog_bar=True, on_step=True, on_epoch=False)
        return None

    def on_test_epoch_end(self):
        test_mse = self.test_mse.compute()
        test_mae = self.test_mae.compute()
        test_ssim = self.test_ssim.compute()
        self.log("test_mse_epoch", test_mse, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_mae_epoch", test_mae, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_ssim_epoch", test_ssim, prog_bar=True, on_step=False, on_epoch=True)
        self.test_mse.reset()
        self.test_mae.reset()
        self.test_ssim.reset()
        test_score = self.test_score.compute()
        self.log_score_epoch_end(score_dict=test_score, mode="test")
        self.test_score.reset()

    def log_score_epoch_end(self, score_dict: Dict, mode: str = "val"):
        if mode == "val":
            log_mode_prefix = "valid"
        elif mode == "test":
            log_mode_prefix = "test"
        else:
            raise ValueError(f"Wrong mode {mode}. Must be 'val' or 'test'.")
        for metrics in self.metrics_list:
            for thresh in self.threshold_list:
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

            _x = rearrange(_x, "b t c h w -> b t h w c")

            # compute prediction using forward method
            pred, _ = self(_x, compute_loss=False)

            _x = rearrange(_x, "b t h w c -> b t c h w")

            # remove leading dimensions missing in the input
            for _ in range(_count):
                pred = pred.squeeze(dim=0)

        return pred.cpu().detach().numpy() if ret_np else pred.to(self.device)
