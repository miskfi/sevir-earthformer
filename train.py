import argparse

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import CSVLogger

from src.dataset.sevir_data_module import SEVIRIndividualDataModule
from src.models.earthformer.earthformer_pl_module import EarthformerPLModule
from src.models.unet.unet_pl_module import UNetPLModule


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="path to the config file")
    parser.add_argument("--gpu", type=str, required=True, help="GPU number ('cpu' for CPU)")

    opt = parser.parse_args()
    opt.gpu = int(opt.gpu) if opt.gpu not in ("cpu", "CPU") else -1
    return opt


def train(options):
    with open(options.config, "r") as f:
        config = yaml.safe_load(f)

    dataset = SEVIRIndividualDataModule(**config["dataset"])

    accelerator = "gpu" if options.gpu != -1 else "cpu"

    if config["model_name"] in ["u-net", "U-Net", "unet"]:
        model = UNetPLModule(
            **config["model"],
            device=f"cuda:{options.gpu}" if accelerator == "gpu" else "cpu",
            denormalization_method=config["dataset"].get("normalization_method", None),
        )
    elif config["model_name"] in ["earthformer", "Earthformer"]:
        model = EarthformerPLModule(
            total_num_steps=config["training"]["max_epochs"]
            * dataset.num_train_samples
            / config["dataset"]["batch_size"],
            oc_file=options.config,
        )
    else:
        raise ValueError(f"Model {config['model_name']} not available")

    seed_everything(374)
    torch.set_float32_matmul_precision("high")

    early_stopping = pl.callbacks.EarlyStopping(monitor="validation_loss", mode="min", patience=5, verbose=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="validation_loss", save_top_k=1, mode="min")

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=[options.gpu] if accelerator == "gpu" else 1,
        callbacks=[early_stopping, checkpoint_callback],
        logger=CSVLogger("logs", config["model_name"]),
        default_root_dir="checkpoints",
        **config["training"],
    )

    trainer.fit(model, dataset)
    print(checkpoint_callback.best_model_path)


if __name__ == "__main__":
    options = parse_args()
    train(options)
