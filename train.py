import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from module import SIGReg
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # Check for NaNs in predictions or targets
    if torch.isnan(pred_emb).any() or torch.isnan(tgt_emb).any():
        print(f"\n[NaN Warning] pred_emb or tgt_emb contains NaNs before loss computation! (stage={stage})")

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  

    # Check for NaN in final loss
    if torch.isnan(output["loss"]):
        print(f"\n[NaN Loss Detected] stage={stage}:")
        print(f"  pred_loss: {output['pred_loss'].item():.6f}")
        print(f"  sigreg_loss: {output['sigreg_loss'].item():.6f}")

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True, prog_bar=True)
    return output


class DetectAnomalyCallback(pl.Callback):
    """Callback to monitor gradients and parameter weights for NaNs."""

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # outputs can be a dict from training_step (lejepa_forward) or a tensor
        loss = None
        if isinstance(outputs, dict):
            loss = outputs.get("loss")
        elif torch.is_tensor(outputs):
            loss = outputs
            
        if loss is not None and torch.isnan(loss).any():
            print(f"\n[NaN Loss Detected] in trainer at epoch {trainer.current_epoch}, batch {batch_idx}!")

        # Check parameter weights for NaNs
        nan_params = []
        for name, param in pl_module.named_parameters():
            if torch.isnan(param).any():
                nan_params.append(name)
        if nan_params:
            print(f"\n[NaN Weight Detected] at epoch {trainer.current_epoch}, batch {batch_idx}! The following weights are NaN:")
            for name in nan_params:
                print(f"  - {name}")

    def on_after_backward(self, trainer, pl_module):
        # Check gradients for NaNs
        nan_grads = []
        for name, param in pl_module.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                nan_grads.append(name)
        if nan_grads:
            print(f"\n[NaN Gradient Detected] during backward pass! Gradients for the following parameters are NaN:")
            for name in nan_grads:
                print(f"  - {name}")


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1,
    )
    anomaly_detector = DetectAnomalyCallback()

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback, anomaly_detector],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == "__main__":
    run()
