import wandb
import torch
from PIL import Image
from collections import defaultdict
from statistics import mean
import torchvision
import numpy as np
import os
from omegaconf import OmegaConf


class WandbLogger:
    def __init__(self, config):
        wandb.login(key=os.environ['WANDB_KEY'].strip())
        if config.train.checkpoint_path:
            # resume training run from checkpoint
            checkpoint = torch.load(config.train.checkpoint_path, map_location=config.exp.device)
            old_logger = checkpoint['logger']
            self.wandb_args = {
                "id": old_logger.logger.wandb_args['id'],
                "project": old_logger.logger.wandb_args['project'],
                "name": old_logger.logger.wandb_args['name'],
                "config": old_logger.logger.wandb_args['config'],
            }
        else:
            # create new wandb run and save args, config and etc.
            self.wandb_args = {
                "id": wandb.util.generate_id(),
                "project": config.exp.project_name,
                "name": config.exp.run_name,
                "config": OmegaConf.to_container(config),
            }
        wandb.init(**self.wandb_args, resume="allow")


    @staticmethod
    def log_values(values_dict: dict, step: int):
        # log values to wandb
        for key, value in values_dict.items():
            wandb.log({key : value}, step=step)


    @staticmethod
    def log_images(images: torch.Tensor, step: int):
        # log images
        nrow = 4
        var = images.cpu().detach()
        var = (var + 1) * 127.5
        var = torch.clip(var, 0, 255)
        grid = torchvision.utils.make_grid(var, nrow=nrow).permute(1, 2, 0)
        grid = grid.data.numpy().astype(np.uint8)
        wandb.log({'gen_image': wandb.Image(grid)}, step=step)


class TrainingLogger:
    def __init__(self, config):
        self.logger = WandbLogger(config)
        self.losses_memory = defaultdict(list)


    def log_train_losses(self, step: int):
        # average losses in losses_memory
        # log them and clear losses_memory
        res_dict = {}
        for loss_name, loss_vals in self.losses_memory.items():
            avg_train_loss = mean(loss_vals)
            res_dict[loss_name] = avg_train_loss
        self.losses_memory.clear()
        self.logger.log_values(res_dict, step)


    def log_val_metrics(self, val_metrics: dict, step: int):
        self.logger.log_values(val_metrics, step)


    def log_batch_of_images(self, batch: torch.Tensor, step: int, images_type: str = ""):
        self.logger.log_images(batch, step)


    def update_losses(self, losses_dict):
        # it is useful to average losses over a number of steps rather than track them at each step
        # this makes training curves smoother
        for loss_name, loss_val in losses_dict.items():
            self.losses_memory[loss_name].append(loss_val)
