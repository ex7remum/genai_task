import torch
import os

from abc import abstractmethod
from datasets.dataloaders import InfiniteLoader
from training.loggers import TrainingLogger
from datasets.datasets import datasets_registry
from metrics.metrics import metrics_registry
from utils.data_utils import preprocess_image
from PIL import Image


class BaseTrainer:
    def __init__(self, config):
        self.config = config

        self.device = config.exp.device
        self.start_step = config.train.start_step
        self.step = 0
    

    def setup(self):
        self.setup_experiment_dir()

        self.setup_models()
        self.setup_optimizers()
        self.setup_losses()

        self.setup_metrics()
        self.setup_logger()

        self.setup_datasets()
        self.setup_dataloaders()


    def setup_inference(self):
        self.setup_experiment_dir()

        self.setup_models()

        self.setup_metrics()
        self.setup_logger()

        self.setup_datasets()
        self.setup_dataloaders()


    @abstractmethod
    def setup_models(self):
        pass

    @abstractmethod
    def setup_optimizers(self):
        pass

    @abstractmethod
    def setup_losses(self):
        pass

    @abstractmethod
    def to_train(self):
        pass

    @abstractmethod
    def to_eval(self):
        pass

    def setup_experiment_dir(self):
        self.experiment_dir = self.config.exp.exp_dir
        os.makedirs(self.experiment_dir, mode=0o777, exist_ok=True)

    def setup_metrics(self):
        self.metrics = []
        for metric_name in self.config.train.val_metrics:
            self.metrics.append(metrics_registry[metric_name]())

    def setup_logger(self):
        self.logger = TrainingLogger(self.config)

    def setup_datasets(self):
        self.train_dataset = datasets_registry[self.config.train.dataset_name](self.config.data.input_train_dir, transforms=preprocess_image)

        dir = self.config.data.input_val_dir
        self.fid_dir = f'{self.experiment_dir}/val_fid'
        os.makedirs(self.fid_dir, mode=0o777, exist_ok=True)
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                    path = os.path.join(root, fname)
                    image = Image.open(path).convert("RGB")
                    image.save(f'{self.fid_dir}/{fname}')

    def setup_dataloaders(self):
        self.train_dataloader = InfiniteLoader(self.train_dataset, 
                                               num_workers=self.config.data.workers,
                                               batch_size=self.config.data.train_batch_size,
                                               drop_last=self.config.data.drop_last,
                                               shuffle=self.config.data.shuffle)


    def training_loop(self):
        self.to_train()

        for self.step in range(self.start_step, self.config.train.steps + 1):
            losses_dict = self.train_step()
            self.logger.update_losses(losses_dict)

            if self.step and self.step % self.config.train.val_step == 0:
                val_metrics_dict, images = self.validate()

                self.logger.log_val_metrics(val_metrics_dict, step=self.step)
                self.logger.log_batch_of_images(images, step=self.step, images_type="validation")

            if self.step and self.step % self.config.train.log_step == 0:
                self.logger.log_train_losses(self.step)

            if self.step and self.step % self.config.train.checkpoint_step == 0:
                self.save_checkpoint()


    @abstractmethod
    def train_step(self):
        pass

    @abstractmethod
    def save_checkpoint(self):
        pass


    @torch.no_grad()
    def validate(self):
        self.to_eval()
        images_sample, images_pth = self.synthesize_images()

        metrics_dict = {}
        for metric in self.metrics:
            metrics_dict[metric.get_name()] = metric(
                orig_path=self.fid_dir, 
                synt_path=images_pth
            )
        return metrics_dict, images_sample


    @abstractmethod
    def synthesize_images(self):
        pass


    @torch.no_grad()
    def inference(self):
        self.validate()
