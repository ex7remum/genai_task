from utils.class_registry import ClassRegistry
from utils.data_utils import move_batch_to_device, tensor2im
from training.trainers.base_trainer import BaseTrainer

from models.diffusion_models import diffusion_models_registry
from models.ddpm_dynamic import noise_scheduler_registry
from training.optimizers import optimizers_registry
from training.losses.diffusion_losses import DiffusionLossBuilder
import torch
import os
import numpy as np


diffusion_trainers_registry = ClassRegistry()


@diffusion_trainers_registry.add_to_registry(name="base_diffusion_trainer")
class BaseDiffusionTrainer(BaseTrainer):
    def setup_models(self):
        # do not forget to load state from checkpoints if provided
        self.unet = diffusion_models_registry[self.config.train.model](self.config.model_args).to(self.device)
        self.noise_scheduler = noise_scheduler_registry[self.config.train.noise_scheduler](self.config.ddpm_args).to(self.device)
        self.uncond_prob = self.config.train.uncond_prob
        if self.config.train.checkpoint_path:
            checkpoint = torch.load(self.config.train.checkpoint_path, map_location=self.device)
            self.start_step = checkpoint['step']
            self.step = self.start_step
            self.unet.load_state_dict(checkpoint['unet_state_dict'])
            self.noise_scheduler.load_state_dict(checkpoint['ddpm_state_dict'])


    def setup_optimizers(self):
        # do not forget to load state from checkpoints if provided
        self.optimizer = optimizers_registry[self.config.train.optimizer](self.unet.parameters(), **self.config.optimizer_args)
        if self.config.train.checkpoint_path:
            checkpoint = torch.load(self.config.train.checkpoint_path, map_location=self.device)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    def setup_losses(self):
        # setup loss
        self.loss_builder = DiffusionLossBuilder(self.config)


    def to_train(self):
        # all trainable modules to .train()
        self.unet.train()


    def to_eval(self):
        # all trainable modules to .eval()
        self.unet.eval()


    def train_step(self):
        # add noise to images according self.noise_scheduler
        # predict noise via self.unet
        # calculate losses, make optimizer step
        # return dict of losses to log
        images = next(self.train_dataloader)
        timesteps = self.noise_scheduler.sample_time_on_device(batch_size=images['images'].shape[0])
        real_noise = torch.randn_like(images['images'])
        labels = images['labels']

        if labels is not None:
            num_classes = self.train_dataset.get_num_classes()
            mask = np.random.choice(np.arange(2), replace=True, size=real_noise.shape[0], p=[1 - self.uncond_prob, self.uncond_prob]).astype(bool)
            labels[mask] = num_classes
            labels = (labels[:, None] == torch.arange(num_classes + 1, device=labels.device)[None, :]).float()

        batch = {
            'x_0': images['images'],
            'eps': real_noise,
            't': timesteps,
            'labels': labels
        }
        batch = move_batch_to_device(batch, self.device)

        self.optimizer.zero_grad()

        ddpm_out = self.noise_scheduler(batch)
        pred_noise = self.unet(ddpm_out['x_t'], batch['t'] / self.noise_scheduler.T, batch['labels'])


        batch_data = {"real_noise": ddpm_out['eps'],
                      "predicted_noise": pred_noise}
        
        log_dict = {
            'max_real_eps' : real_noise.max(),
            'min_real_eps' : real_noise.min(),
            'max_pred_eps' : pred_noise.max(),
            'min_pred_eps' : pred_noise.min() 
        }
        self.logger.logger.log_values(log_dict, self.step)
        batch_data = move_batch_to_device(batch_data, self.device)

        loss, losses_dict = self.loss_builder.calculate_loss(batch_data)
        loss.backward()

        self.optimizer.step()
        return losses_dict


    def save_checkpoint(self):
        # save all necessary parts of your pipeline
        os.makedirs(f'{self.experiment_dir}/checkpoints', mode=0o777, exist_ok=True)
        torch.save({
            'step': self.step,
            'unet_state_dict': self.unet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'ddpm_state_dict': self.noise_scheduler.state_dict(),
            'logger': self.logger
            }, f'{self.experiment_dir}/checkpoints/model_step_{self.step}.pt')


    def sample_image(self, batch_size, labels=None):
        shape = (batch_size, 3, 64, 64)
        x_t = torch.randn(shape).to(self.device)

        
        num_classes = self.train_dataset.get_num_classes()
        if labels is None:
            labels = torch.randint(0, num_classes, (batch_size,), device=self.device)
        labels = (labels[:, None] == torch.arange(num_classes + 1, device=x_t.device)[None, :]).float()

        for t in range(self.noise_scheduler.T - 1, -1, -1):
            t_tensor = torch.ones(batch_size, dtype=torch.int64, device=x_t.device) * t
            model_output = self.unet(x_t, t_tensor / self.noise_scheduler.T, labels)
            if self.config.train.gamma != 0:
                fake_labels = torch.ones((batch_size, ), device=self.device) * num_classes
                fake_labels = (fake_labels[:, None] == torch.arange(num_classes + 1, device=x_t.device)[None, :]).float()
                uncond_gen = self.unet(x_t, t_tensor / self.noise_scheduler.T, fake_labels)
                model_output = (1 + self.config.train.gamma) * model_output - self.config.train.gamma * uncond_gen
            x_0 = self.noise_scheduler.get_x_zero(x_t, model_output, t_tensor)
            x_t = self.noise_scheduler.sample_from_posterior_q(x_t, x_0, t_tensor)
        return x_t


    def synthesize_images(self):
        # synthesize images and save to self.experiment_dir/images
        # synthesized additional batch of images to log
        # return batch_of_images, path_to_saved_pics, 
        num_images = 20 * 101
        path = f'{self.experiment_dir}/images'
        os.makedirs(path, mode=0o777, exist_ok=True)

        all_labels = torch.ones(num_images)
        for i in range(101):
            all_labels[i * 20: (i + 1) * 20] = i


        for idx in range(0, num_images, self.config.data.val_batch_size):
           num_img_to_gen = min(self.config.data.val_batch_size, num_images - idx)
           cur_labels =  (all_labels[idx: idx + num_img_to_gen]).to(self.device)
           gen_imgs = self.sample_image(num_img_to_gen, cur_labels)
           for img_id, img in enumerate(gen_imgs):
               cur_path = f'{path}/sample_{idx + img_id}.jpg'
               cur_img = tensor2im(img)
               cur_img = cur_img.save(cur_path)

        visualize_img = self.sample_image(16)
        return visualize_img, path
