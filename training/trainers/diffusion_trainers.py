from utils.class_registry import ClassRegistry
from utils.data_utils import move_batch_to_device, tensor2im
from training.trainers.base_trainer import BaseTrainer

from models.diffusion_models import diffusion_models_registry
from models.ddpm_dynamic import noise_scheduler_registry
from training.optimizers import optimizers_registry
from training.losses.diffusion_losses import DiffusionLossBuilder
import torch
import os



diffusion_trainers_registry = ClassRegistry()


@diffusion_trainers_registry.add_to_registry(name="base_diffusion_trainer")
class BaseDiffusionTrainer(BaseTrainer):
    def setup_models(self):
        # do not forget to load state from checkpoints if provided
        self.unet = diffusion_models_registry[self.config.train.model](self.config.model_args)
        self.noise_scheduler = noise_scheduler_registry[self.config.train.noise_scheduler](self.config.ddpm_args)


    def setup_optimizers(self):
        # do not forget to load state from checkpoints if provided
        self.optimizer = optimizers_registry[self.config.train.optimizer](self.unet.parameters(), **self.config.optimizer_args)


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

        batch = {
            'x_0': images['images'],
            'eps': real_noise,
            't': timesteps
        }
        batch = move_batch_to_device(batch, self.device)

        self.optimizer.zero_grad()

        ddpm_out = self.noise_scheduler(batch)
        pred_noise = self.unet(ddpm_out['x_t'], batch['t'])

        batch_data = {"real_noise": ddpm_out['eps'],
                      "predicted_noise": pred_noise}
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
            'loss_builder': self.loss_builder,
            'config': self.config,
            'logger':self.logger
            }, f'{self.experiment_dir}/checkpoints/model_step_{self.step}.pt')


    def sample_image(self, batch_size):
        shape = (batch_size, 3, 64, 64)
        x_t = torch.randn(shape).to(self.device)

        for t in range(self.noise_scheduler.T - 1, -1, -1):
            t_tensor = torch.ones(batch_size, dtype=torch.int64, device=x_t.device) * t
            model_output = self.unet(x_t, t_tensor)
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


        for idx in range(0, num_images, self.config.data.val_batch_size):
            num_img_to_gen = min(self.config.data.val_batch_size, num_images - idx)
            gen_imgs = self.sample_image(num_img_to_gen)
            for img_id, img in enumerate(gen_imgs):
                cur_path = f'{path}/sample_{idx + img_id}.jpg'
                cur_img = tensor2im(img)
                cur_img = cur_img.save(cur_path)

        visualize_img = self.sample_image(16)
        return visualize_img, path
