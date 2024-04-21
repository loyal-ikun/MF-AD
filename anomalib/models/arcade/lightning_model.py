from __future__ import annotations

import logging

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.callbacks import Callback, EarlyStopping
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from torch import Tensor, optim
from torch.autograd import Variable

from anomalib.models.components import AnomalyModule
from anomalib.models.arcade.loss import DiscriminatorLoss, GeneratorLoss

from .torch_model import ArcadeModel

logger = logging.getLogger(__name__)


class Arcade(AnomalyModule):
    """PL Lightning Module for the ArcadeModel Algorithm.

    Args:
        batch_size (int): Batch size.
        input_size (tuple[int, int]): Input dimension.
        n_features (int): Number of features layers in the CNNs.
        latent_vec_size (int): Size of autoencoder latent vector.
        extra_layers (int, optional): Number of extra layers for encoder/decoder. Defaults to 0.
        add_final_conv_layer (bool, optional): Add convolution layer at the end. Defaults to True.
        wssim (int, optional): 
        wcritic (int, optional): 
        wGP (int, optional):
    """

    def __init__(
        self,
        batch_size: int,
        input_size: tuple[int, int],
        n_features: int,
        latent_vec_size: int,
        num_input_channels: int,
        extra_layers: int = 0,
        add_final_conv_layer: bool = True,
        wssim: int = 1,
        wcritic: int = 10,
        wGP: int = 100,
        lr: float = 0.0002,
        beta1: float = 0.5,
        beta2: float = 0.999,
    ) -> None:
        super().__init__()

        self.model: ArcadeModel = ArcadeModel(
            input_size=input_size,
            num_input_channels=num_input_channels,
            n_features=n_features,
            latent_vec_size=latent_vec_size,
            extra_layers=extra_layers,
            add_final_conv_layer=add_final_conv_layer,
        )

        self.real_label = torch.ones(size=(batch_size,), dtype=torch.float32)
        self.fake_label = torch.zeros(size=(batch_size,), dtype=torch.float32)

        self.min_scores: Tensor = torch.tensor(float("inf"), dtype=torch.float32)  # pylint: disable=not-callable
        self.max_scores: Tensor = torch.tensor(float("-inf"), dtype=torch.float32)  # pylint: disable=not-callable

        self.generator_loss = GeneratorLoss(wcritic=wcritic, wssim=wssim)
        self.discriminator_loss = DiscriminatorLoss(wGP=wGP)


        self.learning_rate = lr
        self.beta1 = beta1
        self.beta2 = beta2

    def _reset_min_max(self) -> None:
        """Resets min_max scores."""
        self.min_scores = torch.tensor(float("inf"), dtype=torch.float32)  # pylint: disable=not-callable
        self.max_scores = torch.tensor(float("-inf"), dtype=torch.float32)  # pylint: disable=not-callable

    def configure_optimizers(self) -> list[optim.Optimizer]:
        """Configures optimizers for each decoder.

        Note:
            This method is used for the existing CLI.
            When PL CLI is introduced, configure optimizers method will be
                deprecated, and optimizers will be configured from either
                config.yaml file or from CLI.

        Returns:
            Optimizer: Adam optimizer for each decoder
        """
        optimizer_d = optim.Adam(
            self.model.discriminator.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
        )
        optimizer_g = optim.Adam(
            self.model.generator.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
        )
        return [optimizer_d, optimizer_g]

    def training_step(
        self, batch: dict[str, str | Tensor], batch_idx: int, optimizer_idx: int
    ) -> STEP_OUTPUT:  # pylint: disable=arguments-differ
        """Training step.

        Args:
            batch (dict[str, str | Tensor]): Input batch containing images.
            batch_idx (int): Batch index.
            optimizer_idx (int): Optimizer which is being called for current training step.

        Returns:
            STEP_OUTPUT: Loss
        """
        del batch_idx  # `batch_idx` variables is not used.

        # forward pass
        x, fake = self.model(batch["data"])
        
        pred_real, _ = self.model.discriminator(x)

        if optimizer_idx == 0:  # Discriminator
            pred_fake, _ = self.model.discriminator(fake.detach())

            m = 64
            for i in range(m):
                batch_size = batch["data"].shape[0]
                eta = torch.FloatTensor(batch_size,1,1,1).uniform_(0,1)
                eta = eta.expand(batch_size, x.detach().size(1), x.detach().size(2), x.detach().size(3))
                eta = eta.cuda()

                interpolated = eta * x.detach() + ((1 - eta) * fake.detach())
                interpolated = interpolated.cuda()
                # define it to calculate gradient
                interpolated = Variable(interpolated, requires_grad=True)

                # calculate probability of interpolated examples
                prob_interpolated, _ = self.model.discriminator(interpolated)

                if i == 0:
                    loss = self.discriminator_loss(pred_real, pred_fake, interpolated=interpolated, prob_interpolated=prob_interpolated)
                else:
                    loss += self.discriminator_loss(pred_real, pred_fake, interpolated=interpolated, prob_interpolated=prob_interpolated)
                
        else:  # Generator
            pred_fake, _ = self.model.discriminator(fake)
            loss = self.generator_loss(x, fake, pred_fake)

        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def on_validation_start(self) -> None:
        """Reset min and max values for current validation epoch."""
        self._reset_min_max()
        return super().on_validation_start()

    def validation_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Update min and max scores from the current step.

        Args:
            batch (dict[str, str | Tensor]): Predicted difference between z and z_hat.

        Returns:
            (STEP_OUTPUT): Output predictions.
        """
        batch["pred_scores"] = self.model(batch["data"])
        self.max_scores = max(self.max_scores, torch.max(batch["pred_scores"]))
        self.min_scores = min(self.min_scores, torch.min(batch["pred_scores"]))
        return batch

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> EPOCH_OUTPUT:
        """Normalize outputs based on min/max values."""
        logger.info("Normalizing validation outputs based on min/max values.")
        for prediction in outputs:
            prediction["pred_scores"] = self._normalize(prediction["pred_scores"])
        super().validation_epoch_end(outputs)
        return outputs

    def on_test_start(self) -> None:
        """Reset min max values before test batch starts."""
        self._reset_min_max()
        return super().on_test_start()

    def test_step(self, batch: dict[str, str | Tensor], batch_idx: int, *args, **kwargs) -> STEP_OUTPUT:
        """Update min and max scores from the current step."""
        super().test_step(batch, batch_idx)
        self.max_scores = max(self.max_scores, torch.max(batch["pred_scores"]))
        self.min_scores = min(self.min_scores, torch.min(batch["pred_scores"]))
        return batch

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> EPOCH_OUTPUT:
        """Normalize outputs based on min/max values."""
        pred_label = []
        logger.info("Normalizing test outputs based on min/max values.")
        for prediction in outputs:
            prediction["pred_scores"] = self._normalize(prediction["pred_scores"])
            temp = prediction["pred_scores"] >= 0.5019
            pred_label.append(temp.int().cpu().numpy())
        # 0.9569-datacon   0.2935-IDS2017 0.5019-USTC
        np.savetxt(f'MMR_output_predict.csv', np.concatenate(pred_label)[np.newaxis, :], delimiter=',')
        
        super().test_epoch_end(outputs)
        return outputs

    def _normalize(self, scores: Tensor) -> Tensor:
        """Normalize the scores based on min/max of entire dataset.

        Args:
            scores (Tensor): Un-normalized scores.

        Returns:
            Tensor: Normalized scores.
        """
        scores = (scores - self.min_scores.to(scores.device)) / (
            self.max_scores.to(scores.device) - self.min_scores.to(scores.device)
        )
        return scores


class ArcadeLightning(Arcade):
    """PL Lightning Module for the Arcade Algorithm.

    Args:
        hparams (DictConfig | ListConfig): Model params
    """

    def __init__(self, hparams: DictConfig | ListConfig) -> None:
        super().__init__(
            batch_size=hparams.dataset.train_batch_size,
            num_input_channels=hparams.dataset.flow_num,
            input_size=hparams.model.input_size,
            n_features=hparams.model.n_features,
            latent_vec_size=hparams.model.latent_vec_size,
            extra_layers=hparams.model.extra_layers,
            add_final_conv_layer=hparams.model.add_final_conv,
            wssim=hparams.model.wssim,
            wcritic=hparams.model.wcritic,
            wGP=hparams.model.wGP,
            lr=hparams.model.lr,
            beta1=hparams.model.beta1,
            beta2=hparams.model.beta2,
        )
        self.hparams: DictConfig | ListConfig  # type: ignore
        self.save_hyperparameters(hparams)

    def configure_callbacks(self) -> list[Callback]:
        """Configure model-specific callbacks.

        Note:
            This method is used for the existing CLI.
            When PL CLI is introduced, configure callback method will be
                deprecated, and callbacks will be configured from either
                config.yaml file or from CLI.
        """
        early_stopping = EarlyStopping(
            monitor=self.hparams.model.early_stopping.metric,
            patience=self.hparams.model.early_stopping.patience,
            mode=self.hparams.model.early_stopping.mode,
        )
        return [early_stopping]
