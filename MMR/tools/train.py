import random

import numpy as np
import torch
import logging

from utils import get_train_dataloaders, load_backbones
from utils.load_dataset import get_test_dataloaders
from utils.common import freeze_paras, scratch_MAE_decoder

from models.MMR import MMR_base, MMR_pipeline_

import timm.optim.optim_factory as optim_factory

LOGGER = logging.getLogger(__name__)


def train(cfg=None):
    """
    include data loader load, model load, optimizer, training and test.
    """
    # Set random seed from configs.
    random.seed(cfg.RNG_SEED)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    torch.cuda.manual_seed_all(cfg.RNG_SEED)

    # get train dataloader (include each category)
    train_dataloaders = get_train_dataloaders(cfg=cfg, mode='train')
    # get test dataloader (include each category)
    if cfg.DATASET.name == "traffic":
        measured_list = ["traffic"]

        test_dataloader_dict = {}
        for each_class in measured_list:
            cfg.DATASET.domain_shift_category = each_class
            test_dataloaders_ = get_test_dataloaders(cfg=cfg, mode='test')
            test_dataloader_dict[each_class] = test_dataloaders_
    else:
        raise NotImplementedError("DATASET {} does not include in target datasets".format(cfg.DATASET.name))

    cur_device = torch.device("cuda:0")

    # training process
    for idx, individual_dataloader in enumerate(train_dataloaders):
        if cfg.TRAIN.method in ['MMR']:
            # target model
            cur_model = load_backbones(cfg.TRAIN.backbone)
            freeze_paras(cur_model)

            # mask model prepare
            mmr_base = MMR_base(cfg=cfg,
                                scale_factors=cfg.TRAIN.MMR.scale_factors,
                                FPN_output_dim=cfg.TRAIN.MMR.FPN_output_dim)

            if cfg.TRAIN.MMR.load_pretrain_model:
                checkpoint = torch.load(cfg.TRAIN.MMR.model_chkpt)
                checkpoint = scratch_MAE_decoder(checkpoint)
                msg = mmr_base.load_state_dict(checkpoint['model'], strict=False)
            else:
                pass
        else:
            raise NotImplementedError("train method {} does not include in target methods".format(cfg.TRAIN.method))

        # optimizer load
        optimizer = None
        if cfg.TRAIN.method in ['MMR']:
            # following timm: set wd as 0 for bias and norm layers (AdamW)
            param_groups = optim_factory.add_weight_decay(mmr_base, cfg.TRAIN_SETUPS.weight_decay)
            optimizer = torch.optim.AdamW(param_groups, lr=cfg.TRAIN_SETUPS.learning_rate, betas=(0.9, 0.95))
        else:
            raise NotImplementedError("train method {} does not include in target methods".format(cfg.TRAIN.method))

        # start training
        torch.cuda.empty_cache()
        if cfg.TRAIN.method == 'MMR':
            MMR_instance = MMR_pipeline_(cur_model=cur_model,
                                         mmr_model=mmr_base,
                                         optimizer=optimizer,
                                         device=cur_device,
                                         cfg=cfg)
            MMR_instance.fit(individual_dataloader, test_dataloader=test_dataloader_dict["traffic"][idx])
        else:
            raise NotImplementedError("train method {} does not include in target methods".format(cfg.TRAIN.method))
