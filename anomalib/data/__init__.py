"""Anomalib Datasets."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from enum import Enum

from omegaconf import DictConfig, ListConfig

from .avenue import Avenue
from .base import AnomalibDataModule, AnomalibDataset
from .btech import BTech
from .folder import Folder
from .folder_3d import Folder3D
from .inference import InferenceDataset
from .mvtec import MVTec
from .mvtec_3d import MVTec3D
from .shanghaitech import ShanghaiTech
from .task_type import TaskType
from .ucsd_ped import UCSDped
from .visa import Visa
from .traffic_2d_flow import Traffic2DFlow
from .traffic_2d_flowbase import Traffic2DFlowbase

logger = logging.getLogger(__name__)


class DataFormat(str, Enum):
    """Supported Dataset Types"""

    MVTEC = "mvtec"
    MVTEC_3D = "mvtec_3d"
    BTECH = "btech"
    FOLDER = "folder"
    FOLDER_3D = "folder_3d"
    UCSDPED = "ucsdped"
    AVENUE = "avenue"
    VISA = "visa"
    SHANGHAITECH = "shanghaitech"
    Traffic2DFlow = 'traffic_2d_flow'
    Traffic2DFlowbase = 'traffic_2d_flowbase'


def get_datamodule(config: DictConfig | ListConfig) -> AnomalibDataModule:
    """Get Anomaly Datamodule.

    Args:
        config (DictConfig | ListConfig): Configuration of the anomaly model.

    Returns:
        PyTorch Lightning DataModule
    """
    logger.info("Loading the datamodule")

    datamodule: AnomalibDataModule

    # convert center crop to tuple
    center_crop = config.dataset.get("center_crop")
    if center_crop is not None:
        center_crop = (center_crop[0], center_crop[1])

    if config.dataset.format.lower() == DataFormat.MVTEC:
        datamodule = MVTec(
            root=config.dataset.path,
            category=config.dataset.category,
            image_size=(config.dataset.image_size[0], config.dataset.image_size[1]),
            center_crop=center_crop,
            normalization=config.dataset.normalization,
            train_batch_size=config.dataset.train_batch_size,
            eval_batch_size=config.dataset.eval_batch_size,
            num_workers=config.dataset.num_workers,
            task=config.dataset.task,
            transform_config_train=config.dataset.transform_config.train,
            transform_config_eval=config.dataset.transform_config.eval,
            test_split_mode=config.dataset.test_split_mode,
            test_split_ratio=config.dataset.test_split_ratio,
            val_split_mode=config.dataset.val_split_mode,
            val_split_ratio=config.dataset.val_split_ratio,
        )
    elif config.dataset.format.lower() == DataFormat.MVTEC_3D:
        datamodule = MVTec3D(
            root=config.dataset.path,
            category=config.dataset.category,
            image_size=(config.dataset.image_size[0], config.dataset.image_size[1]),
            center_crop=center_crop,
            normalization=config.dataset.normalization,
            train_batch_size=config.dataset.train_batch_size,
            eval_batch_size=config.dataset.eval_batch_size,
            num_workers=config.dataset.num_workers,
            task=config.dataset.task,
            transform_config_train=config.dataset.transform_config.train,
            transform_config_eval=config.dataset.transform_config.eval,
            test_split_mode=config.dataset.test_split_mode,
            test_split_ratio=config.dataset.test_split_ratio,
            val_split_mode=config.dataset.val_split_mode,
            val_split_ratio=config.dataset.val_split_ratio,
        )
    elif config.dataset.format.lower() == DataFormat.BTECH:
        datamodule = BTech(
            root=config.dataset.path,
            category=config.dataset.category,
            image_size=(config.dataset.image_size[0], config.dataset.image_size[1]),
            center_crop=center_crop,
            normalization=config.dataset.normalization,
            train_batch_size=config.dataset.train_batch_size,
            eval_batch_size=config.dataset.eval_batch_size,
            num_workers=config.dataset.num_workers,
            task=config.dataset.task,
            transform_config_train=config.dataset.transform_config.train,
            transform_config_eval=config.dataset.transform_config.eval,
            test_split_mode=config.dataset.test_split_mode,
            test_split_ratio=config.dataset.test_split_ratio,
            val_split_mode=config.dataset.val_split_mode,
            val_split_ratio=config.dataset.val_split_ratio,
        )
    elif config.dataset.format.lower() == DataFormat.FOLDER:
        datamodule = Folder(
            root=config.dataset.root,
            normal_dir=config.dataset.normal_dir,
            abnormal_dir=config.dataset.abnormal_dir,
            task=config.dataset.task,
            normal_test_dir=config.dataset.normal_test_dir,
            mask_dir=config.dataset.mask_dir,
            extensions=config.dataset.extensions,
            image_size=(config.dataset.image_size[0], config.dataset.image_size[1]),
            center_crop=center_crop,
            normalization=config.dataset.normalization,
            train_batch_size=config.dataset.train_batch_size,
            eval_batch_size=config.dataset.eval_batch_size,
            num_workers=config.dataset.num_workers,
            transform_config_train=config.dataset.transform_config.train,
            transform_config_eval=config.dataset.transform_config.eval,
            test_split_mode=config.dataset.test_split_mode,
            test_split_ratio=config.dataset.test_split_ratio,
            val_split_mode=config.dataset.val_split_mode,
            val_split_ratio=config.dataset.val_split_ratio,
        )
    elif config.dataset.format.lower() == DataFormat.FOLDER_3D:
        datamodule = Folder3D(
            root=config.dataset.root,
            normal_dir=config.dataset.normal_dir,
            normal_depth_dir=config.dataset.normal_depth_dir,
            abnormal_dir=config.dataset.abnormal_dir,
            abnormal_depth_dir=config.dataset.abnormal_depth_dir,
            task=config.dataset.task,
            normal_test_dir=config.dataset.normal_test_dir,
            normal_test_depth_dir=config.dataset.normal_test_depth_dir,
            mask_dir=config.dataset.mask_dir,
            extensions=config.dataset.extensions,
            image_size=(config.dataset.image_size[0], config.dataset.image_size[1]),
            center_crop=center_crop,
            normalization=config.dataset.normalization,
            train_batch_size=config.dataset.train_batch_size,
            eval_batch_size=config.dataset.eval_batch_size,
            num_workers=config.dataset.num_workers,
            transform_config_train=config.dataset.transform_config.train,
            transform_config_eval=config.dataset.transform_config.eval,
            test_split_mode=config.dataset.test_split_mode,
            test_split_ratio=config.dataset.test_split_ratio,
            val_split_mode=config.dataset.val_split_mode,
            val_split_ratio=config.dataset.val_split_ratio,
        )
    elif config.dataset.format.lower() == DataFormat.UCSDPED:
        datamodule = UCSDped(
            root=config.dataset.path,
            category=config.dataset.category,
            task=config.dataset.task,
            clip_length_in_frames=config.dataset.clip_length_in_frames,
            frames_between_clips=config.dataset.frames_between_clips,
            target_frame=config.dataset.target_frame,
            image_size=(config.dataset.image_size[0], config.dataset.image_size[1]),
            center_crop=center_crop,
            normalization=config.dataset.normalization,
            transform_config_train=config.dataset.transform_config.train,
            transform_config_eval=config.dataset.transform_config.eval,
            train_batch_size=config.dataset.train_batch_size,
            eval_batch_size=config.dataset.eval_batch_size,
            num_workers=config.dataset.num_workers,
            val_split_mode=config.dataset.val_split_mode,
            val_split_ratio=config.dataset.val_split_ratio,
        )
    elif config.dataset.format.lower() == DataFormat.AVENUE:
        datamodule = Avenue(
            root=config.dataset.path,
            gt_dir=config.dataset.gt_dir,
            task=config.dataset.task,
            clip_length_in_frames=config.dataset.clip_length_in_frames,
            frames_between_clips=config.dataset.frames_between_clips,
            target_frame=config.dataset.target_frame,
            image_size=(config.dataset.image_size[0], config.dataset.image_size[1]),
            center_crop=center_crop,
            normalization=config.dataset.normalization,
            transform_config_train=config.dataset.transform_config.train,
            transform_config_eval=config.dataset.transform_config.eval,
            train_batch_size=config.dataset.train_batch_size,
            eval_batch_size=config.dataset.eval_batch_size,
            num_workers=config.dataset.num_workers,
            val_split_mode=config.dataset.val_split_mode,
            val_split_ratio=config.dataset.val_split_ratio,
        )
    elif config.dataset.format.lower() == DataFormat.VISA:
        datamodule = Visa(
            root=config.dataset.path,
            category=config.dataset.category,
            image_size=(config.dataset.image_size[0], config.dataset.image_size[1]),
            center_crop=center_crop,
            normalization=config.dataset.normalization,
            train_batch_size=config.dataset.train_batch_size,
            eval_batch_size=config.dataset.eval_batch_size,
            num_workers=config.dataset.num_workers,
            task=config.dataset.task,
            transform_config_train=config.dataset.transform_config.train,
            transform_config_eval=config.dataset.transform_config.eval,
            test_split_mode=config.dataset.test_split_mode,
            test_split_ratio=config.dataset.test_split_ratio,
            val_split_mode=config.dataset.val_split_mode,
            val_split_ratio=config.dataset.val_split_ratio,
        )
    elif config.dataset.format.lower() == DataFormat.SHANGHAITECH:
        datamodule = ShanghaiTech(
            root=config.dataset.path,
            scene=config.dataset.scene,
            task=config.dataset.task,
            clip_length_in_frames=config.dataset.clip_length_in_frames,
            frames_between_clips=config.dataset.frames_between_clips,
            target_frame=config.dataset.target_frame,
            image_size=(config.dataset.image_size[0], config.dataset.image_size[1]),
            center_crop=center_crop,
            normalization=config.dataset.normalization,
            transform_config_train=config.dataset.transform_config.train,
            transform_config_eval=config.dataset.transform_config.eval,
            train_batch_size=config.dataset.train_batch_size,
            eval_batch_size=config.dataset.eval_batch_size,
            num_workers=config.dataset.num_workers,
            val_split_mode=config.dataset.val_split_mode,
            val_split_ratio=config.dataset.val_split_ratio,
        )
    elif config.dataset.format.lower() == DataFormat.Traffic2DFlow:
        datamodule = Traffic2DFlow(
        dataset_root=config.dataset.dataset_root,
        train_data_dir=config.dataset.train_data_dir,
        valid_data_dir=config.dataset.valid_data_dir,
        test_data_dir=config.dataset.test_data_dir,
        train_label_dir=config.dataset.train_label_dir,
        valid_label_dir=config.dataset.valid_label_dir,
        test_label_dir=config.dataset.test_label_dir,
        create_validation_set=config.dataset.create_validation_set,
        train_batch_size=config.dataset.train_batch_size,
        test_batch_size=config.dataset.test_batch_size,
        flow_num=config.dataset.flow_num,
        num_workers=config.dataset.num_workers,
        )
    elif config.dataset.format.lower() == DataFormat.Traffic2DFlowbase:
        datamodule = Traffic2DFlowbase(
        dataset_root=config.dataset.dataset_root,
        train_data_dir=config.dataset.train_data_dir,
        valid_data_dir=config.dataset.valid_data_dir,
        test_data_dir=config.dataset.test_data_dir,
        train_label_dir=config.dataset.train_label_dir,
        valid_label_dir=config.dataset.valid_label_dir,
        test_label_dir=config.dataset.test_label_dir,
        create_validation_set=config.dataset.create_validation_set,
        train_batch_size=config.dataset.train_batch_size,
        test_batch_size=config.dataset.test_batch_size,
        flow_num=config.dataset.flow_num,
        num_workers=config.dataset.num_workers,
        )
    else:
        raise ValueError(
            "Unknown dataset! \n"
            "If you use a custom dataset make sure you initialize it in"
            "`get_datamodule` in `anomalib.data.__init__.py"
        )

    return datamodule


__all__ = [
    "AnomalibDataset",
    "AnomalibDataModule",
    "get_datamodule",
    "BTech",
    "Folder",
    "Folder3D",
    "InferenceDataset",
    "MVTec",
    "MVTec3D",
    "Avenue",
    "UCSDped",
    "TaskType",
    "ShanghaiTech",
    "Traffic2DFlow",
    "Traffic2DFlowbase",
]
