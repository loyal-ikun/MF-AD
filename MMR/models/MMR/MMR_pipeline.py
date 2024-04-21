import csv
import os
import time

from .utils import ForwardHook, cal_anomaly_map, each_patch_loss_function, mmr_adjust_learning_rate

from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from thop import profile

import torch
import logging
import numpy as np

LOGGER = logging.getLogger(__name__)


class MMR_pipeline_:

    def __init__(self,
                 cur_model,
                 mmr_model,
                 optimizer,
                 device,
                 cfg):
        self.teacher_outputs_dict = {}
        for extract_layer in cfg.TRAIN.MMR.layers_to_extract_from:
            forward_hook = ForwardHook(self.teacher_outputs_dict, extract_layer)
            network_layer = cur_model.__dict__["_modules"][extract_layer]

            network_layer[-1].register_forward_hook(forward_hook)

        self.cur_model = cur_model.to(device)
        self.mmr_model = mmr_model.to(device)

        self.optimizer = optimizer
        self.device = device
        self.cfg = cfg

    def fit(self, individual_dataloader, test_dataloader):
        temporal_lr = self.cfg.TRAIN_SETUPS.learning_rate
        for epoch in range(self.cfg.TRAIN_SETUPS.epochs):
            start_time = time.time()
            self.cur_model.eval()
            self.mmr_model.train()
            current_lr = mmr_adjust_learning_rate(self.optimizer, epoch, self.cfg)

            loss_list = []
            for image in individual_dataloader:
                if isinstance(image, dict):
                    image = image["image"].to(self.device)
                else:

                    image = torch.from_numpy(image)

                    image = image.to(self.device)

                self.teacher_outputs_dict.clear()
                with torch.no_grad():
                    _ = self.cur_model(image)
                multi_scale_features = [self.teacher_outputs_dict[key]
                                        for key in self.cfg.TRAIN.MMR.layers_to_extract_from]
                reverse_features = self.mmr_model(image,
                                                  mask_ratio=self.cfg.TRAIN.MMR.finetune_mask_ratio)
                multi_scale_reverse_features = [reverse_features[key]
                                                for key in self.cfg.TRAIN.MMR.layers_to_extract_from]

                loss = each_patch_loss_function(multi_scale_features, multi_scale_reverse_features)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_list.append(loss.item())
            auc, f1, acc,  precision, recall = self.evaluation(test_dataloader = test_dataloader, epoch = epoch + 1)
            used_time = time.time() - start_time
            LOGGER.info('epoch [{}/{}], loss:{:.4f}, auc:{:.4f}, acc:{:.4f}, f1:{:.4f}, precision:{:.4f}, recall:{:.4f}, used_time:{:.2f}s, current_lr: {:f}'.format(epoch + 1,
                                                                                                             self.cfg.TRAIN_SETUPS.epochs,
                                                                                                             np.mean(
                                                                                                                 loss_list),
                                                                                                             auc,
                                                                                                            acc,
                                                                                                            f1,
                                                                                                            precision,
                                                                                                            recall,
                                                                                                             used_time,
                                                                                                             current_lr))

        self.cfg.TRAIN_SETUPS.learning_rate = temporal_lr

    def evaluation(self, test_dataloader, epoch):
        self.cur_model.eval()
        self.mmr_model.eval()

        with (torch.no_grad()):
            labels_prediction_2d = []

            for i in range(8):
                labels_gt = []
                labels_prediction = []

                index = 0
                for image in test_dataloader:
                    index += 1
                    if isinstance(image, dict):
                        image["image"] = image["image"][:, i, :, :, :]

                        label_current = image["is_anomaly"].numpy()
                        labels_gt.extend(label_current.tolist())
                        image = image["image"].to(self.device)
                    else:
                        raise Exception("the format of DATA error!")
                    self.teacher_outputs_dict.clear()
                    with torch.no_grad():
                        _ = self.cur_model(image)
                    multi_scale_features = [self.teacher_outputs_dict[key]
                                            for key in self.cfg.TRAIN.MMR.layers_to_extract_from]

                    reverse_features = self.mmr_model(image,
                                                      mask_ratio=self.cfg.TRAIN.MMR.test_mask_ratio)
                    multi_scale_reverse_features = [reverse_features[key]
                                                    for key in self.cfg.TRAIN.MMR.layers_to_extract_from]

                    anomaly_map, _ = cal_anomaly_map(multi_scale_features, multi_scale_reverse_features,
                                                     image.shape[-1],
                                                     amap_mode='a')
                    for item in range(len(anomaly_map)):
                        anomaly_map[item] = gaussian_filter(anomaly_map[item], sigma=4)
                    labels_prediction.extend(np.max(anomaly_map.reshape(anomaly_map.shape[0], -1), axis=1))

                    labels_prediction_2d.append(labels_prediction)
            anomaly_score = np.array(labels_prediction_2d, dtype=np.float64)
            anomaly_score = np.mean(anomaly_score, axis=0)

            if np.isnan(anomaly_score).any() or np.isinf(anomaly_score).any():
                anomaly_score = np.nan_to_num(anomaly_score)
            auroc_samples = round(roc_auc_score(labels_gt, anomaly_score), 4)

            precs, recs, thrs = metrics.precision_recall_curve(labels_gt, anomaly_score)
            f1s = 2 * precs * recs / (precs + recs)
            f1s = f1s[:-1]
            thrs = thrs[~np.isnan(f1s)]
            f1s = f1s[~np.isnan(f1s)]
            best_thre = thrs[np.argmax(f1s)]
            best_predictions = [1 if i > best_thre else 0 for i in anomaly_score]

            log_path = f'/media/dm/新加卷1/MMR/log_traffic_54/2/epoch_{epoch}'
            if os.path.exists(log_path) == False:
                os.makedirs(log_path)
            np.savetxt(os.path.join(log_path, 'output_predict.csv'), np.array(best_predictions)[np.newaxis, :].astype(np.int16), delimiter=',')

            sample1 = torch.randn(1, 3, 64, 64).cuda()

            macs, params = profile(self.cur_model, inputs=(sample1,))   # mmr_model and cur_model
            macs = macs * 8
            print("==> MMR FLOPs: ================> %f G   Params: ================> %f M \n" % (macs / (1000 ** 3), params / (1000 ** 2)))

            return auroc_samples, metrics.f1_score(labels_gt, best_predictions), metrics.accuracy_score(labels_gt, best_predictions), metrics.precision_score(labels_gt, best_predictions), metrics.recall_score(labels_gt, best_predictions)
