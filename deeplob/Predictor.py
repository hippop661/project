import os
import torch
import numpy as np
import pandas as pd
from typing import List
from .model import DeepLOBMultiTask

class Predictor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # input_features=31 与 train4 的 CONFIG['feature_cols'] 数量一致
        self.model = DeepLOBMultiTask(input_features=31, num_classes=3, num_tasks=5)
        self._model_loaded = False
        self._feature_medians = None

    def _load_model_if_needed(self, pth_path):
        if not self._model_loaded and os.path.exists(pth_path):
            ckpt = torch.load(pth_path, map_location=self.device)
            # 从 checkpoint 中加载特征中位数（与 model 一起打包）
            if isinstance(ckpt, dict) and 'feature_medians' in ckpt:
                self._feature_medians = ckpt['feature_medians']
            # train4 保存格式为 {'model': state_dict, ...}
            if isinstance(ckpt, dict) and 'model' in ckpt:
                state_dict = ckpt['model']
            else:
                state_dict = ckpt
            # 兼容 DDP 保存的 "module." 前缀
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[k.replace("module.", "")] = v
            self.model.load_state_dict(new_state_dict, strict=True)
            self.model.to(self.device)
            self.model.eval()
            self._model_loaded = True

    def predict(self, x: List[pd.DataFrame]) -> List[List[int]]:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        pth_path = os.path.join(current_dir, "model.pth")
        self._load_model_if_needed(pth_path)

        with torch.no_grad():
            processed = [self.preprocess(df) for df in x]
            arr = np.stack([p.values for p in processed])
            x_tensor = torch.tensor(arr).float().unsqueeze(1).to(self.device)
            outputs = self.model(x_tensor)

            all_preds = []
            for i in range(len(x)):
                sample = []
                for task_out in outputs:
                    pred = task_out[i].argmax().item()
                    sample.append(pred)
                all_preds.append(sample)
            return all_preds

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # 训练时用的列名：无 n_ 前缀
        bid_cols = ['bid1', 'bid2', 'bid3', 'bid4', 'bid5']
        ask_cols = ['ask1', 'ask2', 'ask3', 'ask4', 'ask5']
        bsize_cols = ['bsize1', 'bsize2', 'bsize3', 'bsize4', 'bsize5']
        asize_cols = ['asize1', 'asize2', 'asize3', 'asize4', 'asize5']

        df = df.copy()

        bid = df[bid_cols].values
        ask = df[ask_cols].values
        bsize = df[bsize_cols].values
        asize = df[asize_cols].values

        epsilon = 1e-8
        spread = ask - bid
        mid_price = (ask + bid) / 2
        weighted_ab = (ask * bsize + bid * asize) / (bsize + asize + epsilon)

        vol1_sum = bsize[:, 0] + asize[:, 0] + epsilon
        vol1_rel_diff = (bsize[:, 0] - asize[:, 0]) / vol1_sum

        volall_sum = bsize.sum(axis=1) + asize.sum(axis=1) + epsilon
        volall_rel_diff = (bsize.sum(axis=1) - asize.sum(axis=1)) / volall_sum

        new_features = {
            'spread1': spread[:, 0], 'spread2': spread[:, 1], 'spread3': spread[:, 2],
            'mid_price1': mid_price[:, 0], 'mid_price2': mid_price[:, 1], 'mid_price3': mid_price[:, 2],
            'weighted_ab1': weighted_ab[:, 0], 'weighted_ab2': weighted_ab[:, 1], 'weighted_ab3': weighted_ab[:, 2],
            'vol1_rel_diff': vol1_rel_diff, 'volall_rel_diff': volall_rel_diff,
        }

        for col, val in new_features.items():
            df[col] = val

        # 与 train4 完全一致的 31 列特征（不含 amount）
        use_cols = [
            'bid1', 'bsize1', 'bid2', 'bsize2', 'bid3', 'bsize3',
            'bid4', 'bsize4', 'bid5', 'bsize5',
            'ask1', 'asize1', 'ask2', 'asize2', 'ask3', 'asize3',
            'ask4', 'asize4', 'ask5', 'asize5',
            'spread1', 'mid_price1', 'spread2', 'mid_price2',
            'spread3', 'mid_price3',
            'weighted_ab1', 'weighted_ab2', 'weighted_ab3',
            'vol1_rel_diff', 'volall_rel_diff'
        ]

        # 与 train4 一致的数据清洗
        feat_df = df[use_cols]
        feat_df = feat_df.replace([np.inf, -np.inf], np.nan)
        # 使用训练时保存的全局 median，而非当前窗口的 median
        if self._feature_medians:
            feat_df = feat_df.fillna(self._feature_medians)
        else:
            feat_df = feat_df.fillna(feat_df.median())
        feat_df = feat_df.clip(-10, 10)
        df[use_cols] = feat_df

        return df[use_cols]
