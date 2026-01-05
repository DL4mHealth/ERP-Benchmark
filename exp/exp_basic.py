import os
import torch
from models import (Medformer, Transformer, BIOT, MedGNN, EEGNet, EEGInception, EEGConformer,
                    EEGFeatures, ERPFeatures, TCN, LaBraM, CBraMod, TimesNet,
                    ModernTCN, PatchTST, iTransformer, TestFormer)


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'Medformer': Medformer,
            'Transformer': Transformer,
            'EEGConformer': EEGConformer,
            'BIOT': BIOT,
            'MedGNN': MedGNN,
            'EEGNet': EEGNet,
            'EEGInception': EEGInception,
            'EEGFeatures': EEGFeatures,
            'ERPFeatures': ERPFeatures,
            'TCN': TCN,
            'TimesNet': TimesNet,
            'LaBraM': LaBraM,
            'CBraMod': CBraMod,
            'ModernTCN': ModernTCN,
            'PatchTST': PatchTST,
            'iTransformer': iTransformer,
            'TestFormer': TestFormer,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
