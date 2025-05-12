import os
import shutil
import time

import numpy as np

from app.utils import BaseModel
from model_REWA.src import utils
from model_REWA.src.utils import RARADataset, pad_collate_func_rara, RandomChunkSampler

from torch.utils.data import DataLoader
from model_REWA.src.RARA import RARAMal

import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint

import pytorch_lightning as pl

import torch
import torch.optim as optim
import torch.nn as nn

from model_REWA.src.perturbation_structure import PerturbationStructure
from model_REWA.src.SincereLoss_old import MultiviewSINCERELoss
from model_REWA.src.generate_adv_sample import find_custom_section_payload_offset
from model_REWA.src.custom_section import add_new_custom_section
from model_REWA.src.tools import generate_wasm_file


class RARAClassifier(pl.LightningModule, BaseModel):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.raraMal = RARAMal(
            self.args
        )

        # Prior-Guided Adversarial Initialization
        self.perturbation_structure = PerturbationStructure()
        self.sincere_loss = MultiviewSINCERELoss(temperature=self.args.temperature)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="sum")
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def forward(self, x):
        return self.raraMal(x)

    def attack(self, x, y, file_path=[]):
        return

    def reconstruction(self, x0, x_adv_embedding, x_adv_perturb_index, E, file_path):
        return

    def training_step(self, batch, batch_idx):
        return

    def validation_step(self, batch, batch_idx):
        x, label, _, _ = batch
        x = x.long()
        # label = label.long().squeeze(dim=1)
        label = label.long()
        # 在 torch.no_grad() 上下文管理器中进行计算，禁用梯度计算
        with torch.no_grad():
            pred, _, _ = self.raraMal(x)
        # loss = self.criterion(pred, label)

        pred = torch.argmax(pred, dim=1)
        self.accuracy(pred, label)
        self.f1_score(pred, label)
        self.precesion(pred, label)
        self.recall(pred, label)

    def training_step_end(self, batch_parts):
        torch.cuda.empty_cache()
        shutil.rmtree(self.args.temp_data_path)
        if not os.path.exists(self.args.temp_data_path):
            os.mkdir(self.args.temp_data_path)

    # 将字典中的 Tensor 对象转换为列表
    def convert_to_serializable(self, values):
        if isinstance(values, torch.Tensor):
            return values.tolist()
        return values

    def training_epoch_end(self, training_step_outputs):
        self.args.now_epochs += 1
        torch.cuda.empty_cache()
        print('\n now_epochs value is : ', self.args.now_epochs)
        self.perturbation_structure.getSize()
        # file_path = self.args.checkpoint_path + 'perturbation.json'
        # serializable_dict = json.loads(json.dumps(self.perturbation_structure.data, default=self.convert_to_serializable))
        # with open(file_path, 'w') as file:
        #     json.dump(serializable_dict, file)
        # 读取文件内容
        # with open(file_path, 'r') as file:
        #     data = json.load(file)

    def validation_epoch_end(self, outputs):
        acc = self.accuracy.compute()
        f1 = self.f1_score.compute()
        precision = self.precesion.compute()
        recall = self.recall.compute()

        self.log("val_acc", acc, prog_bar=True)
        self.log("performance", {"val_acc": acc, "val_precesion": precision, "val_recall": recall, "f1": f1},
                 prog_bar=True)

        self.accuracy.reset()
        self.f1_score.reset()
        self.recall.reset()
        self.precesion.reset()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.args.learning_rate)
        return optimizer
        # optimizer = optim.Adam(self.parameters(), lr=self.args.learning_rate)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=0)
        # return {"optimizer": optimizer, "lr_scheduler": scheduler},

    def train_dataloader(self) -> DataLoader:
        # 训练集，不需要添加头部，指定0列(md5)为索引
        tr_label_table = pd.read_csv(self.args.train_label_path, header=None, index_col=0)
        tr_label_table.index = tr_label_table.index.str.lower()
        tr_label_table = tr_label_table.rename(columns={1: 'ground_truth'})

        print('Train dataset:')
        print('\tTotal', len(tr_label_table), 'files')
        print('\tTraining Count :\n', tr_label_table['ground_truth'].value_counts())

        train_dataset = RARADataset(
            list(tr_label_table.index),
            self.args.train_data_path,
            list(tr_label_table.ground_truth),
            self.args.train_pert_path,
            sort_by_size=True
        )

        # concat_dataset = ConcatDataset([train_dataset1, train_dataset2])

        trainloader = DataLoader(
            dataset=train_dataset,
            batch_size=self.args.batch_size,
            sampler=RandomChunkSampler(train_dataset, self.args.batch_size),
            num_workers=self.args.cpu_num,
            drop_last=False,
            pin_memory=True,
            collate_fn=pad_collate_func_rara
        )

        return trainloader

    def val_dataloader(self):
        # 测试集，不需要添加头部，指定0列(md5)为索引
        val_label_table = pd.read_csv(self.args.valid_label_path, header=None, index_col=0)
        val_label_table.index = val_label_table.index.str.lower()
        val_label_table = val_label_table.rename(columns={1: 'ground_truth'})

        print('Test dataset:')
        print('\tTotal', len(val_label_table), 'files')
        print('\tTest Count :\n', val_label_table['ground_truth'].value_counts())

        validate_dataset = RARADataset(
            list(val_label_table.index),
            self.args.valid_data_path,
            list(val_label_table.ground_truth),
            self.args.test_pert_path,
            True
        )

        validloader = DataLoader(
            dataset=validate_dataset,
            batch_size=self.args.validate_batch_size,
            sampler=RandomChunkSampler(validate_dataset, self.args.validate_batch_size),
            num_workers=self.args.cpu_num,
            drop_last=False,
            pin_memory=True,
            collate_fn=pad_collate_func_rara
        )

        return validloader

    def single_file_process(self, file):
        # 将字节序列转换为整数列表
        x = [i + 1 for i in file]
        # 将数据转换为torch tensor并添加batch维度
        return torch.tensor(x).unsqueeze(0)

    def predict(self, x):
        pred, _, _ = self.raraMal.encoder(x)
        soft = nn.Softmax(dim=1)
        res = soft(pred)
        return res

    def load_model(self, model_path):
        self.load_from_checkpoint(
            checkpoint_path=model_path,
            args=self.args
        )
        self.eval()


def main():
    # 设置 PYTORCH_CUDA_ALLOC_CONF 环境变量
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb=128"

    args = utils.load_config("config.yaml")
    # 设置logger
    log_path = os.path.join(args.log_path)
    logger = utils.logger(log_path)

    pl.seed_everything(args.seed)

    # 设置全局的随机种子，保证实验结果可复现
    # utils.setup_seed(args.seed)

    model = RARAClassifier(args)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath=args.checkpoint_path + "/classifier",
        filename="{epoch}_model_{val_acc:.5f}",
        save_top_k=-1,
        mode="max"
    )

    if args.reload_GCT:
        trainer = pl.Trainer(
            max_epochs=args.max_epoches,
            gpus=1,
            callbacks=[checkpoint_callback],
            check_val_every_n_epoch=1,
            # precision="16",
            deterministic=False,
            log_every_n_steps=1,
            resume_from_checkpoint="/".join([args.checkpoint_path, "classifier",
                                             f"epoch={args.train_malconv_start_epoch}_model_val_acc=0.99909.ckpt"]),
            amp_backend="apex",
            amp_level="O1",
            # tpu_cores=8,
            # profiler="simple"
        )
    else:
        trainer = pl.Trainer(
            max_epochs=args.max_epoches,
            gpus=1,
            callbacks=[checkpoint_callback],
            check_val_every_n_epoch=1,
            # precision="16",
            deterministic=False,
            log_every_n_steps=1,
            amp_backend="apex",
            amp_level="O1",
        )

    trainer.fit(model)
    # trainer.test()


if __name__ == "__main__":
    # main()
    # from model_REWA.src.RARA_pytorch_lightning import RARAClassifier as Wasm_M
    # from model_REWA.src.utils import load_config
    # model = Wasm_M(
    #     args=load_config("./model_REWA/src/config.yaml")
    # )
    # model.load_model("./model_REWA/ckpt/classifier/epoch=15_model_val_acc=0.99867.ckpt")
    # processed_file = model.single_file_process(file)
    # result = model.predict(processed_file)[0].tolist()[1]