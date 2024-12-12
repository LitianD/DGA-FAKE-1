from typing import Any

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
import numpy as np
from src.models.components.utility import calculate_hit
class DiffusionModule(LightningModule):
    def __init__(
        self,
        num_classes:2,
        net: torch.nn.Module,
        # classifier:torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        # bert_weighted_path:str,
        # tokenizer:str,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        #self.classifier = classifier
        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.hit_20 = 0.0
        self.ng_20 = 0.0
        
        self.test_hit_20 = 0.0
        self.test_ng_20 = 0.0
        
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, seq,len_seq,target,text):
        #y = self.net(text,image_path)
        y = self.net.p_losses(seq,len_seq,target,text)
        #y = self.classifier(x1)
        return y
    
    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(self, batch: Any):
        seq,len_seq,target,id,text = batch
        loss, predicted_x = self.forward(seq,len_seq,target,text)
        # loss = self.criterion(logits, y)
        # preds = torch.argmax(logits, dim=1)
        # preds = torch.eye(2,dtype=torch.float).to("cuda")[preds]
        return loss, predicted_x

    def training_step(self, batch: Any, batch_idx: int):
        loss, predicted_x = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        # self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        hit_purchase=[0,0,0,0]
        ndcg_purchase=[0,0,0,0]
        seq,len_seq,target,id,text = batch
        x, scores = self.net.sample(seq,len_seq,target,text)
        # 返回值的index
        _, topK = scores.topk(100, dim=1, largest=True, sorted=True)
        topK = topK.cpu().detach().numpy()
        sorted_list2=np.flip(topK,axis=1)
        sorted_list2 = sorted_list2
        topk=[10, 20, 50]
        calculate_hit(sorted_list2,topk,target.cpu().detach().numpy(),hit_purchase,ndcg_purchase)
        hit_20 = hit_purchase[2]/len_seq.shape[0]
        ng_20=ndcg_purchase[2]/len_seq.shape[0]
        if type(ng_20) is np.ndarray:
            ng_20 = ng_20[0,0]
        self.hit_20 =(self.hit_20 + hit_20)/2
        self.ng_20 =(self.ng_20 + ng_20)/2 

        # update and log metrics
        # self.val_loss(loss)
        # self.val_acc(preds, targets)
        # self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss", self.hit_20, on_step=False, on_epoch=True, prog_bar=True)

        self.log("val/hit_20", self.hit_20, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/ng_20", self.ng_20, on_step=False, on_epoch=True, prog_bar=True)
        
    def on_validation_epoch_end(self):
        print('epoch:',self.current_epoch,"hit_20",self.hit_20,"ndcg_20",self.ng_20)
        self.hit_20 = 0.0
        self.ng_20 = 0.0
        pass
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        hit_purchase=[0,0,0,0]
        ndcg_purchase=[0,0,0,0]
        seq,len_seq,target,id,text = batch
        x, scores = self.net.sample(seq,len_seq,target,text)
        # 返回值的index
        _, topK = scores.topk(100, dim=1, largest=True, sorted=True)
        topK = topK.cpu().detach().numpy()
        sorted_list2=np.flip(topK,axis=1)
        sorted_list2 = sorted_list2
        topk=[10, 20, 50]
        calculate_hit(sorted_list2,topk,target.cpu().detach().numpy(),hit_purchase,ndcg_purchase)
        hit_20 = hit_purchase[2]/len_seq.shape[0]
        ng_20=ndcg_purchase[2]/len_seq.shape[0]
        if type(ng_20) is np.ndarray:
            ng_20 = ng_20[0,0]
        self.test_hit_20 =(self.test_hit_20 + hit_20)/2
        self.test_ng_20 =(self.test_ng_20 + ng_20)/2 

        # update and log metrics
        # self.val_loss(loss)
        # self.val_acc(preds, targets)
        # self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/hit_20", self.test_hit_20, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/ng_20", self.test_ng_20, on_step=False, on_epoch=True, prog_bar=True)
        print("hit_20",self.hit_20,"ndcg_20",self.ng_20)
    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = ClipModule(None, None, None)