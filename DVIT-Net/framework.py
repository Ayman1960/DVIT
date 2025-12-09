import torch
import torch.nn as nn
from torch.autograd import Variable as V
import torch.nn.functional as F
import cv2
import numpy as np
from tqdm import tqdm
from loss.dice_loss import dice_loss


class MyFrame:
    def __init__(self, net, lr=2e-4, evalmode=False, batchsize=1):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(
            self.net, device_ids=range(torch.cuda.device_count())
        )
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        self.log_vars = nn.Parameter(torch.zeros((2))).cuda()
        self.old_lr = lr
        self.batchsize = batchsize
        if evalmode:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()

    def forward(self):
            self.img = self.img.cuda()
            if self.mask is not None:
                self.mask = self.mask.cuda()
            
    def optimize(self):
        self.forward()
        self.optimizer.zero_grad()
        pred = self.net.forward(self.img)
        loss = dice_loss(self.mask,pred)
        return loss  # 只返回loss，不进行反向传播

    def set_input(self, img_batch, mask_batch=None, img_id=None):
        self.img = img_batch
        self.mask = mask_batch
        self.img_id = img_id

    def test_one_img(self, img):
        pred = self.net.forward(img)
        thresholds = 0.5
        pred[pred > thresholds] = 1
        pred[pred <= thresholds] = 0
        pred = pred.float()
        pred = pred.squeeze().cpu().data.numpy()
        return pred

    def test_batch(self):
        self.forward(volatile=True)
        mask = self.net.forward(self.img).cpu().data.numpy().squeeze(1)
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        return mask, self.img_id

    def test_one_img_from_path(self, path):
        img = cv2.imread(path)
        img = np.array(img, np.float32) / 255.0 * 3.2 - 1.6
        img = torch.Tensor(img).cuda()

        mask = self.net.forward(img).squeeze().cpu().data.numpy()  # .squeeze(1)
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        return mask

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

    def update_lr(self, new_lr, mylog, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

        # print >> mylog, 'update learning rate: %f -> %f' % (self.old_lr, new_lr)
        print("update learning rate: %f -> %f" % (self.old_lr, new_lr))
        self.old_lr = new_lr
