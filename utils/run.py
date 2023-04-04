import torch
from torch import LongTensor, FloatTensor
from torch.nn.utils import clip_grad_norm_
import time
from utils.tools import save2file
import os
from math import inf
from tqdm import tqdm
import torch.distributed as dist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reduce(x, type):
    if type == 'long':
        x = LongTensor([x]).cuda(non_blocking=True)
    else:
        x = FloatTensor([x]).cuda(non_blocking=True)
    return x.item()


class Fit:

    def __init__(self, args, params):

        self.train_data = params['train_data']
        self.model = params['model']
        self.optimizer = params['optimizer']
        self.criterion = params['criterion']
        self.checkpoint = params['checkpoint']
        self.PAD_index = getattr(args, 'PAD_index')
        self.EPOCH = getattr(args, 'epoch')
        self.eval = getattr(args, 'eval', None)
        self.grad_accum = getattr(args, 'grad_accum', 1)
        self.batchPrintInfo = getattr(args, 'batch_print_info', 500)
        self.rank = getattr(args, 'rank', 0)
        self.world_size = getattr(args, 'world_size', 1)

    def train_ce(self, source, target_inputs, target_outputs, norm):
        outputs = self.model(source,
                             mode='train',
                             target=target_inputs,
                             return_memory=False,
                             step='ce')
        return self.criterion(outputs, target_outputs, norm)

    def run(self, data):
        self.model.to(device)
        self.model.train()
        total_loss = 0
        total_cnt = 0
        total_tok = 0
        st_time = time.time()
        grad_accum_cnt = 0
        self.optimizer.zero_grad()
        for i, batch in enumerate(data):

            source_input_ids = batch['source_input_ids'].to(device)
            graph = batch['graph'].to(device)
            target_input_ids = batch['target_input_ids'].to(device)

            ntoken = (source_input_ids != self.PAD_index).sum().item()
            total_tok += ntoken
            total_cnt += 1
            outputs = self.model(mode='train',
                                 source=source_input_ids,
                                 graph=graph,
                                 ground_truth=target_input_ids)

            loss = self.criterion(outputs, target_input_ids)

            total_loss += loss.item()
            loss.backward()

            if i % self.batchPrintInfo == 0:
                if self.world_size > 1:
                    total_loss = reduce(total_loss, type='float')
                    total_tok = reduce(total_tok, type='long')
                    total_cnt = reduce(total_cnt, type='long')
                total_time = time.time() - st_time
                st_time = time.time()
                if self.rank == 0:
                    print('Batch: %d\tloss: %f\tTok pre Sec: %d\t\tTime: %d' %
                          (i, total_loss / total_cnt, total_tok / total_time, total_time))
                total_loss = 0
                total_cnt = 0
                total_tok = 0
            grad_accum_cnt += 1
            if grad_accum_cnt == self.grad_accum:
                grad_accum_cnt = 0
                self.optimizer.step()
                self.optimizer.zero_grad()

        if grad_accum_cnt != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

    def __call__(self, start_epoch):
        torch.backends.cudnn.benchmark = True
        for epoch in range(start_epoch, self.EPOCH):
            if self.rank == 0:
                print('+' * 80)
                print('EPOCH: %d' % (epoch + 1))
                print('-' * 80)
            self.run(self.train_data)
            if self.rank == 0 and self.checkpoint is not None:
                self.checkpoint.save_point(model=self.model,
                                           optim=self.optimizer,
                                           epoch=epoch + 1)
