import time

import torch
from progress.bar import Bar
from torch import optim
from torch.utils.data import DataLoader

from dlmcol.InvariantColorNNet import InvariantColorNNet


class AverageMeter:
    """Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class NNetWrapper:
    def __init__(self, size, k, dropout, remix, verbose, nbEpochTraining, layers_size):
        self.size = size
        self.k = k
        self.verbose = verbose
        self.nnet = InvariantColorNNet(size, k, dropout, remix, layers_size)
        self.nbEpochTraining = nbEpochTraining
        self.device = ""

    def set_to_device(self, device):
        self.device = device
        self.nnet.to(device)

    def train(self, examples, batch_size):
        optimizer = optim.Adam(self.nnet.parameters())
        dataloader = DataLoader(examples, batch_size=batch_size, shuffle=True)
        self.nnet.train()
        for _ in range(self.nbEpochTraining):
            data_time = AverageMeter()
            batch_time = AverageMeter()
            end = time.time()
            if self.verbose:
                p_bar = Bar("Training Net", max=int(len(examples) / batch_size) + 1)
            batch_idx = 0
            for sample_batched in dataloader:
                graphs, fit = sample_batched
                graphs = graphs.to(self.device)
                target_fit = fit.to(self.device)
                data_time.update(time.time() - end)
                out_fit = self.nnet(graphs)
                Loss_fit = self.loss_fit(target_fit, out_fit)
                regul = 0
                total_loss = Loss_fit
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                batch_idx += 1
                if self.verbose:
                    # plot progress
                    p_bar.suffix = f"({batch_idx}/{int(len(examples) / batch_size) + 1}) Data: {data_time.avg:.3f}s | Batch: {batch_time.avg:.3f}s | Total: {p_bar.elapsed_td:} | Loss_c: {Loss_fit:.3f} | Regul: {regul:.3f}"
                    p_bar.next()
            if self.verbose:
                p_bar.finish()

            torch.cuda.empty_cache()

    def predict(self, graph, convert=False):
        graph = graph.to(self.device)
        self.nnet.eval()
        with torch.no_grad():
            c = self.nnet(graph)
        if convert:
            return c.data.cpu().numpy()
        return c.data

    def predict_batch(self, all_graphs, batch_size):
        self.nnet.eval()
        dataloader = DataLoader(all_graphs, batch_size=batch_size, shuffle=False)
        data_time = AverageMeter()
        batch_time = AverageMeter()
        end = time.time()
        if self.verbose:
            p_bar = Bar("Training Net", max=int(len(all_graphs) / batch_size) + 1)
        batch_idx = 0
        all_expected_fit = []
        for sample_batched in dataloader:
            graphs = sample_batched
            graphs = graphs.to(self.device)
            data_time.update(time.time() - end)
            out_fit = self.nnet(graphs)
            all_expected_fit.extend(out_fit[:, 0].tolist())
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            batch_idx += 1
            if self.verbose:
                # plot progress
                p_bar.suffix = f"({batch_idx}/{int(len(all_graphs) / batch_size) + 1}) Data: {data_time.avg:.3f}s | Batch: {batch_time.avg:.3f}s"
                p_bar.next()
        return all_expected_fit

    def loss_fit(self, targets, outputs):
        return torch.mean((targets - outputs) ** 2)
