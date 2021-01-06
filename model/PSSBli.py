import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import logging
import os
import json
from collections import OrderedDict


from IO import WordDictionary, MonoDictionary, Language,\
    CrossLingualDictionary, Batcher
from sinkhorn import Prior_sinkhorn
from evaluation import CSLS, Evaluator
from model import bliMethod, LinearTrans
from utils import to_cuda

class PSSBli(bliMethod):
    def __init__(self, src, tgt, cuda, seed, batcher, data_dir, save_dir):   
        """
        inputs:
            :param src (str): name of source lang
            :param tgt (str): name of target lang
            :param cuda (int): number of gpu to be used
            :param seed (int): seed for torch and numpy
            :param batcher (Batcher): The batcher object
            :param data_dir (str): Loaction of the data
            :param save_dir (str): Location to save models
        """
        super(PSSBli, self).__init__(src, tgt, cuda, seed, batcher, data_dir, save_dir)
        embed_dim = self.batcher.name2lang[self.src].embeddings.shape[1]
        self.transform1 = LinearTrans(embed_dim).double().to(self.device)
        self.Q1 = None
        self.transform2 = LinearTrans(embed_dim).double().to(self.device)
        self.Q2 = None

        self.rcslsQ = None

    def P_solver(self, embi, embj, T, epsilon):
        Mt = -torch.mm(embi.mm(self.Q2), embj.t())
        ones1 = torch.ones(Mt.shape[0], device = self.device) / Mt.shape[0]
        ones2 = torch.ones(Mt.shape[1], device = self.device) / Mt.shape[1]
        P = Prior_sinkhorn(ones1, ones2, Mt, T, 0, epsilon, stopThr=1e-3)
        return P

    def orthogonal_mapping_update(self, GQ, learning_rate):
        next_Q = (self.Q2 - learning_rate * GQ).cpu().numpy()
        U, S, VT = np.linalg.svd(next_Q)
        self.Q2 = torch.from_numpy((U.dot(VT))).to(self.device)
        self.transform2.setWeight(self.Q2)
    
    def supervised_rcsls_loss(self, src, tgt, nn_src, nn_tgt, k=10):
        # first an assert to ensure unit norming
        if not hasattr(self, "check_rcsls_valid"):
            self.check_rcsls_valid = True
            for l in self.batcher.name2lang.values():
                if l.unit_norm is False:
                    self.check_rcsls_valid = False
                    break
        if not self.check_rcsls_valid:
            raise RuntimeError("For RCSLS, need to unit norm")

        xtrans = self.transform1(Variable(src))
        yvar = Variable(tgt)
        sup_loss = 2 * torch.sum(xtrans * yvar)
        # Compute nearest nn loss wrt src
        nn_tgt = Variable(nn_tgt)
        dmat = torch.mm(xtrans, nn_tgt.t())
        _, tix = torch.topk(dmat, k, dim=1)
        nnbrs = nn_tgt[tix.view(-1)].view((tix.shape[0], tix.shape[1], -1))
        nnbrs = Variable(nnbrs.data)
        nnloss = torch.bmm(nnbrs, xtrans.unsqueeze(-1)).squeeze(-1)
        nn_tgt_loss = torch.sum(nnloss) / k
        # Compute nearest nn loss wrt tgt
        nn_src = Variable(nn_src)
        nn_src_transform1 = Variable(self.transform1(nn_src).data)
        dmat = torch.mm(yvar, nn_src_transform1.t())
        _, tix = torch.topk(dmat, k, dim=1)
        nnbrs = nn_src[tix.view(-1)].view((tix.shape[0], tix.shape[1], -1))
        nnbrs = Variable(nnbrs.data)
        nnloss = torch.bmm(self.transform1(nnbrs), yvar.unsqueeze(-1)).squeeze(-1)
        nn_src_loss = torch.sum(nnloss) / k
        return - (sup_loss - nn_tgt_loss - nn_src_loss) / src.size(0)

    def procrustes_onestep(self, src_aligned_embeddings, tgt_aligned_embeddings):
        matrix = torch.mm(tgt_aligned_embeddings.transpose(1, 0), src_aligned_embeddings)
        u, _, v = torch.svd(matrix)
        weight = torch.mm(u, v.t())
        return weight

    def computePrior(self, X, Y, Q, t):
        M = torch.mm(X.mm(Q), Y.t())
        M = - M + M.topk(10, 1)[0].sum(1).reshape(M.shape[0], 1) / 10 + M.topk(10, 0)[0].sum(0).reshape(1, M.shape[1]) / 10
        Mmin, _ = M.min(axis = 1, keepdim=True)
        T = torch.zeros_like(M, device=self.device) 
        torch.exp(-M / t, out=T)
        T = T / torch.sum(T, axis = 1, keepdim=True)
        return T

    def unsupervised_loss(self, transform, mode="csls"):
        csls = self.get_csls(transform)
        max_src_word_considered = 10000
        _, unsup_loss = self.evaluator.get_match_samples(
            csls, np.arange(
                min(self.evaluator.src_lang.vocab, int(max_src_word_considered))),
            1, mode=mode)
        return unsup_loss

    def selectBest(self, mode = "csls"):
        logger = logging.getLogger(__name__)
        logger.info("----Choose the best between Q_sup and Q_unsup----")
        metric1 = self.unsupervised_loss(self.transform1)
        logger.info("metric of trans. learned by Sup: {0:.2f}".format(metric1))

        metric2 = self.unsupervised_loss(self.transform2)
        logger.info("metric of trans. learned by UnSup: {0:.2f}".format(metric2))

        if metric1 > metric2:
            logger.info("Choose trans learned by (Sup.) as final output")
            self.transform = self.transform1
        else:
            logger.info("Choose trans learned by (UnSup.) as final output")
            self.transform = self.transform2



    def train(
        self, epochs, unsup_lr, unsup_epsilon, unsup_bsz, unsup_steps,
        unsup_t, sup_steps, expand_dict_size, expand_rank, save = True, sup_rcsls_k=10, sup_rcsls_tgt_rank=50000,
        sup_opt_params={"name": "SGD", "lr": 1.0},
        sup_bsz=-1, skipfirst=True
        ):
        logger = logging.getLogger(__name__)
        logger.info("Parallel Optimization(PSS) between Sup and UnSup")
        # train rcsls
        word_dict = self.batcher.pair2ix[f"{self.src}-{self.tgt}"]
        pairs = word_dict.word_map
        pairs = pairs[:sup_bsz]
        # init with procrutes
        logger.info("Initialize with procrutes")
        src, tgt = self.batcher.supervised_minibatch(-1, self.src, self.tgt)
        weight = self.procrustes_onestep(src, tgt)
        self.Q1 = weight.t()
        self.Q2 = self.Q1
        self.transform1.transform.weight.data.copy_(weight)
        self.transform2.transform.weight.data.copy_(weight)

        sup_lr = sup_opt_params["lr"]
        name = sup_opt_params.pop("name")

        if skipfirst:
            skipcount = 0
        else:
            skipcount = 2
        for epoch in range(epochs):
            start = time.time()
            logger.info("-------------------Start of Epoch {}/{}-------------------".format(epoch+1, epochs))
            # start optimization with RCSLS
            logger.info("-----Supervised RCSLS Optimization-----")
            if skipcount == 1:
                src, tgt = self.batcher.supervised_minibatch(-1, self.src, self.tgt)
                weight = self.procrustes_onestep(src, tgt)
                self.Q1 = weight.t()
                self.transform1.transform.weight.data.copy_(weight)
            skipcount += 1

            fold = np.inf
            sup_opt_params["lr"] = sup_lr
            rcsls_optimizer = getattr(optim, name)(self.transform1.parameters(), **sup_opt_params)
            logafter = sup_steps / 4
            for iter in range(1, sup_steps+1):
                if sup_opt_params["lr"] < 1e-4:
                    break
                rcsls_optimizer.zero_grad()
                src, tgt, nn_src, nn_tgt = self.batcher.supervised_rcsls_minibatch(sup_bsz, self.src, self.tgt, sup_rcsls_tgt_rank)
                loss = self.supervised_rcsls_loss(
                    src, tgt, nn_src, nn_tgt, k=sup_rcsls_k)
                f = loss.item()
                lr_str = sup_opt_params["lr"]
                if f > fold and batch_size == -1:
                    sup_opt_params["lr"] /= 2
                    rcsls_optimizer = getattr(optim, name)(
                        self.transform1.parameters(), **sup_opt_params)
                    f = fold
                else:
                    loss.backward()
                    rcsls_optimizer.step()
                    self.Q1 = self.transform1.transform.weight.data.t()
                if iter == 1 or iter == sup_steps + 1 or iter % logafter == 0:
                    logger.info("Sup. {0:4d}/{1:4d} iteration completes, supervied loss: {2:.4f}".format(iter, sup_steps, loss))
            self.PriorQ = self.Q1
            self.evaluate_test(self.transform1)
            logger.info("--------Supervised-Phase-Finised--------")
            # Unsupervised
            logger.info("-----Unsupervised-Phase-Optimization-----")
            first_batch = self.batcher.firstNbatch(20000)
            embj = first_batch[self.tgt]
            logafter = unsup_steps / 4
            for it in range(1, unsup_steps + 1):
                torch.cuda.empty_cache()
                rcsls_optimizer.zero_grad()
                # sample mini-batch
                mini_batch = self.batcher.minibatch(unsup_bsz)
                embi = mini_batch[self.src][1]
                T = self.computePrior(embi, embj, self.PriorQ, unsup_t)
                # update P and Q alterantively
                P = self.P_solver(embi, embj, T, unsup_epsilon)
                GQ = - torch.mm(embi.t(), P.mm(embj))
                self.orthogonal_mapping_update(GQ, unsup_lr/unsup_bsz)
                loss = torch.norm(torch.mm(embi, self.Q2) - torch.mm(P, embj))
                if it == 1 or it == unsup_steps + 1 or it % logafter == 0:
                    logger.info("Unsup. {0:2d}/{1:2d} iteration completes, unsupervised loss: {2:.4f}".format(it, unsup_steps, loss))
            self.evaluate_test(self.transform2)
            logger.info("-----Unsupervised-Phase-Finised-----")

            # expand the supervised dictionary
            pairs = self.expand_dict(self.Q2, expand_dict_size, expand_rank)
            self.batcher.expand_supervised(self, self.src, self.tgt, pairs)
            logger.info("Finished epoch ({0:d} / {1:d}). Took {2:.2f}s.".format(epoch + 1, epochs, time.time() - start))

        sup_opt_params["name"] = name
        logger.info("Finished Training after {0} epochs".format(epochs))

        self.selectBest()
        self.evaluate_test(self.transform)