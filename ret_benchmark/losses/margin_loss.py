import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ret_benchmark.losses.registry import LOSS


class DistanceWeightedSampling(object):
    """
    """
    def __init__(self, cfg):
        super(DistanceWeightedSampling, self).__init__()
        self.cutoff = cfg.LOSSES.MARGIN_LOSS.CUTOFF
        self.upper_cutoff = cfg.LOSSES.MARGIN_LOSS.UPPER_CUTOFF

    def sample(self, batch, labels):

        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        bs = batch.shape[0]
        distances = self.p_dist(batch.detach()).clamp(min=self.cutoff)

        positives, negatives = [], []

        for i in range(bs):
            pos = labels == labels[i]
            q_d_inv = self.inverse_sphere_distances(batch, distances[i], labels, labels[i])
            # sample positives randomly
            pos[i] = 0
            positives.append(np.random.choice(np.where(pos)[0]))
            # sample negatives by distance
            negatives.append(np.random.choice(bs, p=q_d_inv))

        sampled_triplets = [[a, p, n] for a, p, n in zip(list(range(bs)), positives, negatives)]
        return sampled_triplets

    @staticmethod
    def p_dist(A, eps=1e-4):
        prod = torch.mm(A, A.t())
        norm = prod.diag().unsqueeze(1).expand_as(prod)
        res = (norm + norm.t() - 2 * prod).clamp(min=0)
        return res.clamp(min=eps).sqrt()

    def inverse_sphere_distances(self, batch, dist, labels, anchor_label):
        bs, dim = len(dist), batch.shape[-1]
        # negated log-distribution of distances of unit sphere in dimension <dim>
        log_q_d_inv = ((2.0 - float(dim)) * torch.log(dist) - (float(dim-3) / 2)
                       * torch.log(1.0 - 0.25 * (dist.pow(2))))
        # set sampling probabilities of positives to zero
        log_q_d_inv[np.where(labels == anchor_label)[0]] = 0

        q_d_inv = torch.exp(log_q_d_inv - torch.max(log_q_d_inv))  # - max(log) for stability
        # set sampling probabilities of positives to zero
        q_d_inv[np.where(labels == anchor_label)[0]] = 0

        # NOTE: Cutting of values with high distances made the results slightly worse.
        # q_d_inv[np.where(dist > self.upper_cutoff)[0]] = 0

        q_d_inv = q_d_inv/q_d_inv.sum()
        return q_d_inv.detach().cpu().numpy()


@LOSS.register("margin_loss")
class MarginLoss(nn.Module):
    """Margin based loss with DistanceWeightedSampling
    """
    def __init__(self, cfg):
        super(MarginLoss, self).__init__()
        self.beta_val = 1.2
        self.margin = 0.2
        self.nu = 0.0
        self.n_classes = cfg.LOSSES.MARGIN_LOSS.N_CLASSES
        self.beta_constant = cfg.LOSSES.MARGIN_LOSS.BETA_CONSTANT
        if self.beta_constant:
            self.beta = self.beta_val
        else:
            self.beta = torch.nn.Parameter(torch.ones(self.n_classes)*self.beta_val)
        self.sampler = DistanceWeightedSampling(cfg)

    def forward(self, batch, labels):
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        sampled_triplets = self.sampler.sample(batch, labels)

        # compute distances between anchor-positive and anchor-negative.
        d_ap, d_an = [], []
        for triplet in sampled_triplets:
            train_triplet = {'Anchor': batch[triplet[0], :],
                             'Positive': batch[triplet[1], :], 'Negative': batch[triplet[2]]}
            pos_dist = ((train_triplet['Anchor']-train_triplet['Positive']).pow(2).sum()+1e-8).pow(1/2)
            neg_dist = ((train_triplet['Anchor']-train_triplet['Negative']).pow(2).sum()+1e-8).pow(1/2)

            d_ap.append(pos_dist)
            d_an.append(neg_dist)
        d_ap, d_an = torch.stack(d_ap), torch.stack(d_an)

        # group betas together by anchor class in sampled triplets (as each beta belongs to one class).
        if self.beta_constant:
            beta = self.beta
        else:
            beta = torch.stack([self.beta[labels[triplet[0]]] for
                                triplet in sampled_triplets]).type(torch.cuda.FloatTensor)
        # compute actual margin positive and margin negative loss
        pos_loss = F.relu(d_ap-beta+self.margin)
        neg_loss = F.relu(beta-d_an+self.margin)

        # compute normalization constant
        pair_count = torch.sum((pos_loss > 0.)+(neg_loss > 0.)).type(torch.cuda.FloatTensor)
        # actual Margin Loss
        loss = torch.sum(pos_loss+neg_loss) if pair_count == 0. else torch.sum(pos_loss+neg_loss)/pair_count

        # (Optional) Add regularization penalty on betas.
        # if self.nu: loss = loss + beta_regularisation_loss.type(torch.cuda.FloatTensor)
        return loss
