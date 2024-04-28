"""
Polarized message-passing graph PageRank network (PMP-GPN)
Acknowledgement: The source code here is developed based on DGL. The authors would like to express their sincere
gratitude to DGL team.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
from dgl._ffi.base import DGLError
from dgl.ops import edge_softmax
from dgl import reverse
from dgl.heterograph import DGLBlock
from dgl.convert import block_to_graph


def position_loss(hlatp, weights, graph):
    # loss for latent position
    with graph.local_scope():
        graph.edata['lapw'] = weights
        graph.srcdata.update({"slp": hlatp})
        graph.dstdata.update({"dlp": hlatp})

        graph.apply_edges(fn.u_mul_e('slp', 'lapw', 's'))
        graph.apply_edges(fn.v_mul_e('dlp', 's', 'ss'))

        ss = graph.edata.pop('ss')
        lp_loss = torch.sum(ss, dim=-1)
        lp_loss = torch.sum(lp_loss)

        return lp_loss


def divergence_in_graph(h, graph):
    with graph.local_scope():
        graph.srcdata.update({"sh": h})
        graph.dstdata.update({"dsh": h})
        graph.apply_edges(fn.u_sub_v("sh", "dsh", "ds"))
        ds = graph.edata.pop("ds")
        ds = torch.pow(ds, 2)
        sds = torch.sum(ds, dim=-1)
        sds = sds.unsqueeze(dim=1)
        return sds


class EdgeWeightNorm(nn.Module):
    # Learning message weights for PMP-GPN
    def __init__(self, norm="both", eps=0.0):
        super(EdgeWeightNorm, self).__init__()
        self._norm = norm
        self._eps = eps

    def forward(self, graph, edge_weight):

        with graph.local_scope():
            if isinstance(graph, DGLBlock):
                graph = block_to_graph(graph)
            if len(edge_weight.shape) > 1:
                raise DGLError(
                    "Currently the normalization is only defined "
                    "on scalar edge weight. Please customize the "
                    "normalization for your high-dimensional weights."
                )
            edge_weight = edge_weight + 1e-9
            if self._norm == "both" and torch.any(edge_weight < 0).item():
                raise DGLError(
                    'Non-positive edge weight detected with `norm="both"`. '
                    "This leads to square root of zero or negative values."
                )

            graph.edata["_edge_w"] = edge_weight

            if self._norm == "both":
                reversed_g = reverse(graph)
                reversed_g.edata["_edge_w"] = edge_weight
                reversed_g.update_all(
                    fn.copy_e("_edge_w", "m"), fn.sum("m", "out_weight")
                )
                degs = reversed_g.dstdata["out_weight"] + self._eps
                norm = torch.pow(degs, -0.5)
                graph.srcdata["_src_out_w"] = norm

            if self._norm != "none":
                graph.update_all(
                    fn.copy_e("_edge_w", "m"), fn.sum("m", "in_weight")
                )
                degs = graph.dstdata["in_weight"] + self._eps
                if self._norm == "both":
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                graph.dstdata["_dst_in_w"] = norm

            graph.apply_edges(
                lambda e: {
                    "_norm_edge_weights": e.src["_src_out_w"]
                    * e.dst["_dst_in_w"]
                    * e.data["_edge_w"]
                }
            )
            return graph.edata["_norm_edge_weights"]


class PMPGNN(nn.Module):
    def __init__(self, nsample, nfeat, nhid, nclass, dropout, k, alpha, WL, random_split):
        """Network structure of PMP-GPN"""
        super(PMPGNN, self).__init__()
        self.dropout = dropout
        self.edge_dropout = dropout

        # latent position
        self.latp = nn.Parameter(torch.FloatTensor(size=(nsample, nclass)))
        nn.init.xavier_normal_(self.latp.data, gain=1.414)

        if random_split:

            self.random_split = random_split

            d_feat = 512

            self.dimension_reduct = nn.Linear(nfeat, d_feat, bias=False)

            nfeat = d_feat
        else:
            self.random_split = False

        # Two-layer network structure 1: MLP layers, 1 propagate layer
        # input layer
        self.mlp_input = nn.Linear(nfeat, nhid, bias=False)
        # output layer
        self.mlp_out = nn.Linear(nhid, nclass, bias=False)

        # appnp layers
        self.propagate = PMPLayer(k=k, out_feats=nclass, num_class=nclass, alpha=alpha,
                                  edge_drop=self.edge_dropout, WL=WL)

        # MLP layers for learning latent positions
        self.structure_layer = nn.Linear(nclass, nclass)
        self.structure_layer_out = nn.Linear(nclass, nclass)

    def forward(self, graph, x, weights):
        # x is input feature matrix, graph is a dgl graph, weights for laplacian regularization

        if self.random_split:
            x = self.dimension_reduct(x)

        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.mlp_input(x))
        x = self.mlp_out(x)
        x = F.dropout(x, self.dropout, training=self.training)

        hlatp = self.latp
        hlatp = F.dropout(hlatp, self.dropout, training=self.training)
        hlatp = F.elu(self.structure_layer(hlatp))
        hlatp = F.dropout(hlatp, self.dropout, training=self.training)
        hlatp = F.elu(self.structure_layer_out(hlatp))

        x = self.propagate(graph, x, hlatp)

        lp_loss = position_loss(hlatp, weights, graph)

        x = F.elu(x)
        x = F.log_softmax(x, dim=1)

        return x, lp_loss


class PMPLayer(nn.Module):
    # PMP-GPN layer is built upon APPNP layer
    def __init__(self,
                 k,
                 out_feats,
                 num_class,
                 alpha,
                 edge_drop=0.,
                 negative_slope=0.2,
                 wl_drop=0.0,
                 autobalance=True,
                 WL=True
                 ):
        super(PMPLayer, self).__init__()
        self._k = k
        self._alpha = alpha

        self.edge_drop = nn.Dropout(edge_drop)
        self.WL = WL
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, out_feats)))
        self.s_attn = nn.Parameter(torch.FloatTensor(size=(1, num_class)))
        self.edge_norm = EdgeWeightNorm(norm='both')
        self.beta = nn.Parameter(torch.zeros(size=(1, 1)))

        if autobalance:
            self.aw = nn.Parameter(torch.zeros(size=(1, 2)))
            self.autobalance = True
        else:
            self.fw = 0.75
            self.autobalance = False

        if self.WL:
            self.theta = nn.Parameter(torch.zeros(size=(1, 1)))
            self.WL_drop = nn.Dropout(wl_drop)

        nn.init.xavier_normal_(self.beta.data, gain=1.414)
        nn.init.xavier_normal_(self.attn_l, gain=1.414)
        nn.init.xavier_normal_(self.attn_r, gain=1.414)
        nn.init.xavier_normal_(self.s_attn, gain=1.414)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        if self.autobalance:
            nn.init.xavier_uniform_(self.aw.data, gain=1.414)

        # learnable parameter for WL
        if self.WL:
            nn.init.xavier_normal_(self.theta.data, gain=1.414)

    def forward(self, graph, feat, lposition):

        sfeat_src = lposition

        with graph.local_scope():

            feat_0 = feat

            el = (self.leaky_relu(feat_0) * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (self.leaky_relu(feat_0) * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"el": el})
            graph.dstdata.update({"er": er})
            graph.apply_edges(fn.u_add_v("el", "er", "e"))
            e = graph.edata.pop("e")

            sel = sfeat_src * self.s_attn
            graph.srcdata.update({"sel": sel})
            graph.dstdata.update({"ser": sfeat_src})
            graph.apply_edges(fn.u_mul_v("sel", "ser", "se"))
            se = graph.edata.pop("se")
            se = se.sum(dim=-1)
            se = se.view(-1, 1)
            e = e + se

            # Node differences
            sdf = divergence_in_graph(feat_0, graph)
            sdf = sdf.view(-1, 1)
            sds = divergence_in_graph(sfeat_src, graph)
            sds = sds.view(-1, 1)

            # weighted sum of heterogeneous divergences
            if self.autobalance:
                weight = nn.functional.softmax(self.aw, dim=1)
                d = weight[0, 0] * sdf + weight[0, 1] * sds
            else:
                d = self.fw * sdf + (1 - self.fw) * sds

            betaw = 2.0 / (torch.exp(-self.beta) + 1)

            eweight = torch.exp(e - betaw * d).view(-1)

            graph.edata['w'] = self.edge_norm(graph, eweight).view(-1, 1)

            if self.WL:
                perm = 1e-9 / (torch.exp(-self.theta) + 1)
                graph.edata['w'] = graph.edata['w'] + perm

            feat_pass = feat

            for i in range(self._k):

                graph.srcdata.update({"h": feat_pass})

                graph.update_all(fn.u_mul_e('h', 'w', 'm'),
                                 fn.sum('m', 'h'))

                feat_pass = (1-self._alpha) * graph.dstdata["h"] + self._alpha * feat_0

            return feat_pass
