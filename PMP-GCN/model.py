"""
Polarized message-passing graph convolution network (PMP-GCN)
Acknowledgement: The source code here is developed based on DGL. The authors would like to express their sincere
gratitude to DGL team.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
from dgl._ffi.base import DGLError
from dgl.utils import expand_as_pair
from dgl import reverse
from dgl.heterograph import DGLBlock
from dgl.convert import block_to_graph


def position_loss(hlatp, weights, graph):
    # latent position regularized by graph laplacian
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
    def __init__(self, nsample, nfeat, nhid, nclass, dropout, WL, random_split):
        """Network structure of PMP-GCN"""
        super(PMPGNN, self).__init__()
        self.dropout = dropout

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

        # Two-layer network structure
        self.hidden = PMPLayer(nfeat, nhid, nclass, WL=WL)

        self.out = PMPLayer(nhid, nclass, nclass, WL=WL, concat=False)

        # MLP layers for learning latent positions
        self.structure_layer = nn.Linear(nclass, nclass)
        self.structure_layer_out = nn.Linear(nclass, nclass)

    def forward(self, graph, x, weights):
        # x is input feature matrix, graph is a dgl graph, weights for laplacian regularization

        if self.random_split:
            x = self.dimension_reduct(x)

        x = F.dropout(x, self.dropout, training=self.training)

        hlatp = self.latp

        hlatp = F.dropout(hlatp, self.dropout, training=self.training)
        hlatp = F.elu(self.structure_layer(hlatp))

        x = self.hidden(graph, x, hlatp)

        x = F.dropout(x, self.dropout, training=self.training)

        hlatp = F.dropout(hlatp, self.dropout, training=self.training)
        hlatp = F.elu(self.structure_layer_out(hlatp))

        x = self.out(graph, x, hlatp)
        lp_loss = position_loss(hlatp, weights, graph)

        x = F.elu(x)
        x = F.log_softmax(x, dim=1)

        return x, lp_loss


class PMPLayer(nn.Module):
    # PMP-GCN layer
    def __init__(
            self,
            in_feats,
            out_feats,
            num_class,
            feat_drop=0.0,
            negative_slope=0.2,
            wl_drop=0.0,
            allow_zero_in_degree=False,
            autobalance=True,
            WL=True,
            concat=True,
    ):
        super(PMPLayer, self).__init__()

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.concat = concat
        self.WL = WL
        self.w_u = nn.Parameter(torch.FloatTensor(size=(1, 1, out_feats)))
        self.w_v = nn.Parameter(torch.FloatTensor(size=(1, 1, out_feats)))
        self.w_s = nn.Parameter(torch.FloatTensor(size=(1, num_class)))

        # activation for positive message
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.edge_norm = EdgeWeightNorm(norm='both')

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, out_feats, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, out_feats, bias=False)

        self.feat_drop = nn.Dropout(feat_drop)
        self.WL_drop = nn.Dropout(wl_drop)

        # automatically learn \beta
        self.beta = nn.Parameter(torch.zeros(size=(1, 1)))

        if autobalance:
            self.aw = nn.Parameter(torch.zeros(size=(1, 2)))
            self.autobalance = True
        else:
            self.fw = 0.75
            self.autobalance = False

        if self.WL:
            self.theta = nn.Parameter(torch.zeros(size=(1, 1)))

        self.reset_parameters()

    def reset_parameters(self):

        nn.init.xavier_normal_(self.beta.data, gain=1.414)
        nn.init.xavier_normal_(self.w_u, gain=1.414)
        nn.init.xavier_normal_(self.w_v, gain=1.414)
        nn.init.xavier_normal_(self.w_s, gain=1.414)

        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=1.414)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=1.414)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=1.414)
        if self.autobalance:
            nn.init.xavier_normal_(self.aw.data, gain=1.414)
            nn.init.xavier_uniform_(self.aw.data, gain=1.414)

        # learnable parameter for WL
        if self.WL:
            nn.init.xavier_normal_(self.theta.data, gain=1.414)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, lposition):

        sfeat_src = lposition

        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    assert False

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, "fc_src"):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src, feat_dst = h_src, h_dst
                feat_src = self.fc_src(h_src).view(-1, 1, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, 1, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src, feat_dst = h_src, h_dst
                feat_src = feat_dst = self.fc(h_src).view(-1, 1, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]

            if self.WL:
                in_degree = (graph.in_degrees() + 1e-9).unsqueeze(dim=1).unsqueeze(dim=1)
                perm = 1e-9 / (torch.exp(-self.theta) + 1)
                perm = perm / in_degree
                permfeat = self.WL_drop(feat_src * perm)

            el = (self.leaky_relu(feat_src) * self.w_u).sum(dim=-1).unsqueeze(-1)
            er = (self.leaky_relu(feat_dst) * self.w_v).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"el": el})
            graph.dstdata.update({"er": er})
            graph.apply_edges(fn.u_add_v("el", "er", "e"))
            e = graph.edata.pop("e")

            sel = sfeat_src * self.w_s
            graph.srcdata.update({"sel": sel})
            graph.dstdata.update({"ser": sfeat_src})
            graph.apply_edges(fn.u_mul_v("sel", "ser", "se"))
            se = graph.edata.pop("se")
            se = se.sum(dim=-1)
            se = se.view(-1, 1, 1)
            e = e + se

            # Node differences
            sdf = divergence_in_graph(feat_src, graph)
            sds = divergence_in_graph(sfeat_src, graph)
            sds = sds.unsqueeze(dim=1)

            # weighted sum of heterogeneous dissimilarity
            if self.autobalance:
                weight = nn.functional.softmax(self.aw, dim=1)
                d = weight[0, 0] * sdf + weight[0, 1] * sds
            else:
                d = self.fw * sdf + (1 - self.fw) * sds

            betaw = 2.0 / (torch.exp(-self.beta) + 1)

            eweight = torch.exp(e - betaw * d).view(-1)

            graph.edata['a'] = self.edge_norm(graph, eweight).view(-1, 1, 1)

            graph.srcdata.update({"ft": feat_src})

            # message passing
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]

            if self.WL:
                rst = rst + permfeat

            # activation
            if self.concat:
                rst = F.elu(torch.flatten(rst, start_dim=1))
            else:
                rst = torch.flatten(rst, start_dim=1)

            return rst
