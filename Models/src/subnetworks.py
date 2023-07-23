# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
import numpy as np

from .utils import normalize_logprob, clamp_probabilities, inverse_softplus

SCALE_OFFSET = 1e-6


class SelectIndex(nn.Module):
    def __init__(self, index=0):
        """Helper module to select the tensor at the given index
        from a tuple of tensors.

        Args:
            index (int, optional):
                The index to select. Defaults to 0.
        """
        super().__init__()
        self.index = index

    def forward(self, x):
        return x[self.index]


class PositionalEncoding(nn.Module):
    def __init__(self, feat_dim, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, feat_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, feat_dim, 2).float() * (-np.log(10000.0) / feat_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if feat_dim > 1:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        _, B, _ = x.shape
        pe = self.pe[: x.size(0), :].repeat(1, B, 1)
        x = torch.cat([x, pe], dim=-1)
        return x


class TransformerEmbedder(nn.Module):
    def __init__(
        self,
        obs_dim,
        emb_dim=4,
        use_pe=True,
        nhead=1,
        dim_feedforward=32,
        dropout=0.1,
        n_layers=1,
    ):
        super().__init__()
        self.use_pe = use_pe
        self.linear_map = nn.Linear(obs_dim, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=2 * emb_dim if use_pe else emb_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.pos_encoder = PositionalEncoding(emb_dim) if use_pe else None
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)

    def forward(self, src):
        # Flip batch and time dim
        src = torch.transpose(src, 0, 1)
        src = self.linear_map(src)
        if self.use_pe:
            src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        # Flip batch and time dim back
        output = torch.transpose(output, 0, 1)
        return output


class RawControlToFeat(nn.Module):
    def __init__(self, network, n_staticfeat, n_timefeat, embedding_dim=50):
        super().__init__()
        self.n_staticfeat = n_staticfeat
        self.n_timefeat = n_timefeat
        self.embedding = nn.Embedding(
            num_embeddings=n_staticfeat,
            embedding_dim=embedding_dim,
        )
        self.network = network

    def forward(self, feat_static, n_timesteps, feat_time=None):
        feat_static = feat_static.type(torch.int64)
        feat_static_embed = self.embedding(feat_static.squeeze(dim=-1))[
            :, None, :
        ].repeat(1, n_timesteps, 1)
        if self.n_timefeat > 0:
            assert feat_time is not None, (
                "Time features cannot be None" "for n_timefeat > 0."
            )
            input_to_network = torch.cat([feat_static_embed, feat_time], dim=-1)
        else:
            input_to_network = feat_static_embed
        ctrl_feats = self.network(input_to_network)
        return ctrl_feats


class ControlToNSTF(nn.Module):
    def __init__(self, network, num_categories, d_max, d_min=1):
        super().__init__()
        self.network = network
        self.num_categories = num_categories
        assert d_min < d_max
        self.d_max = d_max
        self.d_min = d_min

    def forward(self, ctrl_feats, temperature=1.0):
        rho = self.rho(ctrl_feats, temperature=temperature)
        rho = torch.flip(rho, [-1])
        u = 1 - rho / torch.cumsum(rho, -1)
        u = u.flip([-1])
        log_u = torch.log(clamp_probabilities(u))
        return log_u

    def rho(self, ctrl_feats, temperature=1.0):
        B, T, _ = ctrl_feats.shape
        u_rho = self.network(ctrl_feats).view(B, T, self.num_categories, self.d_max)
        if self.d_min > 1:
            mask = torch.full_like(u_rho, np.log(1e-18))
            log_rho1 = mask[..., : self.d_min - 1]
            log_rho2 = normalize_logprob(
                u_rho[..., self.d_min - 1 :], axis=-1, temperature=temperature
            )[0]
            log_rho = torch.cat([log_rho1, log_rho2], dim=-1)
            assert log_rho.size() == u_rho.size()
        else:
            log_rho = normalize_logprob(u_rho, axis=-1, temperature=temperature)[0]
        rho = clamp_probabilities(torch.exp(log_rho))
        return rho


class InitialDiscreteDistribution(nn.Module):
    def __init__(
        self, 
        network, 
        num_categories, 
        takes_ctrl=False,
        no_interaction=True,
    ):
        super().__init__()
        self.network = network
        self.num_categories = num_categories
        self.takes_ctrl = takes_ctrl
        self.no_interaction=no_interaction

    def forward(self, ctrl_feats0):
        B, O, _ = ctrl_feats0.shape
        if not self.takes_ctrl:
            ctrl_feats0 = torch.ones(B, O, 1, device=ctrl_feats0.device)
        if not self.no_interaction:
            ctrl_feats0 = ctrl_feats0.view(B, -1)
        h = self.network(ctrl_feats0)
        if not self.no_interaction:
            h = h.view(B, O, -1)
        return normalize_logprob(h, axis=-1)[0]


class InitialContinuousDistribution(nn.Module):
    def __init__(
        self,
        networks,
        dist_dim,
        num_categories,
        use_tied_cov=False,
        use_trainable_cov=False,
        sigma=None,
        takes_ctrl=False,
        max_scale=1.0,
        scale_nonlinearity="softplus",
    ):
        super().__init__()
        assert (
            len(networks) == num_categories
        ), "The number of networks != num_categories!"
        self.x0_networks = nn.ModuleList(networks)
        self.K = num_categories
        self.use_tied_cov = use_tied_cov
        self.use_trainable_cov = use_trainable_cov
        self.max_scale = max_scale
        self.scale_nonlinearity = scale_nonlinearity
        if not self.use_trainable_cov:
            assert (
                sigma is not None
            ), "sigma cannot be None for non-trainable covariance!"
            self.sigma = sigma
        if self.use_trainable_cov and self.use_tied_cov:
            self.usigma = nn.Parameter(
                inverse_softplus(
                    torch.full(
                        [num_categories, dist_dim], sigma if sigma is not None else 1e-1
                    )
                )
            )
        self.dist_dim = dist_dim
        self.takes_ctrl = takes_ctrl

    def forward(self, ctrl_feats0):
        B, _ = ctrl_feats0.shape
        if not self.takes_ctrl:
            ctrl_feats0 = torch.ones(B, 1, device=ctrl_feats0.device)
        args_tensor = torch.stack([net(ctrl_feats0) for net in self.x0_networks])
        args_tensor = args_tensor.permute(1, 0, 2)
        mean_tensor = args_tensor[..., :self.dist_dim]
        if self.use_trainable_cov:
            if self.use_tied_cov:
                if self.scale_nonlinearity == "sigmoid":
                    scale_tensor = self.max_scale * torch.sigmoid(self.usigma)
                else:
                    scale_tensor = F.softplus(self.usigma)
                out_dist = td.normal.Normal(mean_tensor, scale_tensor + SCALE_OFFSET) # SCALE_OFFSET = 1e-6
                out_dist = td.independent.Independent(out_dist, 1)
            else:
                if self.scale_nonlinearity == "sigmoid":
                    scale_tensor = self.max_scale * torch.sigmoid(
                        args_tensor[..., self.dist_dim:]
                    )
                else:
                    scale_tensor = F.softplus(args_tensor[..., self.dist_dim:])
                out_dist = td.normal.Normal(mean_tensor, scale_tensor + SCALE_OFFSET)
                out_dist = td.independent.Independent(out_dist, 1)
        else:
            out_dist = td.normal.Normal(mean_tensor, self.sigma)
            out_dist = td.independent.Independent(out_dist, 1)
        return out_dist


class ContinuousStateTransition(nn.Module):
    def __init__(
        self,
        transition_networks,
        dist_dim,
        num_categories,
        use_tied_cov=False,
        use_trainable_cov=False,
        sigma=None,
        takes_ctrl=False,
        max_scale=1.0,
        scale_nonlinearity="softplus",
    ):
        """A torch module that models p(x[t] | x[t-1], z[t]).

        Args:
            transition_networks (List[nn.Module]):
                List of torch modules of length num_categories.
                The k-th element of the list is a neural network that
                outputs the parameters of the distribution
                p(x[t] | x[t-1], z[t] = k).
            dist_dim (int):
                The dimension of the random variables x[t].
            num_categories (int):
                The number of discrete states.
            use_tied_cov (bool, optional):
                Whether to use a tied covariance matrix.
                Defaults to False.
            use_trainable_cov (bool, optional):
                True if the covariance matrix is to be learned.
                Defaults to True. If False, the covariance matrix is set to I.
            takes_ctrl (bool, optional):
                True if the dataset has control inputs.
                Defaults to False.
            max_scale (float, optional):
                Maximum scale when using sigmoid non-linearity.
            scale_nonlinearity (str, optional):
                Which non-linearity to use for scale -- sigmoid or softplus.
                Defaults to softplus.
        """
        super().__init__()

        assert (
            len(transition_networks) == num_categories
        ), "The number of transition networks != num_categories!"
        self.x_trans_networks = nn.ModuleList(transition_networks)
        self.K = num_categories
        self.use_tied_cov = use_tied_cov
        self.use_trainable_cov = use_trainable_cov
        self.max_scale = max_scale
        self.scale_nonlinearity = scale_nonlinearity
        if not self.use_trainable_cov:
            assert (
                sigma is not None
            ), "sigma cannot be None for non-trainable covariance!"
            self.sigma = sigma
        if self.use_trainable_cov and self.use_tied_cov:
            self.usigma = nn.Parameter(
                inverse_softplus(
                    torch.full(
                        [num_categories, dist_dim], sigma if sigma is not None else 0.1
                    )
                )
            )
        self.dist_dim = dist_dim
        self.takes_ctrl = takes_ctrl

    def forward(self, x, ctrl_feats):
        """The forward pass.

        Args:
            x (torch.Tensor):
                Pseudo-observations x[1:T] sampled from the variational
                distribution q(x[1:T] | y[1:T]).
                Expected shape: [batch, time, x_dim]

        Returns:
            out_dist:
                The Gaussian distributions p(x[t] | x[t-1], z[t]).
        """
        B, T, dist_dim = x.shape
        assert self.dist_dim == dist_dim, "The distribution dimensions do not match!"
        if self.takes_ctrl:
            assert (
                ctrl_feats is not None
            ), "ctrl_feats cannot be None when self.takes_ctrl = True!"
            # Concat observations and controls on feature dimension
            x = torch.cat([x, ctrl_feats], dim=-1)
        args_tensor = torch.stack([net(x) for net in self.x_trans_networks])
        args_tensor = args_tensor.permute(1, 2, 0, 3)
        mean_tensor = args_tensor[..., :dist_dim]
        if self.use_trainable_cov:
            if self.use_tied_cov:
                if self.scale_nonlinearity == "sigmoid":
                    scale_tensor = self.max_scale * torch.sigmoid(self.usigma)
                else:
                    scale_tensor = F.softplus(self.usigma)
                out_dist = td.normal.Normal(mean_tensor, scale_tensor + SCALE_OFFSET)
                out_dist = td.independent.Independent(out_dist, 1)
            else:
                if self.scale_nonlinearity == "sigmoid":
                    scale_tensor = self.max_scale * torch.sigmoid(
                        args_tensor[..., dist_dim:]
                    )
                else:
                    scale_tensor = F.softplus(args_tensor[..., dist_dim:])
                out_dist = td.normal.Normal(mean_tensor, scale_tensor + SCALE_OFFSET)
                out_dist = td.independent.Independent(out_dist, 1)
        else:
            out_dist = td.normal.Normal(mean_tensor, self.sigma)
            out_dist = td.independent.Independent(out_dist, 1)
        return out_dist


class DiscreteStateTransition(nn.Module):
    def __init__(
        self,
        transition_network,
        num_categories,
        takes_ctrl=False,
        takes_x=True,
        takes_y=False,
        takes_hidden_states=False, 
        no_interaction=True,
        interaction_simple=False,
        interaction_gnn=False,
        interaction_gnn_fully_connected=False,
        interaction_gnn_est_edge=False,
        interaction_gnn_gt_edge=False,
        n_samples=1,
        n_obj=1,
        x_size=1,
        y_size=1,
        hidden_states_size=1,
    ):
        """A torch module that models p(z[t] | z[t-1], x[t-1]).

        Args:
            transition_network (nn.Module):
                A torch module that outputs the parameters of the
                distribution p(z[t] | z[t-1], x[t-1]).
            num_categories (int):
                The number of discrete states.
            takes_x (bool, optional):
                Whether there is recurrent connection from state to switch.
            takes_y (bool, optional):
                Whether there is recurrent connection from obs to switch.
            takes_ctrl (bool, optional):
                True if the dataset has control inputs.
        """
        super().__init__()
        self.network = transition_network
        self.K = num_categories
        self.takes_ctrl = takes_ctrl
        self.takes_x = takes_x
        self.takes_y = takes_y
        self.takes_hidden_states = takes_hidden_states
        self.no_interaction=no_interaction
        self.interaction_simple = interaction_simple
        self.interaction_gnn = interaction_gnn
        self.interaction_gnn_fully_connected = interaction_gnn_fully_connected
        self.interaction_gnn_est_edge = interaction_gnn_est_edge
        self.interaction_gnn_gt_edge = interaction_gnn_gt_edge
        self.n_samples = n_samples
        self.n_obj = n_obj
        self.x_size = x_size
        self.y_size = y_size
        self.hidden_states_size = hidden_states_size

        from .model_utils import encode_onehot
        edges = np.ones(self.n_obj) - np.eye(self.n_obj)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]
        self.edge2node_mat = nn.Parameter(torch.FloatTensor(encode_onehot(self.recv_edges).transpose()), requires_grad=False)
        self.skip_first_edge_type = True # skip the first NO-EDGE type
        in_size = (self.x_size if self.takes_x else 0) + (self.y_size if self.takes_y else 0) + \
                  (self.hidden_states_size if self.takes_hidden_states else 0) + \
                  (self.K if self.interaction_gnn else 0)
        self.num_edge_types = 2
        self.msg_hid_shape = in_size * self.n_obj
        self.msg_out_shape = in_size * self.n_obj
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * in_size, self.msg_hid_shape) for _ in range(self.num_edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(self.msg_hid_shape, self.msg_out_shape) for _ in range(self.num_edge_types)])

    def forward(self, y, x, hidden_states, forward_probs, edge_est, edge_gt, ctrl_feats=None):
        """The forward pass.

        Args:
            y (torch.Tensor):
                The observations.
                Expected shape: [batch, time, obs_dim]

        Returns:
            transition_tensor:
                The unnormalized transition matrix for each timestep.
                Output shape: [batch, time, num_categories, num_categories]
        """
        # print(y.shape, x.shape, hidden_states.shape)
        # print(edge_est.shape, edge_gt.shape) 
        # edge_gt: (1,0; 2,0; 2,1) edge_est: (0,1; 0,2; 1,0; 1,2; 2,0; 2,1)
        # edge_gt: (1,0; 2,0; 2,1; 3,0; 3,1; 3,2) edge_est: (0,1; 0,2; 0,3; 1,0; 1,2; 1,3; 2,0; 2,1; 2,3; 3,0; 3,1; 3,2)

        B, O, T, _ = y.shape
        inputs_to_net = []
        if self.takes_x:
            inputs_to_net += [x]
        if self.takes_y:
            inputs_to_net += [y]
        if self.takes_hidden_states:
            inputs_to_net += [hidden_states]
        if self.takes_ctrl:
            assert (
                ctrl_feats is not None
            ), "ctrl_feats cannot be None when self.takes_ctrl = True!"
            inputs_to_net += [ctrl_feats]
        if self.interaction_gnn:
            inputs_to_net = inputs_to_net + [forward_probs]
        inputs_to_net = torch.cat(inputs_to_net, dim=-1)
        # No recurrence
        if len(inputs_to_net) == 0:
            dummy_inputs = torch.ones(B, O, T, 1, device=y.device)
            inputs_to_net += [dummy_inputs]
            inputs_to_net = torch.cat(inputs_to_net, dim=-1)
        inputs_to_net = inputs_to_net.permute(0,2,1,3).contiguous() # B, T, O, D

        if self.no_interaction:
            inputs_to_net_all = inputs_to_net
            transition_tensor = self.network(inputs_to_net_all).permute(0,2,1,3).view([B, O, T, self.K, self.K])
        elif self.interaction_simple:
            x_all_objects = x.permute(0,2,1,3).contiguous().view(B, T, self.n_obj*self.x_size)
            y_all_objects = y.permute(0,2,1,3).contiguous().view(B, T, self.n_obj*self.y_size)
            hidden_states_all_objects = hidden_states.permute(0,2,1,3).contiguous().view(B, T, self.n_obj*self.hidden_states_size)
            inputs_to_net_holistic = []
            if self.takes_x:
                inputs_to_net_holistic += [x_all_objects]
            if self.takes_y:
                inputs_to_net_holistic += [y_all_objects]
            if self.takes_hidden_states:
                inputs_to_net_holistic += [hidden_states_all_objects]
            inputs_to_net_holistic = torch.cat(inputs_to_net_holistic, dim=-1)
            inputs_to_net_holistic_final = inputs_to_net_holistic[:,:,None,:].repeat(1,1,self.n_obj,1)
            inputs_to_net_all = torch.cat([inputs_to_net, inputs_to_net_holistic_final], dim=-1)
            transition_tensor = self.network(inputs_to_net_all).permute(0,2,1,3).view([B, O, T, self.n_obj*self.K, self.K])
        elif self.interaction_gnn:
            if self.interaction_gnn_fully_connected:
                if inputs_to_net.is_cuda:
                    edge_used = torch.cuda.FloatTensor(B, T, self.n_obj*(self.n_obj-1), self.num_edge_types).fill_(0.)
                else:
                    edge_used = torch.zeros(B, T, self.n_obj*(self.n_obj-1), self.num_edge_types)
                edge_used[:,:,:,1] = 1
            elif self.interaction_gnn_est_edge:
                from .model_utils import gumbel_softmax
                gumbel_temp = 0.5
                hard_sample = not self.training
                old_shape = edge_est.shape
                edge_used = gumbel_softmax(
                    edge_est.reshape(-1, old_shape[-1]), 
                    tau=gumbel_temp, 
                    hard=hard_sample).view(old_shape) # B, T, E, D
            elif self.interaction_gnn_gt_edge:
                if inputs_to_net.is_cuda:
                    edge_used_LowerPart = torch.cuda.FloatTensor(B, T, self.n_obj, self.n_obj-1).fill_(0.)
                    edge_used = torch.cuda.FloatTensor(B, T, self.n_obj*(self.n_obj-1), self.num_edge_types).fill_(0.)
                else:
                    edge_used_LowerPart = torch.zeros(B, T, self.n_obj, self.n_obj-1)
                    edge_used = torch.zeros(B, T, self.n_obj*(self.n_obj-1), self.num_edge_types)
                # edge_gt: (1,0; 2,0; 2,1)    
                # edge_est: (0,1; 0,2;   (0,0; 0,1;
                #            1,0; 1,2;    1,0; 1,1;
                #            2,0; 2,1)    2,0; 2,1)
                idx_edge_gt = -1
                for i_obj in range(self.n_obj):
                    for j_obj in range(i_obj):
                        idx_edge_gt = idx_edge_gt + 1
                        edge_used_LowerPart[:,:,i_obj,j_obj] = edge_gt[:,:,idx_edge_gt]
                        edge_used_LowerPart[:,:,j_obj,i_obj-1] = edge_gt[:,:,idx_edge_gt]
                edge_used_LowerPart = edge_used_LowerPart.view(B, T, self.n_obj*(self.n_obj-1)) 
                edge_used_UpperPart = 1 - edge_used_LowerPart
                edge_used[:,:,:,0] = edge_used_UpperPart
                edge_used[:,:,:,1] = edge_used_LowerPart
        
            receivers = inputs_to_net[:, :, self.recv_edges, :]
            senders = inputs_to_net[:, :, self.send_edges, :]
            pre_msg = torch.cat([receivers, senders], dim=-1)
            if inputs_to_net.is_cuda:
                all_msgs = torch.cuda.FloatTensor(pre_msg.size(0), pre_msg.size(1), pre_msg.size(2),
                                    self.msg_out_shape).fill_(0.)
            else:
                all_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1), pre_msg.size(2),
                                    self.msg_out_shape)
            if self.skip_first_edge_type:
                start_idx = 1
            else:
                start_idx = 0
            # Run separate MLP for every edge type
            for i in range(start_idx, len(self.msg_fc1)):
                msg = F.relu(self.msg_fc1[i](pre_msg))
                msg = F.relu(self.msg_fc2[i](msg))
                msg = msg * edge_used[:, :, :, i:i+1]
                all_msgs += msg
            # Aggregate all msgs to receiver
            agg_msgs = torch.matmul(self.edge2node_mat, all_msgs)
            # Skip connection
            inputs_to_net_all = torch.cat([inputs_to_net, agg_msgs], dim=-1)
            transition_tensor = self.network(inputs_to_net_all).permute(0,2,1,3).view([B, O, T, self.K, self.K])

        return transition_tensor

class GaussianDistributionOutput(nn.Module):
    def __init__(
        self,
        network,
        dist_dim,
        use_tied_cov=False,
        use_trainable_cov=True,
        sigma=None,
        max_scale=1.0,
        scale_nonlinearity="softplus",
    ):
        """A Gaussian distribution on top of a neural network.

        Args:
            network (nn.Module):
                A torch module that outputs the parameters of the
                Gaussian distribution.
            dist_dim ([type]):
                The dimension of the Gaussian distribution.
            use_tied_cov (bool, optional):
                Whether to use a tied covariance matrix.
                Defaults to False.
            use_trainable_cov (bool, optional):
                True if the covariance matrix is to be learned.
                Defaults to True. If False, the covariance matrix is set to I.
            sigma (float, optional):
                Initial value of scale.
            max_scale (float, optional):
                Maximum scale when using sigmoid non-linearity.
            scale_nonlinearity (str, optional):
                Which non-linearity to use for scale -- sigmoid or softplus.
                Defaults to softplus.
        """
        super().__init__()
        self.dist_dim = dist_dim
        self.network = network
        self.use_tied_cov = use_tied_cov
        self.use_trainable_cov = use_trainable_cov
        self.max_scale = max_scale
        self.scale_nonlinearity = scale_nonlinearity
        if not self.use_trainable_cov:
            assert (
                sigma is not None
            ), "sigma cannot be None for non-trainable covariance!"
            self.sigma = sigma
        if self.use_trainable_cov and self.use_tied_cov:
            self.usigma = nn.Parameter(
                inverse_softplus(
                    torch.full([1, dist_dim], sigma if sigma is not None else 1e-1)
                )
            )

    def forward(self, tensor):
        """The forward pass.

        Args:
            tensor (torch.Tensor):
                The input tensor.

        Returns:
            out_dist:
                The Gaussian distribution with parameters obtained by passing
                the input tensor through self.network.
        """
        args_tensor = self.network(tensor)
        mean_tensor = args_tensor[..., :self.dist_dim]
        if self.use_trainable_cov:
            if self.use_tied_cov:
                if self.scale_nonlinearity == "sigmoid":
                    scale_tensor = self.max_scale * torch.sigmoid(self.usigma)
                else:
                    scale_tensor = F.softplus(self.usigma)
                out_dist = td.normal.Normal(mean_tensor, scale_tensor + SCALE_OFFSET)
                out_dist = td.independent.Independent(out_dist, 1)
            else:
                if self.scale_nonlinearity == "sigmoid":
                    scale_tensor = self.max_scale * torch.sigmoid(
                        args_tensor[..., self.dist_dim :]
                    )
                else:
                    scale_tensor = F.softplus(args_tensor[..., self.dist_dim :])

                out_dist = td.normal.Normal(mean_tensor, scale_tensor + SCALE_OFFSET)
                out_dist = td.independent.Independent(out_dist, 1)
        else:
            out_dist = td.normal.Normal(mean_tensor, self.sigma)
            out_dist = td.independent.Independent(out_dist, 1)
        return out_dist


class RNNInferenceNetwork(nn.Module):
    def __init__(
        self,
        posterior_rnn,
        posterior_dist,
        x_dim,
        embedding_network=None,
        takes_ctrl=False,
        n_obj=3,
    ):
        """A torch module that models, q(x[1:T] | y[1:T]), the variational
        distribution of the continuous state.

        Args:
            posterior_rnn (nn.Module):
                The causal rnn with the following recurrence:
                    r[t] = posterior_rnn([h[t], x[t-1]], r[t-1])
                where r is hidden state of the rnn, x is the continuous
                latent variable, and h is an embedding of the observations y
                which is output by the embedding_network.
            posterior_dist (nn.Module):
                A torch module that models q(x[t] | x[1:t-1], y[1:T]) given the
                hidden state r[t] of the posterior_rnn.
            x_dim (int):
                The dimension of the random variables x[t].
            embedding_network (nn.Module, optional):
                A neural network that outputs the embedding h[1:T]
                of the observations y[1:T]. Defaults to None.
            takes_ctrl (bool, optional):
                True if the dataset has control inputs.
        """
        super().__init__()
        self.x_dim = x_dim
        self.posterior_rnn = posterior_rnn
        self.posterior_dist = posterior_dist
        if embedding_network is None:
            embedding_network = lambda x: x  # noqa: E731
        self.embedding_network = embedding_network
        self.takes_ctrl = takes_ctrl

        # for edge distill
        inp_size = 32
        hidden_size = 64
        dropout = 0.0
        no_bn = True
        import src.model_utils as model_utils
        self.mlp1 = model_utils.RefNRIMLP(inp_size, hidden_size, hidden_size, dropout, no_bn=no_bn)
        self.mlp2 = model_utils.RefNRIMLP(hidden_size * 2, hidden_size, hidden_size, dropout, no_bn=no_bn)
        self.mlp3 = model_utils.RefNRIMLP(hidden_size, hidden_size, hidden_size, dropout, no_bn=no_bn)
        self.mlp4 = model_utils.RefNRIMLP(hidden_size * 3, hidden_size, hidden_size, dropout, no_bn=no_bn)
        from .model_utils import encode_onehot
        self.n_obj = n_obj
        edges = np.ones(self.n_obj) - np.eye(self.n_obj)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]
        self.edge2node_mat = nn.Parameter(torch.FloatTensor(encode_onehot(self.recv_edges).transpose()), requires_grad=False)
        self.forward_rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.num_edge_types = 2
        self.edge_fc_out = nn.Linear(hidden_size, self.num_edge_types)

    def node2edge(self, node_embeddings):
        # Input size: [batch, num_vars, num_timesteps, embed_size]
        if len(node_embeddings.shape) == 4:
            send_embed = node_embeddings[:, self.send_edges, :, :]
            recv_embed = node_embeddings[:, self.recv_edges, :, :]
        else:
            send_embed = node_embeddings[:, self.send_edges, :]
            recv_embed = node_embeddings[:, self.recv_edges, :]
        return torch.cat([send_embed, recv_embed], dim=-1)

    def edge2node(self, edge_embeddings):
        if len(edge_embeddings.shape) == 4:
            old_shape = edge_embeddings.shape
            tmp_embeddings = edge_embeddings.view(old_shape[0], old_shape[1], -1)
            incoming = torch.matmul(self.edge2node_mat, tmp_embeddings).view(old_shape[0], -1, old_shape[2], old_shape[3])
        else:
            incoming = torch.matmul(self.edge2node_mat, edge_embeddings)
        return incoming/(self.n_obj-1) #TODO: do we want this average?

    def forward(self, y, ctrl_feats=None, num_samples=1, deterministic=False):
        """The forward pass.

        Args:
            y (torch.Tensor):
                The observations.
                Expected shape: [batch, time, obs_dim]
            num_samples (int):
                Number of samples for the Importance Weighted (IW) bound.

        Returns:
            x_samples:
                Pseudo-observations x[1:T] sampled from the variational
                distribution q(x[1:T] | y[1:T]).
                Output shape: [batch, time, x_dim]
            entropies:
                The entropy of the variational posterior q(x[1:T] | y[1:T]).
                Each element on the time axis represents the entropy of
                the distribution q(x[t] | x[1:t-1], y[1:T]).
                Output shape: [batch, time]
            log_probs:
                The log probability q(x[1:T] | y[1:T]).
                Each element on the time axis represents the log probability
                q(x[t] | x[1:t-1], y[1:T]).
                Output shape: [batch, time]
        """
        B_old, O, T, D = y.shape
        y = y.view(-1, T, D)
        B, _, _ = y.shape
        
        latent_dim = self.x_dim
        y = self.embedding_network(y)

        # for edge distill
        y_ForEdge = y.view(B_old, O, T, -1)
        y_ForEdge = y_ForEdge.repeat(num_samples, 1, 1, 1)
        y_ForEdge = self.mlp1(y_ForEdge)  # 2-layer ELU net per node
        y_ForEdge = self.node2edge(y_ForEdge)
        y_ForEdge = self.mlp2(y_ForEdge)
        y_ForEdge_skip = y_ForEdge
        y_ForEdge = self.edge2node(y_ForEdge)
        y_ForEdge = self.mlp3(y_ForEdge)
        y_ForEdge = self.node2edge(y_ForEdge)
        y_ForEdge = torch.cat((y_ForEdge, y_ForEdge_skip), dim=-1)  # Skip connection
        y_ForEdge = self.mlp4(y_ForEdge)
        old_shape = y_ForEdge.shape
        y_ForEdge = y_ForEdge.contiguous().view(-1, old_shape[2], old_shape[3])
        forward_y_ForEdge, prior_state = self.forward_rnn(y_ForEdge)
        edge_type_result = self.edge_fc_out(forward_y_ForEdge).view(old_shape[0], old_shape[1], old_shape[2], self.num_edge_types).transpose(1,2).contiguous()

        # for continuous state
        if self.takes_ctrl:
            assert (
                ctrl_feats is not None
            ), "ctrl_feats cannot be None when self.takes_ctrl = True!"
            # Concat observations and controls on feature dimension
            y = torch.cat([y, ctrl_feats], dim=-1)
        if self.posterior_rnn is not None:
            #  Initialize the hidden state or the RNN and the latent sample
            h0 = torch.zeros(
                B * num_samples, self.posterior_rnn.hidden_size, device=y.device
            ) # self.posterior_rnn.hidden_size == 16
            l0 = torch.zeros(B * num_samples, latent_dim, device=y.device)
            hh, ll = h0, l0
        x_samples = []
        entropies = []
        log_probs = []
        # for modeling interactions between objects
        hidden_states_list = []
        for t in range(T):
            yt = y[:, t, :]
            #  Repeat y samples for braodcast
            yt_tiled = yt.repeat(num_samples, 1)

            if self.posterior_rnn is not None:
                #  Concatenate yt with x[t-1]
                rnn_in = torch.cat([yt_tiled, ll], dim=-1)
                #  Feed into the RNN cell and get the hidden state
                hh = self.posterior_rnn(rnn_in, hh)
            else:
                hh = yt_tiled
            # for modeling interactions between objects
            hidden_states_list.append(hh)
            #  Construct the distribution q(x[t] | x[1:t-1], y[1:T])
            dist = self.posterior_dist(hh)
            #  Sample from ll ~ q(x[t] | x[1:t-1], y[1:T])
            if deterministic:
                ll = dist.mean
            else:
                ll = dist.rsample()
            x_samples.append(ll)
            #  Compute the entropy of q(x[t] | x[1:t-1], y[1:T])
            entropies.append(dist.entropy())
            #  Compute the log_prob of ll under q(x[t] | x[1:t-1], y[1:T])
            log_probs.append(dist.log_prob(ll))

        x_samples = torch.stack(x_samples)
        # T x (B * num_samples) x latent_dim
        x_samples = x_samples.permute(1, 0, 2).view(num_samples, B, T, latent_dim).view(num_samples, B_old, O, T, latent_dim)
        entropies = torch.stack(entropies)  # T x (B * num_samples)
        entropies = entropies.permute(1, 0).view(num_samples, B, T).view(num_samples, B_old, O, T)
        log_probs = torch.stack(log_probs)  # T x (B * num_samples)
        log_probs = log_probs.permute(1, 0).view(num_samples, B, T).view(num_samples, B_old, O, T)
        # for modeling interactions between objects
        hidden_states_list = torch.stack(hidden_states_list)
        hidden_states_list= hidden_states_list.permute(1, 0, 2).view(num_samples, B, T, self.posterior_rnn.hidden_size).view(num_samples, B_old, O, T, self.posterior_rnn.hidden_size)
        return x_samples, entropies, log_probs, hidden_states_list, edge_type_result