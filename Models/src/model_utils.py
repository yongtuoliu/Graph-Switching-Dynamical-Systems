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
import torch.nn.functional as F

from .subnetworks import (
    InitialContinuousDistribution,
    ContinuousStateTransition,
    InitialDiscreteDistribution,
    DiscreteStateTransition,
    GaussianDistributionOutput,
    RNNInferenceNetwork,
    SelectIndex,
    RawControlToFeat,
    ControlToNSTF,
    TransformerEmbedder,
)
from .model import SNLDS, REDSDS

def build_model(config):
    (emission_network,
        posterior_rnn,
        posterior_mlp,
        x0_networks,
        x_transition_networks,
        z0_network,
        z_transition_network,
        embedding_network,
        ctrl_feat_network,
        duration_network,
    ) = get_network_instances(config)

    x_init = InitialContinuousDistribution(
        networks=x0_networks,
        dist_dim=config["x_dim"],
        num_categories=config["num_categories"],
        use_tied_cov=config["initial_state"]["tied_cov"], # True
        use_trainable_cov=config["initial_state"]["trainable_cov"], # True
        sigma=config["initial_state"]["fixed_sigma"], # 0.02
        takes_ctrl=config["control"]["x"], # False
        max_scale=config["initial_state"].get("max_scale", 1.0), # 2
        scale_nonlinearity=config["initial_state"].get(
            "scale_nonlinearity", "softplus" # softplus
        ),
    )

    x_transition = ContinuousStateTransition(
        transition_networks=x_transition_networks,
        dist_dim=config["x_dim"],
        num_categories=config["num_categories"],
        use_tied_cov=config["continuous_transition"]["tied_cov"], # False
        use_trainable_cov=config["continuous_transition"]["trainable_cov"], # True
        sigma=config["continuous_transition"]["fixed_sigma"], # 0.02
        takes_ctrl=config["control"]["x"], # False
        max_scale=config["continuous_transition"].get("max_scale", 1.0), # 2
        scale_nonlinearity=config["continuous_transition"].get(
            "scale_nonlinearity", "softplus" # softplus
        ),
    )

    z_init = InitialDiscreteDistribution(
        network=z0_network,
        num_categories=config["num_categories"],
        takes_ctrl=config["control"]["z"],
        no_interaction=config["discrete_transition"]["no_interaction"],
    )

    z_transition = DiscreteStateTransition(
        transition_network=z_transition_network,
        num_categories=config["num_categories"],
        takes_ctrl=config["control"]["z"],
        takes_x=config["discrete_transition"]["takes_x"], # True
        takes_y=config["discrete_transition"]["takes_y"], # True
        takes_hidden_states=config["discrete_transition"]["takes_hidden_states"], 
        no_interaction=config["discrete_transition"]["no_interaction"],
        interaction_simple=config["discrete_transition"]["interaction_simple"],
        interaction_gnn=config["discrete_transition"]["interaction_gnn"],
        interaction_gnn_fully_connected=config["discrete_transition"]["interaction_gnn_fully_connected"],
        interaction_gnn_est_edge=config["discrete_transition"]["interaction_gnn_est_edge"],
        interaction_gnn_gt_edge=config["discrete_transition"]["interaction_gnn_gt_edge"],
        n_samples=config["num_samples"], # 1
        n_obj=config["n_obj"], # 3
        x_size=config["x_dim"],
        y_size=config["obs_dim"],
        hidden_states_size=config["inference"]["causal_rnndim"],
    )

    emission_network = GaussianDistributionOutput(
        network=emission_network,
        dist_dim=config["obs_dim"],
        use_tied_cov=config["emission"]["tied_cov"], # True
        use_trainable_cov=config["emission"]["trainable_cov"], # True
        sigma=config["emission"]["fixed_sigma"], # 0.02
        max_scale=config["emission"].get("max_scale", 1.0), # 2
        scale_nonlinearity=config["emission"].get("scale_nonlinearity", "softplus"),
    )

    posterior_dist = GaussianDistributionOutput(
        network=posterior_mlp,
        dist_dim=config["x_dim"],
        use_tied_cov=config["inference"]["tied_cov"], # False
        use_trainable_cov=config["inference"]["trainable_cov"], # True
        sigma=config["inference"]["fixed_sigma"], # 0.02
        max_scale=config["inference"].get("max_scale", 1.0), # 2
        scale_nonlinearity=config["inference"].get("scale_nonlinearity", "softplus"),
    )

    posterior_network = RNNInferenceNetwork(
        posterior_rnn=posterior_rnn,
        posterior_dist=posterior_dist,
        x_dim=config["x_dim"],
        embedding_network=embedding_network,
        takes_ctrl=config["control"]["inference"], # False
        n_obj=config["n_obj"], # 3
    )

    rawctrl2feat_network = None
    if config["control"]["has_ctrl"]:
        assert ctrl_feat_network is not None
        rawctrl2feat_network = RawControlToFeat(
            ctrl_feat_network,
            n_staticfeat=config["control"]["n_staticfeat"],
            n_timefeat=config["control"]["n_timefeat"],
            embedding_dim=config["control"]["emb_dim"],
        )

    if config["model"] == "REDSDS":
        assert duration_network is not None
        d_min = config.get("d_min", 1)
        ctrl2nstf_network = ControlToNSTF(
            duration_network, config["num_categories"], config["d_max"], d_min=d_min
        )

    if config["model"] == "SNLDS":
        model = SNLDS(
            x_init=x_init,
            continuous_transition_network=x_transition,
            z_init=z_init,
            discrete_transition_network=z_transition,
            emission_network=emission_network,
            inference_network=posterior_network,
            ctrl_transformer=rawctrl2feat_network,
            continuous_state_dim=config["x_dim"],
            num_categories=config["num_categories"],
            context_length=config["context_length"], # 100
            prediction_length=config["prediction_length"], # 0
            discrete_state_prior=None,
            transform_target=config["transform_target"], # False
            transform_only_scale=config["transform_only_scale"], # False
            use_jacobian=config["use_jacobian"], # Flase
            no_interaction=config["discrete_transition"]["no_interaction"],
            interaction_simple=config["discrete_transition"]["interaction_simple"],
            interaction_gnn=config["discrete_transition"]["interaction_gnn"],
            interaction_gnn_fully_connected=config["discrete_transition"]["interaction_gnn_fully_connected"],
            interaction_gnn_est_edge=config["discrete_transition"]["interaction_gnn_est_edge"],
            interaction_gnn_gt_edge=config["discrete_transition"]["interaction_gnn_gt_edge"],
        )
    elif config["model"] == "REDSDS":
        model = REDSDS(
            x_init=x_init,
            continuous_transition_network=x_transition,
            z_init=z_init,
            discrete_transition_network=z_transition,
            emission_network=emission_network,
            inference_network=posterior_network,
            ctrl_transformer=rawctrl2feat_network,
            ctrl2nstf_network=ctrl2nstf_network,
            continuous_state_dim=config["x_dim"],
            num_categories=config["num_categories"],
            d_max=config["d_max"],
            context_length=config["context_length"],
            prediction_length=config["prediction_length"],
            discrete_state_prior=None,
            transform_target=config["transform_target"],
            transform_only_scale=config["transform_only_scale"],
            use_jacobian=config["use_jacobian"],
        )
    return model


def get_network_instances(config):
    if (
        config["experiment"] == "bouncing_ball"
        or config["experiment"] == "3modesystem"
        or config["experiment"] == "bee"
        or config["experiment"] == "balls"
        or config["experiment"] == "gts_univariate"
    ):
        # Inference Network
        inf_ctrl_dim = (
            config["control"]["feat_dim"] if config["control"]["inference"] else 0
        ) # 0
        inf_out_dim = config["x_dim"] # 8
        if not config["inference"]["tied_cov"]:
            inf_out_dim *= 2 # 16
        if config["inference"]["embedder"] == "brnn": # brnn
            embedding_network = nn.Sequential(
                nn.GRU(
                    input_size=config["obs_dim"], # 2
                    hidden_size=config["inference"]["embedding_rnndim"], # 16
                    num_layers=config["inference"]["embedding_rnnlayers"], # 1
                    batch_first=True,
                    bidirectional=True,
                ),
                SelectIndex(index=0),
            )
            # 2 * because bidirectional
            emb_out_dim = 2 * config["inference"]["embedding_rnndim"] # 32
        elif config["inference"]["embedder"] == "transformer":
            embedding_network = TransformerEmbedder(
                obs_dim=config["obs_dim"],
                emb_dim=config["inference"]["embedding_trans_embdim"],
                use_pe=config["inference"]["embedding_trans_usepe"],
                nhead=config["inference"]["embedding_trans_nhead"],
                dim_feedforward=config["inference"]["embedding_trans_mlpdim"],
                n_layers=config["inference"]["embedding_trans_nlayers"],
            )
            # 2 * because of positional encoding
            emb_out_dim = 2 * config["inference"]["embedding_trans_embdim"]
        else:
            embedding_network = None
            emb_out_dim = config["obs_dim"]

        posterior_rnn = None
        if config["inference"]["use_causal_rnn"]: # True
            posterior_rnn = nn.RNNCell(
                emb_out_dim + config["x_dim"] + inf_ctrl_dim, # 32 + 8 + 0
                config["inference"]["causal_rnndim"], # 16
            )

        posterior_mlp_in_dim = (
            config["inference"]["causal_rnndim"] # 16
            if config["inference"]["use_causal_rnn"]
            else (emb_out_dim + inf_ctrl_dim)
        )
        posterior_mlp = nn.Sequential(
            nn.Linear(posterior_mlp_in_dim, config["inference"]["mlp_hiddendim"]), # 32
            nn.ReLU(True),
            nn.Linear(config["inference"]["mlp_hiddendim"], inf_out_dim),
        )

        # Emission Network
        if config["emission"]["model_type"] == "linear":
            emission_network = nn.Sequential(
                nn.Linear(config["x_dim"], config["obs_dim"], False)
            )
        else:
            if config["dataset"] == "bee":
                emission_network = nn.Sequential(
                    nn.Linear(config["x_dim"], 32),
                    nn.ReLU(True),
                    nn.Linear(32, 64),
                    nn.ReLU(True),
                    nn.Linear(64, config["obs_dim"]),
                )
            else:
                emission_network = nn.Sequential(
                    nn.Linear(config["x_dim"], 8),
                    nn.ReLU(True),
                    nn.Linear(8, 32),
                    nn.ReLU(True),
                    nn.Linear(32, config["obs_dim"]),
                )

        # Initial Continuous State
        if config["control"]["has_ctrl"] and config["control"]["x"]: # False, False
            x0_networks = [
                nn.Sequential(
                    nn.Linear(
                        config["control"]["feat_dim"],
                        config["initial_state"]["mlp_hiddendim"],
                    ),
                    nn.ReLU(True),
                    nn.Linear(
                        config["initial_state"]["mlp_hiddendim"], config["x_dim"]
                    ),
                )
                for _ in range(config["num_categories"])
            ]
        else:
            x0_networks = [
                nn.Sequential(nn.Linear(1, config["x_dim"], bias=False))
                for _ in range(config["num_categories"])
            ]

        # Continuous Transition
        x_ctrl_dim = config["control"]["feat_dim"] if config["control"]["x"] else 0
        x_trans_out_dim = config["x_dim"] # 8
        if not config["continuous_transition"]["tied_cov"]:
            x_trans_out_dim *= 2 # 16
        if config["continuous_transition"]["model_type"] == "linear":
            x_transition_networks = [
                nn.Sequential(
                    nn.Linear(config["x_dim"] + x_ctrl_dim, x_trans_out_dim, bias=False)
                )
                for _ in range(config["num_categories"])
            ]
        else:
            x_transition_networks = [
                nn.Sequential(
                    nn.Linear(
                        config["x_dim"] + x_ctrl_dim,
                        config["continuous_transition"]["mlp_hiddendim"],
                    ),
                    nn.ReLU(True),
                    nn.Linear(
                        config["continuous_transition"]["mlp_hiddendim"],
                        x_trans_out_dim,
                    ),
                )
                for _ in range(config["num_categories"])
            ]

        # Initial Discrete State
        if config["discrete_transition"]["no_interaction"]:
            z0_network_out_incre = 1
        else:
            z0_network_out_incre = config["n_obj"]
        if config["control"]["has_ctrl"] and config["control"]["z"]: # False False
            z0_network = nn.Sequential(
                nn.Linear(
                    config["control"]["feat_dim"],
                    config["initial_switch"]["mlp_hiddendim"],
                ),
                nn.ReLU(True),
                nn.Linear(
                    config["initial_switch"]["mlp_hiddendim"], config["num_categories"]*z0_network_out_incre
                ),
            )
        else:
            z0_network = nn.Linear(1*z0_network_out_incre, config["num_categories"]*z0_network_out_incre, bias=False)

        # Discrete Transition
        if config["discrete_transition"]["no_interaction"]:
            n_added_objs = 0
            z_tran_out_dim = config["num_categories"]*config["num_categories"]
        elif config["discrete_transition"]["interaction_simple"]:
            n_added_objs = config["n_obj"]
            z_tran_out_dim = config["n_obj"]*config["num_categories"]*config["num_categories"]
        elif config["discrete_transition"]["interaction_gnn"]: 
            n_added_objs = config["n_obj"]
            z_tran_out_dim = config["num_categories"]*config["num_categories"]
            
        discrete_in_dim = (
            config["x_dim"]*(1+n_added_objs) if config["discrete_transition"]["takes_x"] else 0
        )
        discrete_in_dim += (
            config["obs_dim"]*(1+n_added_objs) if config["discrete_transition"]["takes_y"] else 0
        )
        discrete_in_dim += (
            config["inference"]["causal_rnndim"]*(1+n_added_objs) if config["discrete_transition"]["takes_hidden_states"] else 0
        )
        discrete_in_dim += (
            config["num_categories"]*(1+n_added_objs) if config["discrete_transition"]["interaction_gnn"] else 0
        )
        z_ctrl_dim = config["control"]["feat_dim"] if config["control"]["z"] else 0
        if discrete_in_dim + z_ctrl_dim > 0:
            z_transition_network = nn.Sequential(
                nn.Linear(
                    discrete_in_dim + z_ctrl_dim, 4 * z_tran_out_dim
                ),
                nn.ReLU(True),
                nn.Linear(
                    4 * z_tran_out_dim, z_tran_out_dim
                ),
            )
        else:
            z_transition_network = nn.Linear(
                1, z_tran_out_dim, bias=False
            )
            print("[*] No recurrence!")

        # Control Transformer
        ctrl_feat_network = None
        if config["control"]["has_ctrl"]:
            n_input_feats = (
                config["control"]["emb_dim"] + config["control"]["n_timefeat"]
            )
            ctrl_feat_network = nn.Sequential(
                nn.Linear(n_input_feats, config["control"]["mlp_hiddendim"]),
                nn.ReLU(True),
                nn.Linear(
                    config["control"]["mlp_hiddendim"], config["control"]["feat_dim"]
                ),
            )

        # Control to Duration (NSTF)
        duration_network = None
        if "d_max" in config: # 50
            if config["control"]["has_ctrl"]:
                duration_network = nn.Sequential(
                    nn.Linear(config["control"]["feat_dim"], 64),
                    nn.ReLU(True),
                    nn.Linear(64, config["num_categories"] * config["d_max"]),
                )
            else:
                duration_network = nn.Linear(
                    1, config["num_categories"] * config["d_max"], bias=False
                )
    else:
        raise ValueError(f"Unknown experiment: {config['experiment']}!")
    return (
        emission_network,
        posterior_rnn,
        posterior_mlp,
        x0_networks,
        x_transition_networks,
        z0_network,
        z_transition_network,
        embedding_network,
        ctrl_feat_network,
        duration_network,
    )

class RefNRIMLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0., no_bn=False):
        super(RefNRIMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_in, n_hid),
            nn.ELU(inplace=True),
            nn.Dropout(do_prob),
            nn.Linear(n_hid, n_out),
            nn.ELU(inplace=True)
        )
        if no_bn:
            self.bn = None
        else:
            self.bn = nn.BatchNorm1d(n_out)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            # if isinstance(m, nn.Linear):
            #     nn.init.xavier_normal_(m.weight.data)
            #     m.bias.data.fill_(0.1)
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        orig_shape = inputs.shape
        x = inputs.view(-1, inputs.size(-1))
        x = self.bn(x)
        return x.view(orig_shape)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = self.model(inputs)
        if self.bn is not None:
            return self.batch_norm(x)
        else:
            return x

import numpy as np
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def sample_gumbel(shape, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Sample from Gumbel(0, 1)
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(shape).float()
    return - torch.log(eps - torch.log(U + eps))

def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Draw a sample from the Gumbel-Softmax distribution
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + gumbel_noise
    return F.softmax(y / tau, dim=-1)

def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes
    Constraints:
    - this implementation only works on batch_size x num_features tensor for now
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = y_hard - y_soft.data + y_soft
    else:
        y = y_soft
    return y