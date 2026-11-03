"""
Models used for forward-backward RL algorithms.
"""

import logging

import torch
import torch.nn as nn
import coloredlogs

logger = logging.getLogger(__name__)
coloredlogs.install(level="DEBUG", logger=logger)


class IdentityFeaturesExtractor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.output_dim = input_dim

    def forward(self, x):
        return x


class CNNFeaturesExtractor(nn.Module):
    """
    used for a 64x64x3 image input
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        obs_shape = (input_dim, 64, 64)
        with torch.inference_mode():
            conv_output_dim = self.cnn(torch.zeros(1, *obs_shape)).shape[1]
        self.linear = nn.Sequential(nn.Linear(conv_output_dim, output_dim), nn.ReLU())
        self.output_dim = output_dim

    def forward(self, x):
        x = self.cnn(x)
        x = self.linear(x)
        return x


class MLPFeaturesExtractor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 256]):
        super().__init__()
        hidden_layers = nn.ModuleList()
        hidden_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        hidden_layers.append(nn.Mish())
        for i in range(len(hidden_dims) - 1):
            hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            hidden_layers.append(nn.Mish())
        hidden_layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.hidden_block = nn.Sequential(*hidden_layers)
        self.output_dim = output_dim

    def forward(self, x):
        return self.hidden_block(x)


class RadialBasisXYFeaturesExtractor(nn.Module):
    def __init__(self, input_dim, low, high, dim, sigma):
        assert input_dim == 2, "RadialBasisXYFeaturesExtractor only supports 2D input."
        super().__init__()
        self.dim = dim
        self.sigma = sigma
        self.output_dim = dim * dim

        xlist = torch.linspace(low, high, dim)
        ylist = torch.linspace(low, high, dim)
        # clone: meshgrid returns expanded views, which load_state_dict cannot copy into
        XX, YY = torch.meshgrid(xlist, ylist, indexing="ij")
        XX, YY = XX.clone(), YY.clone()
        x_mu = XX.flatten()
        y_mu = YY.flatten()

        self.register_buffer("XX", XX)
        self.register_buffer("YY", YY)
        self.register_buffer("x_mu", x_mu)
        self.register_buffer("y_mu", y_mu)

    def forward(self, x):
        return self.transform(x)

    def transform(self, x):
        distance = (x[:, 0].reshape(-1, 1) - self.x_mu[None]) ** 2 + (x[:, 1].reshape(-1, 1) - self.y_mu[None]) ** 2
        weights = torch.exp(-distance / (2 * (self.sigma**2)))
        return weights / weights.sum(axis=1, keepdims=True)

    def inverse_transform(self, A):
        index = torch.argmax(A, axis=1)
        result = []
        for idx in index:
            i, j = self._1d_index_to_2d_index(idx)
            result.append([self.XX[i][j], self.YY[i][j]])

        return torch.tensor(result)

    def _1d_index_to_2d_index(self, index):
        i = index // self.dim
        j = index % self.dim

        return i, j


class CubeGoalFeaturesExtractor(nn.Module):
    """
    Extracts x,y,z coordinates of the cube from the observations and processes them with an MLP or RBF layer.
    This is mostly used for the backwardnet of the FBRL algorithm, in order to not get confused
    by the other information in the observations and only focus on the cube coordinates.
    Indexes 19, 20, and 21 refer to the x,y,z coordinates of the cube (single) in the OGBench Cube environment.
    """

    def __init__(self, features_extractor):
        super().__init__()
        self.features_extractor = features_extractor
        self.output_dim = features_extractor.output_dim

    def forward(self, obs):
        # Extract x,y,z coordinates (first 3 dimensions of the observation)
        xyz = obs[:, 19:22]
        return self.features_extractor(xyz)


class CubeDoubleGoalFeaturesExtractor(nn.Module):
    """
    Extracts x,y,z coordinates of the cube from the observations and processes them with an MLP or RBF layer.
    This is mostly used for the backwardnet of the FBRL algorithm, in order to not get confused
    by the other information in the observations and only focus on the cube coordinates.
    Indexes 19, 20, and 21 refer to the x,y,z coordinates of the first cube in the OGBench CubeDouble environment.
    Indexes 28, 29, and 30 refer to the x,y,z coordinates of the second cube in the OGBench CubeDouble environment.
    """

    def __init__(self, features_extractor):
        super().__init__()
        self.features_extractor = features_extractor
        self.output_dim = features_extractor.output_dim

    def forward(self, obs):
        # Extract x,y,z coordinates (first 3 dimensions of the observation)
        xyz_1 = obs[:, 19:22]
        xyz_2 = obs[:, 28:31]
        xyzs = torch.cat([xyz_1, xyz_2], dim=-1)
        return self.features_extractor(xyzs)


class SceneGoalFeaturesExtractor(nn.Module):
    """
    Extract the positions (and state for buttons) of cube, buttons, drawer and window from the observation.
    """

    def __init__(self, features_extractor):
        super().__init__()
        self.features_extractor = features_extractor
        self.output_dim = features_extractor.output_dim

    def forward(self, obs):
        block_pos = obs[:, 19:22]
        button1_state = obs[:, 28:30]  # one-hot encoding of button 1 state
        button2_state = obs[:, 32:34]  # one-hot encoding of button 2 state
        drawer_pos = obs[:, 36:37]
        window_pos = obs[:, 38:39]
        goal_obs = torch.cat(
            [block_pos, button1_state, button2_state, drawer_pos, window_pos],
            dim=-1,
        )
        return self.features_extractor(goal_obs)


class EnsembleForwardNet(nn.Module):
    """
    Ensemble of ForwardNets for continuous state and action spaces. F(s,a,z)
    """

    def __init__(
        self,
        features_extractor,
        action_dim,
        z_dim,
        hidden_dims,
        ensemble_size=2,
    ):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.forward_nets = nn.ModuleList()
        for _ in range(ensemble_size):
            forward_net = ForwardNet(
                features_extractor,
                action_dim,
                z_dim,
                hidden_dims,
            )
            self.forward_nets.append(forward_net)

    def forward(self, obs, action, z):
        outputs = []
        for net in self.forward_nets:
            output = net(obs, action, z)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=0)  # (ensemble_size, batch_size, z_dim)
        return outputs


class ForwardNet(nn.Module):
    """
    ForwardNet for continuous state and action spaces. F(s,a,z)
    """

    def __init__(
        self,
        features_extractor,
        action_dim,
        z_dim,
        hidden_dims,
    ):
        """
        Args:
            features_extractor: Feature extractor to process the observations.
            action_dim: Dimension of the action space.
            z_dim: Dimension of the latent variable z.
            hidden_dims: List of hidden layer dimensions.
            activation: Activation function to use.
        """
        super().__init__()
        self.features_extractor = features_extractor
        self.features_dim = self.features_extractor.output_dim
        self.s_z_block = nn.Sequential(
            nn.Linear(self.features_dim + z_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.Mish(),  # Tanh(),
            nn.Linear(hidden_dims[0], hidden_dims[0] // 2),
            nn.Mish(),
        )
        self.s_a_block = nn.Sequential(
            nn.Linear(self.features_dim + action_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.Mish(),  # Tanh(),
            nn.Linear(hidden_dims[0], hidden_dims[0] // 2),
            nn.Mish(),
        )
        hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            hidden_layers.append(nn.Mish())
        hidden_layers.append(nn.Linear(hidden_dims[-1], z_dim))
        self.hidden_block = nn.Sequential(*hidden_layers)

    def forward(self, obs, action, z):
        s = self.features_extractor(obs)
        s_z_embed = self.s_z_block(torch.cat([s, z], dim=-1))
        s_a_embed = self.s_a_block(torch.cat([s, action], dim=-1))
        x = torch.cat([s_z_embed, s_a_embed], dim=-1)
        x = self.hidden_block(x)
        return x


class BackwardNet(nn.Module):
    """
    BackwardNet for continuous state and action spaces. B(s)
    """

    def __init__(self, features_extractor, z_dim, hidden_dims):
        super().__init__()
        self.features_extractor = features_extractor
        self.features_dim = self.features_extractor.output_dim
        self.first_block = nn.Sequential(
            nn.Linear(self.features_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.Mish(),  # Tanh(),
        )
        hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            hidden_layers.append(nn.Mish())
        hidden_layers.append(nn.Linear(hidden_dims[-1], z_dim))
        self.hidden_block = nn.Sequential(*hidden_layers)

    def forward(self, obs):
        s = self.features_extractor(obs)
        x = self.first_block(s)
        x = self.hidden_block(x)
        return x


class StochasticActor(nn.Module):
    LOG_STD_MIN = -1
    LOG_STD_MAX = 2

    def __init__(
        self,
        features_extractor,
        z_dim,
        action_dim,
        hidden_dims,
        action_lower_bound=-1.0,
        action_upper_bound=1.0,
    ):
        super().__init__()
        self.features_extractor = features_extractor
        self.features_dim = self.features_extractor.output_dim
        self.s_z_block = nn.Sequential(
            nn.Linear(self.features_dim + z_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.Mish(),  # Tanh(),
            nn.Linear(hidden_dims[0], hidden_dims[0] // 2),
            nn.Mish(),
        )

        self.s_block = nn.Sequential(
            nn.Linear(self.features_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.Mish(),  # Tanh(),
            nn.Linear(hidden_dims[0], hidden_dims[0] // 2),
            nn.Mish(),
        )

        hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            hidden_layers.append(nn.Mish())
        self.hidden_block = nn.Sequential(*hidden_layers)

        self.fc_mean = nn.Linear(hidden_dims[-1], action_dim)
        self.fc_logstd = nn.Linear(hidden_dims[-1], action_dim)

        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (action_upper_bound - action_lower_bound) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (action_upper_bound + action_lower_bound) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, obs, z):
        s = self.features_extractor(obs)
        s_z_embed = self.s_z_block(torch.cat([s, z], dim=1))
        s_embed = self.s_block(s)
        x = torch.cat([s_z_embed, s_embed], dim=1)
        x = self.hidden_block(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, obs, z):
        mean, log_std = self(obs, z)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def predict(self, obs, z):
        """
        Predict the action given the observation and latent variable z.
        Used for evaluation.
        """
        mean, _ = self(obs, z)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return mean
