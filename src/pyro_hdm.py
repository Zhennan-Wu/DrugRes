import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import constraints
from torch.func import vmap
import pyro
import pyro.distributions as dist
import numpy as np
from itertools import product


def all_same_length(*lists):
    lengths = list(map(len, lists))
    return all(length == lengths[0] for length in lengths)


def mix_weights(beta):
    beta1m_cumprod = (1 - beta).cumprod(-1)
    return F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)


def is_valid_probability_matrix(matrix, tol=1e-5):
    values_in_range = (matrix >= 0).all() and (matrix <= 1).all()
    row_sums_close_to_1 = torch.allclose(matrix.sum(dim=1), torch.ones(matrix.size(0)), atol=tol)
    return values_in_range and row_sums_close_to_1


class HDM:

    def __init__(self, gamma, H_c, H_r, mixture_concentrations, cat_concentrations, mixture_truncate_upper_bound, cat_truncate_upper_bounds, device="cpu"):
        '''
        Parameters:
        ----------
        gamma: float
            concentration parameter for the mixture component
        H_c: torch.distributions
            prior distribution for the classification mixture component
        H_r: torch.distributions
            prior distribution for the regression mixture component
        mixture_concentrations: list of float
            concentration parameters for each level of the HDP
        cat_concentrations: list of float
            concentration parameters for each level of the nCRP
        mixture_truncate_upper_bound: int
            upper bound for the number of mixture components
        cat_truncate_upper_bounds: list of int
            upper bounds for the number of categories at each level of the nCRP
        device: torch.device, optional
            device to use for the model (default is cuda if available, otherwise cpu)
        -------
        Products:
        -------
        self.gamma: float
            concentration parameter for the mixture component
        self.alphas: list of float
            concentration parameters for each level of the HDP
        self.etas: list of float
            concentration parameters for each level of the nCRP
        self.K: int
            upper bound for the number of mixture components
        self.Cs: list of int
            upper bounds for the number of categories at each level of the nCRP
        self.depth: int
            depth of the HDP
        self.device: torch.device
            device to use for the model
        self.mixture_components_c: torch.Tensor
            mixture components for classification
        self.mixture_components_r: torch.Tensor
            mixture components for regression
        -------
        Returns:
        -------
        None
        -------
        '''
        if device == None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if not all_same_length(mixture_concentrations, cat_concentrations, cat_truncate_upper_bounds):
            raise ValueError("Hierarchical structure parameters do not match")
        self.gamma = gamma
        self.alphas = mixture_concentrations
        self.etas = cat_concentrations
        self.K = mixture_truncate_upper_bound
        self.Cs = cat_truncate_upper_bounds
        self.depth = len(mixture_concentrations)
        self.generate_mixture_components(H_c, H_r)

    def _gen_cluster_name(self, level, keyword):
        '''
        Generate cluster names for the nCRP and HDP processes.
        Parameters:
        ----------
        level: int
            level of the HDP or nCRP process
        keyword: str
            keyword to use for the cluster names (e.g. 'C' for nCRP, 'G' for HDP)
        -------
        Products:
        -------
        None
        -------
        Returns:
        -------
        list of str
            list of cluster names for the given level and keyword
        '''
        if level < 0 or level > self.depth:
            raise ValueError("Level out of range")
        if keyword not in ['C', 'G']:
            raise ValueError("Keyword must be 'C' or 'G'")
        levels = [list(range(1, self.Cs[l]+1)) for l in range(0, level)]
        result = [keyword + ''.join(map(str, items)) for items in product(*levels)]
        return result

    def generate_mixture_components(self, H_c, H_r):
        '''
        Generate the mixture components for the regression and classification tasks.
        Parameters:
        ----------
        H_c: torch.distributions
            prior distribution for the classification mixture component
        H_r: torch.distributions
            prior distribution for the regression mixture component
        Products:
        -------
        self.mixture_components_c: torch.Tensor
            mixture components for classification
        self.mixture_components_r: torch.Tensor
            mixture components for regression
        -------
        Returns:
        -------
        None
        -------
        '''
        if not isinstance(H_c, torch.distributions.Distribution) or not isinstance(H_r, torch.distributions.Distribution):
            raise ValueError("H_c and H_r must be torch.distributions.Distribution objects")
        self.mixture_components_c = pyro.param(f"mixture_c", H_c.sample([self.K])).to(self.device)
        self.mixture_components_r = pyro.param(f"mixture_r", H_r.sample([self.K])).to(self.device)
        
    def generate_nCRP(self):
        '''
        Generate the nCRP processes.
        Parameters:
        ----------
        None
        -------
        Products:
        -------
        self.CRPs: dict
            dictionary of nCRP processes
                key: str, category name
                value: torch.tensor(num_child_cat), child process weights of the category
        self.level_CRPs: dict
            dictionary of nCRP processes at each level
                key: str, level name
                value: torch.tensor(num_root_cat, ..., num_p_cat, num_c_cat), child process weights of the level
        -------
        Returns:
        -------
        None
        -------
        '''
        self.CRPs = {}
        self.level_CRPs = {}
        level_cats = list(np.cumprod(np.array(self.Cs)))
        level_cats.insert(0, 1)

        for level in range(self.depth):
            total_cat = level_cats[level]
            eta = self.etas[level]
            C = self.Cs[level]
            with pyro.plate(f"nCRP_{level}", total_cat):
                cat_dist = pyro.sample(f"CRP_{level+1}", dist.Beta(torch.ones(C-1, device=self.device), eta*torch.ones(C-1, device=self.device)).expand([total_cat, C-1]).to_event(1))
            cat_names = self._gen_cluster_name(level, 'C')
            child_dist = vmap(mix_weights)(cat_dist)
            self.CRPs.update(dict(zip(cat_names, child_dist)))
            # print(f"[MODEL] Level {level} CRP_{level+1} shape:", child_dist.shape)
            if (level == 0):
                self.level_CRPs[f"L{level}"] = child_dist.reshape([1, self.Cs[0]])
            else:
                self.level_CRPs[f"L{level}"] = child_dist.reshape(self.Cs[: level+1])

    def approximate_nCRP(self):
        '''
        Generate the nCRP processes.
        Parameters:
        ----------
        None
        -------
        Products:
        -------
        self.CRPs: dict
            dictionary of nCRP processes
                key: str, category name
                value: torch.tensor(num_child_cat), child process weights of the category
        self.level_CRPs: dict
            dictionary of nCRP processes at each level
                key: str, level name
                value: torch.tensor(num_root_cat, ..., num_p_cat, num_c_cat), child process weights of the level
        -------
        Returns:
        -------
        None
        -------
        '''
        self.approx_CRPs = {}
        self.approx_CRP_params = {}
        level_cats = list(np.cumprod(np.array(self.Cs)))
        level_cats.insert(0, 1)
    
        for level in range(self.depth):
            total_cat = level_cats[level]
            C = self.Cs[level]
            with pyro.plate(f"nCRP_{level}", total_cat):
                params = pyro.param(f"zeta_{level}", lambda: dist.Uniform(0, 2).sample([total_cat, C-1]), constraint=constraints.positive).to(self.device)
                child_dist = pyro.sample(f"CRP_{level+1}", dist.Beta(torch.ones([total_cat, C-1], device=self.device), params).to_event(1))
            cat_names = self._gen_cluster_name(level, 'C')
            self.approx_CRPs.update(dict(zip(cat_names, child_dist)))
            self.approx_CRP_params.update(dict(zip(cat_names, params)))

    def generate_HDP(self):
        '''
        Generate the HDP processes.
        Parameters:
        ----------
        None
        -------
        Products:
        -------
        self.Gs: dict
            dictionary of HDP processes
                key: str, category name
                value: torch.tensor(K), mixture component weights of the category
        self.level_Gs: dict
            dictionary of HDP processes at each level
                key: str, level name
                value: torch.tensor(num_root_cat, ..., num_p_cat, num_c_cat, K), mixture component weights of the level
        -------
        Returns:
        -------
        None
        -------
        '''
        self.Gs = {}
        self.level_Gs = {}
        beta = pyro.param("beta", lambda: dist.Beta(1, self.gamma).sample([self.K-1])).to(self.device)
        self.Gs['G'] = mix_weights(beta)
        for level in range(self.depth):
            alpha =  self.alphas[level]
            process_names = self._gen_cluster_name(level+1, 'G')
            level_G = []
            for child_p in process_names:
                parent_p = child_p[:-1]
                self.Gs[child_p] = pyro.sample(child_p, dist.Dirichlet(alpha*self.Gs[parent_p])).to(self.device)
                level_G.append(self.Gs[child_p])
            self.level_Gs[f"L{level}"] = torch.stack(level_G).reshape(self.Cs[: level+1] + [self.K])

    def approximate_HDP(self):
        '''
        Generate the HDP processes.
        Parameters:
        ----------
        None
        -------
        Products:
        -------
        self.Gs: dict
            dictionary of HDP processes
                key: str, category name
                value: torch.tensor(K), mixture component weights of the category
        self.level_Gs: dict
            dictionary of HDP processes at each level
                key: str, level name
                value: torch.tensor(num_root_cat, ..., num_p_cat, num_c_cat, K), mixture component weights of the level
        -------
        Returns:
        -------
        None
        -------
        '''
        self.approx_Gs = {}
        self.approx_G_params = {}
        
        for level in range(self.depth):
            process_names = self._gen_cluster_name(level+1, 'G')
            for child_p in process_names:
                self.approx_G_params[child_p] = pyro.param(f"chi_{child_p}", lambda: dist.Uniform(0, 2).sample([self.K]), constraint=constraints.positive).to(self.device)
                self.approx_Gs[child_p] = pyro.sample(child_p, dist.Dirichlet(self.approx_G_params[child_p])).to(self.device)

    def init_regressor(self, data):
        self.regressor = nn.Sequential(
            nn.Linear(data.shape[1], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def model(self, data, label, dataset_size, temperature=0.5):
        '''
        Parameters:
        ----------
        data: torch.Tensor
            input data for the model
        label: torch.Tensor
            labels for the input data
        temperature: float
            temperature parameter for the relaxed one-hot categorical distribution
        -------
        Products:
        -------
        None
        -------
        Returns:
        -------
        None
        -------
        '''
        # self.init_regressor(data)
        # pyro.module("regressor", self.regressor)
        # self.regressor.to(self.device)
        data = data.to(self.device)
        label = label.to(self.device)
        temperature = torch.tensor(temperature).to(self.device)

        self.generate_nCRP()
        self.generate_HDP()

        N = data.shape[0]
        scale = dataset_size / N
        latent_dim_c = self.mixture_components_c.shape[1]
        with pyro.poutine.scale(scale=scale):
            with pyro.plate("data", N):
                cat_asignments = []
                cat_asignments.append(torch.ones((N, 1), device=self.device))
                for l in range(self.depth):
                    probs = self.level_CRPs[f"L{l}"].to(self.device).unsqueeze(0)
                    z = cat_asignments[-1].to(self.device).reshape([N]+self.Cs[:l]).unsqueeze(-1)
                    sample_probs = torch.mul(z, probs).reshape(N, -1)
                    sample_probs = sample_probs / sample_probs.sum(dim=-1, keepdim=True)
                    cat_asignments.append(pyro.sample(f"z_{l}", dist.RelaxedOneHotCategorical(temperature, probs=sample_probs)).to(self.device))
                final_z = cat_asignments[-1].reshape([N]+self.Cs).to(self.device).unsqueeze(-1)
                mix_probs = self.level_Gs[f"L{self.depth-1}"].to(self.device).unsqueeze(0)
                mixture_weights = torch.mul(final_z, mix_probs)
                mixture_weights = mixture_weights.sum(dim=tuple(range(1, mixture_weights.ndim - 1)))
                mixture_weights = mixture_weights / mixture_weights.sum(dim=-1, keepdim=True)
                h = pyro.sample(f"latent", dist.MultivariateNormal(torch.matmul(mixture_weights, self.mixture_components_c), torch.eye(latent_dim_c)), obs=data)
                # h = pyro.sample(f"latent", dist.Dirichlet(torch.matmul(mixture_weights, self.mixture_components_c)), obs=data)
                # x = torch.stack(cat_asignments)
                # x = torch.cat([x, h], dim=-1)
                # mu, sigma = self.regressor(x)
                # pyro.sample("y", dist.Normal(mu, sigma), obs=label)

    def guide(self, data, label, dataset_size, temperature=0.5):
        N = data.shape[0]
        temperature = torch.tensor(temperature).to(self.device)
        self.approximate_nCRP()
        self.approximate_HDP()

        data_cats = []
        data_cats_params = []
        level_cats = list(np.cumprod(np.array(self.Cs)))
        with pyro.plate("data", N):
            cat_asignments = []
            params = []
            for l in range(self.depth):
                C = level_cats[l]
                params.append(pyro.param(f'kappa_{l}', dist.Dirichlet(1/C * torch.ones(C)).sample(), constraint=constraints.simplex).to(self.device))
                cat_asignments.append(pyro.sample(f"z_{l}", dist.RelaxedOneHotCategorical(temperature, probs=params[-1])))
            data_cats.append(cat_asignments)
            data_cats_params.append(params)
            
            





            


