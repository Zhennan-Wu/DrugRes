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


class HDM:

    def __init__(self, gamma, H, mixture_concentrations, cat_concentrations, mixture_truncate_upper_bound, cat_truncate_upper_bounds, device=None):
        if device == None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if not all_same_length(mixture_concentrations, cat_concentrations, cat_truncate_upper_bounds):
            raise ValueError("HDP parameters do not match")
        self.gamma = gamma
        self.alphas = mixture_concentrations
        self.etas = cat_concentrations
        self.K = mixture_truncate_upper_bound
        self.Cs = cat_truncate_upper_bounds
        self.depth = len(mixture_concentrations)
        self.generate_nCRP
        self.generate_HDP
        self.generate_mixture_components(H)

    def _gen_cluster_name(self, level, keyword):
        levels = [list(range(1, self.Cs[l]+1)) for l in range(0, level)]
        result = [keyword + ''.join(map(str, items)) for items in product(*levels)]
        return result

    def generate_mixture_components(self, H):
        self.mixture_components = pyro.param(f"mixture", H.sample([self.K]))
        
    def generate_nCRP(self):
        self.CRPs = {}
        self.level_CRPs = {}
        level_cats = list(np.cumprod(np.array(self.Cs)))
        level_cats.insert(0, 1)
    
        for level in range(self.depth):
            total_cat = level_cats[level]
            eta = self.etas[level]
            C = self.Cs[level]

            cat_dist = pyro.param(f"CRP_{level+1}", dist.Beta(1, eta).sample([total_cat, C-1]))
            cat_names = self._gen_cluster_name(level, 'C')
            child_dist = vmap(mix_weights, cat_dist)
            self.CRPs.update(dict(zip(cat_names, child_dist)))
            self.level_CRPs[f"L{level}": child_dist.reshape(self.Cs[: level])]

    def generate_HDP(self):
        self.Gs = {}
        self.level_Gs = {}
        beta = pyro.param("beta", dist.Beta(1, self.gamma).sample([self.K-1]))
        self.Gs['G'] = mix_weights(beta)
        for level in range(self.depth):
            alpha =  self.alphas[level]
            process_names = self._gen_cluster_name(level, 'G')
            level_Gs = []
            for child_p in process_names:
                parent_p = child_p[:-1]
                self.Gs[child_p] = pyro.param(child_p, dist.Dirichlet(alpha*self.Gs[parent_p]).sample())
                level_Gs.append(self.Gs[child_p])
            self.level_Gs[f"L{level}"] = torch.stack(level_Gs).reshape(self.Cs[: level])

    def regression(self, data):
        self.regressor = nn.Sequential(
            nn.Linear(data.shape[1], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def model(self, data, label, temperature=0.5):
        pyro.module("regressor", self.regressor)
        self.regressor.to(self.device)
        data = data.to(self.device)
        label = label.to(self.device)

        N = data.shape[0]
        with pyro.plate("data", N):
            cat_asignments = []
            cat_asignments.append(torch.ones(self.Cs[0], device=self.device))
            for l in range(self.depth):
                probs = self.level_CRPs[l]
                z = cat_asignments[-1]
                cat_asignments.append(pyro.sample(f"z_{l}", dist.RelaxedOneHotCategorical(temperature, probs=torch.matmul(z, probs))))
            mixture_weights = torch.matmul(cat_asignments[-1], self.level_Gs[f"L{self.depth-1}"])
            h = pyro.sample(f"latent", dist.Dirichlet(torch.matmul(mixture_weights, self.mixture_components)), obs=data)
            x = torch.stack(cat_asignments)
            x = torch.cat([x, h], dim=-1)
            mu, sigma = self.regressor(x)
            pyro.sample("y", dist.Normal(mu, sigma), obs=label)


    def guide(self, data, label, temperature=0.5):
        N = data.shape[0]
        params = []
        for l in range(self.depth):
            num_cats = len(self._gen_cluster_name(l, 'C'))
            params.append(pyro.param('phi', lambda: dist.Dirichlet(1/num_cats * torch.ones(num_cats)).sample([N]), constraint=constraints.simplex))
        
        with pyro.plate("data", N):
            cat_asignments = []
            for l in range(self.depth):
                cat_asignments.append(pyro.sample(f"z_{l}", dist.RelaxedOneHotCategorical(temperature, probs=params[l])))
            
            





            


