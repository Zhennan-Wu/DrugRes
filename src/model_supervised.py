import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Independent, Normal

def print_versions(h, tag):
    if isinstance(h, (list, tuple)):
        versions = [x._version if isinstance(x, torch.Tensor) else None for x in h]
        shapes = [tuple(x.shape) if isinstance(x, torch.Tensor) else None for x in h]
        print(f"{tag}: versions={versions}, shapes={shapes}")
    else:
        print(f"{tag}: version={h._version}, shape={tuple(h.shape)}")


class DBM(nn.Module):
    def __init__(self, nv, nh=None, ny=1, L=2, nMult=100, y_sigma=1., rho=0.1, known_y=True):
        super().__init__()
        if nh is None:
            nh = [nv] * L
        self.weight = nn.ParameterList([nn.Parameter(torch.Tensor(nh[0], nv))])
        self.weight.extend([nn.Parameter(torch.Tensor(nh[i], nh[i-1])) for i in range(1, L)])
        self.weight.extend([nn.Parameter(torch.Tensor(ny, nh[-1]))])
        self.bias = nn.ParameterList([nn.Parameter(torch.Tensor(nv))])
        self.bias.extend([nn.Parameter(torch.Tensor(nh[i])) for i in range(L)])
        self.bias.extend([nn.Parameter(torch.Tensor(ny))])

        self.nv = nv
        self.nh = nh
        self.ny = ny
        self.L = L
        self.y_sigma = y_sigma
        self.rho = rho
        self.known_y = known_y
        self.reset_parameters()

    def reset_parameters(self):
        for w in self.weight:
            nn.init.orthogonal_(w)

        for b in self.bias:
            nn.init.zeros_(b)

    def forward(self, v, y):
        N = v.size(0)
        device = v.device
        if y is None:
            self.known_y = False
            self.rho = 0.0
            y = torch.zeros(N, self.ny, device=device)

        # Positive phase
        v = v.flatten(1).float()
        if self.ny > 1:
            y = y.flatten(1).float()
        else:
            y = y.float()

        if self.L == 1:
            v, y, h, _ = self.gibbs_step(v, y, None, True, True,
                                    torch.ones(N, device=device), torch.ones(N, device=device))
            energy_pos = self.energy(v, y, h)
        else:
            h = []
            for i in range(self.L):
                h_i = torch.empty(N, self.nh[i], device=device).bernoulli_()
                # print_versions(h_i, "created h")
                h.append(h_i)

            v, y, h = self.local_search(v, y, h, True, True)
            # print_versions(h, "after local search")
            v, y, h, _ = self.gibbs_step(v, y, h, True, True)
            # print_versions(h, "after gibbs step")

            energy_pos, v, y, h = self.coupling(v, y, h, True, True)

        # Negative phase
        # print("Negative phase")
        v = torch.empty_like(v).bernoulli_()
        y = torch.empty_like(y).normal_()

        h = []
        for i in range(self.L):
            h_i = torch.empty(N, self.nh[i], device=device).bernoulli_()
            # print_versions(h_i, "created h (neg phase)")
            h.append(h_i)

        v, y, h = self.local_search(v, y, h)
        # print_versions(h, "after local search (neg phase)")
        v, y, h, _ = self.gibbs_step(v, y, h)
        # print_versions(h, "after gibbs step (neg phase)")

        energy_neg, v, y, h = self.coupling(v, y, h)
        # print_versions(h, "after coupling (neg phase)")

        loss = energy_pos - energy_neg

        return loss

    @torch.no_grad()
    def local_search(self, v, y, h, fix_v=False, fix_y=False):
        N = v.size(0)
        device= v.device

        rand_u = torch.rand(N, device=device)
        _v = v.clone().detach()
        _y = y.clone().detach()
        _h = [hi.clone().detach() for hi in h]
        v, y, h, _ = self.gibbs_step(v, y, h, fix_v, fix_y, rand_u=rand_u, T=0)

        converged = torch.ones(N, dtype=torch.bool, device=device) if fix_v \
                    else torch.all(v == _v, 1)

        converged = converged.logical_and(torch.all(y == _y, 1))

        for i in range(self.L):
            converged = converged.logical_and(torch.all(h[i] == _h[i], 1))
        if not converged.all():
            v = v.clone().detach()
            y = y.clone().detach()
            h = [hi.clone().detach() for hi in h]

        while not converged.all():
            not_converged = converged.logical_not()
            _v = v[not_converged]
            _y = y[not_converged]
            _h = [h[i][not_converged] for i in range(self.L)]
            M = _v.size(0)

            v_, y_, h_, _ = self.gibbs_step(_v, _y, _h, fix_v, fix_y,
                                     rand_u=rand_u[not_converged], T=0)

            if fix_v:
                converged_ = torch.ones(M, dtype=torch.bool, device=device)
            else:
                converged_ = torch.all(v_ == _v, 1)
                v[not_converged] = v_

            if fix_y:
                pass
            else:
                converged_ = converged_.logical_and(torch.all(y_ == _y, 1))
                y[not_converged] = y_


            for i in range(self.L):
                converged_ = converged_.logical_and(torch.all(h_[i] == _h[i], 1))
                h[i][not_converged] = h_[i]

            converged[not_converged] = converged_

        return v, y, h

    def coupling(self, v, y, h, fix_v=False, fix_y=False, max_iter=20):
        N = v.size(0)
        device = v.device
        _v = v.clone().detach()
        _y = y.clone().detach()
        _h = [hi.clone().detach() for hi in h]

        v, y, h = self.mh_step(v, y, h, fix_v, fix_y)
        energy = self.energy(v, y, h)

        converged = torch.ones(N, dtype=torch.bool, device=device) if fix_v \
                    else torch.all(v == _v, 1)
        if fix_y:
            pass
        else:
            converged = converged.logical_and(torch.all(y == _y, 1))

        for i in range(self.L):
            converged = converged.logical_and(torch.all(h[i] == _h[i], 1))
        # print("mh iteration counting")
        iteration = 0
        if not converged.all():
            v = v.clone()
            y = y.clone()
            h = [hi.clone() for hi in h]
        while not converged.all() and iteration < max_iter:
            iteration += 1
            # print(f"  iteration {iteration}, not converged: {converged.logical_not().sum().item()}")
            not_converged = converged.logical_not()

            _v = v[not_converged]
            _y = y[not_converged]
            _h = [h[i][not_converged] for i in range(self.L)]
            M = _v.size(0)

            rand_v = None if fix_v else torch.rand_like(_v)
            rand_y = None if fix_y else torch.randn_like(_y)
            rand_h = [torch.rand_like(_h[i]) for i in range(self.L)]
            rand_u = torch.rand(M, device=device)

            v_, y_, h_ = self.mh_step(_v, _y, _h, fix_v, fix_y, rand_v, rand_y, rand_h, rand_u)
            energy[not_converged] = energy[not_converged] + self.energy(v_, y_, h_) - self.energy(_v, _y, _h)
            with torch.no_grad():
                if fix_v:
                    converged_ = torch.ones(M, dtype=torch.bool, device=device)
                else:
                    converged_ = torch.all(v_ == _v, 1)
                    v[not_converged] = v_

                if fix_y:
                    pass
                else:
                    converged_ = converged_.logical_and(torch.all(y_ == _y, 1))
                    y[not_converged] = y_

                for i in range(self.L):
                    converged_ = converged_.logical_and(torch.all(h_[i] == _h[i], 1))
                    h[i][not_converged] = h_[i]

                converged[not_converged] = converged_
        if iteration == max_iter:
            print("Warning: coupling MH did not converge within max_iter")

        return energy, v, y, h

    def energy(self, v, y, h, show=False):
        energy_gen = - torch.sum(v * self.bias[0].unsqueeze(0), 1)

        for i in range(self.L):
            logits = F.linear(v if i==0 else h[i-1],
                              self.weight[i], self.bias[i+1])

            energy_gen = energy_gen - torch.sum(h[i] * logits, 1)

        energy_reg = torch.sum((y-self.bias[-1].unsqueeze(0))**2, 1) / (2.*(self.y_sigma**2)) 
        logits = F.linear(y/(self.y_sigma**2),
                            self.weight[-1].t(), self.bias[-2])
        energy_reg = energy_reg - torch.sum(h[-1] * logits, 1)

        # for i in range(self.L):
        #     logits = F.linear(y/self.y_sigma**2 if i == 0 else h[-i],
        #                       self.weight[-i-1].t(), self.bias[-i-2])
        #     energy_reg = energy_reg - torch.sum(h[-i-1] * logits, 1)
        if show:     
            print("energy_gen:", energy_gen)
            print("energy_reg:", energy_reg)

        return (1-self.rho)*energy_gen + self.rho*energy_reg

    def marginal_energy(self, v, y):
        N = v.size(0)
        device = v.device

        v = v.flatten(1).float()
        if (self.ny > 1):
            y = y.flatten(1).float()
        else:
            y = y.float()
        h = [torch.empty(N, self.nh[i],
                         device=device).bernoulli_() for i in range(self.L)]

        v, y, h = self.local_search(v, y, h, True, True)
        v_mode, y_mode, h_mode, _ = self.gibbs_step(v, y, h, T=0)
        v_rand, y_rand, h_rand, _ = self.gibbs_step(v, y, h)

        energy = (self.energy(v, y, h_mode) + self.energy(v, y, h_rand)) / 2
        return energy

    @torch.no_grad()
    def gibbs_step(self, v, y, h, fix_v=False, fix_y=False, 
                   rand_v=None, rand_y=None, rand_h=None, rand_u=None, rand_z=None, T=1):
        N = v.size(0)
        device = v.device

        v_ = v.clone().detach()
        y_ = y.clone().detach()
        h_ = [hi.clone().detach() for hi in h] if h is not None else [torch.empty(N, self.nh[i],
                                                                   device=device) for i in range(self.L)]

        if rand_u is None:
            rand_u = torch.rand(N, device=device)

        even = rand_u < 0.5
        odd = even.logical_not()
        latent_logits = torch.empty_like(h[-1], device=device)

        if even.sum() > 0:
            if not fix_v:
                logits = F.linear(h_[0][even],
                                  self.weight[0].t(), self.bias[0])

                if T == 0:
                    v_[even] = (logits >= 0).float()
                else:
                    logits /= T

                    if rand_v is None:
                        v_[even] = Independent(Bernoulli(logits=logits), 1).sample()
                    else:
                        v_[even] = (rand_v[even] < logits.sigmoid()).float()

            if (self.known_y):
                if not fix_y:
                    logits = F.linear(h_[-1][even],
                                    self.weight[-1], self.bias[-1])

                    if T == 0:
                        y_[even] = logits.clone()
                    else:
                        logits /= T

                        if rand_y is None:
                            y_[even] = Independent(Normal(loc=logits, scale=self.y_sigma), 1).sample()
                        else:
                            raise NotImplementedError("rand_y used for continuous variables in gibbs_step --- DEBUG ---")
                    
            for i in range(1, len(h), 2):
                logits = F.linear(h_[i-1][even],
                                  self.weight[i], self.bias[i+1])
                if i+1 < len(h):
                    logits.add_(F.linear(h_[i+1][even],
                                       self.weight[i+1].t(), None))
                if i+1 == len(h):
                    if (self.known_y):
                        logits.add_(F.linear(y[even]/self.y_sigma**2, self.weight[i+1].t(), None))
                    latent_logits[even] = logits

                if T == 0:
                    h_[i][even] = (logits >= 0).float()
                else:
                    logits /= T

                    if rand_h is None:
                        h_[i][even] = Independent(Bernoulli(logits=logits), 1).sample()
                    else:
                        h_[i][even] = (rand_h[i][even] < logits.sigmoid()).float()

            for i in range(0, len(h), 2):
                logits = F.linear(v_[even] if i==0 else h_[i-1][even],
                                  self.weight[i], self.bias[i+1])
                if i+1 < len(h):
                    logits.add_(F.linear(h_[i+1][even],
                                       self.weight[i+1].t(), None))
                if i+1 == len(h):
                    if (self.known_y):
                        logits.add_(F.linear(y[even]/self.y_sigma**2, self.weight[i+1].t(), None))
                    latent_logits[even] = logits

                if T == 0:
                    h_[i][even] = (logits >= 0).float()
                else:
                    logits /= T

                    if rand_h is None:
                        h_[i][even] = Independent(Bernoulli(logits=logits), 1).sample()
                    else:
                        h_[i][even] = (rand_h[i][even] < logits.sigmoid()).float()


        if odd.sum() > 0:
            for i in range(0, len(h), 2):
                logits = F.linear(v_[odd] if i==0 else h_[i-1][odd],
                                  self.weight[i], self.bias[i+1])
                if i+1 < len(h):
                    logits.add_(F.linear(h_[i+1][odd],
                                       self.weight[i+1].t(), None))
                if i+1 == len(h):
                    if (self.known_y):
                        logits.add_(F.linear(y[odd]/self.y_sigma**2, self.weight[i+1].t(), None))
                    latent_logits[odd] = logits

                if T == 0:
                    h_[i][odd] = (logits >= 0).float()
                else:
                    logits /= T

                    if rand_h is None:
                        h_[i][odd] = Independent(Bernoulli(logits=logits), 1).sample()
                    else:
                        h_[i][odd] = (rand_h[i][odd] < logits.sigmoid()).float()

            if not fix_v:
                logits = F.linear(h_[0][odd],
                                  self.weight[0].t(), self.bias[0])

                if T == 0:
                    v_[odd] = (logits >= 0).float()
                else:
                    logits /= T

                    if rand_v is None:
                        v_[odd] = Independent(Bernoulli(logits=logits), 1).sample()
                    else:
                        v_[odd] = (rand_v[odd] < logits.sigmoid()).float()

            if (self.known_y):
                if not fix_y:
                    logits = F.linear(h_[-1][even],
                                    self.weight[-1], self.bias[-1])

                    if T == 0:
                        y_[even] = logits.clone()
                    else:
                        logits /= T

                        if rand_y is None:
                            y_[even] = Independent(Normal(loc=logits, scale=self.y_sigma), 1).sample()
                        else:
                            raise NotImplementedError("rand_y used for continuous variables in gibbs_step --- DEBUG ---")
                    
            for i in range(1, len(h), 2):
                logits = F.linear(h_[i-1][odd],
                                  self.weight[i], self.bias[i+1])
                if i+1 < len(h):
                    logits.add_(F.linear(h_[i+1][odd],
                                       self.weight[i+1].t(), None))
                if i+1 == len(h):
                    if (self.known_y):
                        logits.add_(F.linear(y[odd]/self.y_sigma**2, self.weight[i+1].t(), None))
                    latent_logits[odd] = logits

                if T == 0:
                    h_[i][odd] = (logits >= 0).float()
                else:
                    logits /= T

                    if rand_h is None:
                        h_[i][odd] = Independent(Bernoulli(logits=logits), 1).sample()
                    else:
                        h_[i][odd] = (rand_h[i][odd] < logits.sigmoid()).float()

        return v_, y_, h_, latent_logits

    @torch.no_grad()
    def mh_step(self, v, y, h, fix_v=False, fix_y=False,
                rand_v=None, rand_y=None, rand_h=None, rand_u=None):
        # print("MH step called with fix_v =", fix_v, "fix_y =", fix_y)
        N = v.size(0)
        device = v.device

        if fix_v:
            v_ = v
        else:
            if rand_v is None:
                v_ = torch.empty_like(v).bernoulli_()
            else:
                v_ = (rand_v < 0.5).float()
        
        if fix_y:
            y_ = y
        else:
            if rand_y is None:
                y_ = torch.randn_like(y)
            else:
                y_ = rand_y

        if rand_h is None:
            h_ = [torch.empty_like(h[i]).bernoulli_() for i in range(self.L)]
        else:
            h_ = [(rand_h[i] < 0.5).float() for i in range(self.L)]
        # print("v = v_?", torch.all(v == v_).item())
        # print("y = y_?", torch.all(y == y_).item())
        # print("h = h_?", all(torch.all(h[i] == h_[i]).item() for i in range(self.L)))
        log_ratio = self.energy(v, y, h) - self.energy(v_, y_, h_)
        # print("log_ratio?", log_ratio)

        # print("data energy", self.energy(v, y, h, show=True))
        # print("sample energy", self.energy(v_, y_, h_, show=True))

        if rand_u is None:
            accepted = log_ratio.exp().clamp(0, 1).bernoulli().bool()
        else:
            accepted = rand_u < log_ratio.exp()
        # print("rand_u:", rand_u)
        # print("log_ratio.exp():", log_ratio.exp())
        # print("accepted:", accepted)

        if not fix_v:
            v = torch.where(accepted.unsqueeze(1), v_, v)
        if not fix_y:
            y = torch.where(accepted.unsqueeze(1), y_, y)
        h = [torch.where(accepted.unsqueeze(1), h_[i], h[i]) for i in range(self.L)]

        return v, y, h

    @torch.no_grad()
    def sample(self, N):
        device = next(self.parameters()).device

        v = torch.empty(N, self.nv, device=device).bernoulli_()
        y = torch.empty(N, self.ny, device=device).normal_()
        h = [torch.empty(N, self.nh[i],
                         device=device).bernoulli_() for i in range(self.L)]

        v_mode, y_mode, h_mode = self.local_search(v, y, h)
        v_rand, y_rand, h_rand, _ = self.gibbs_step(v_mode, y_mode,h_mode)

        return v_mode, v_rand
    
    @torch.no_grad()
    def encode(self, v, y=None):
        N = v.size(0)
        device = v.device
        if (y is None):
            y = torch.zeros(N, self.ny, device=device)
            self.known_y = False
            self.rho = 0.0

        v = v.flatten(1).float()
        h = [torch.empty(N, self.nh[i],
                         device=device).bernoulli_() for i in range(self.L)]

        v, y, h = self.local_search(v, y, h, True, True)
        v_mode, y_mode, h_mode, _ = self.gibbs_step(v, y, h, T=0)
        v_rand, y_rand, h_rand, latent_logits = self.gibbs_step(v, y, h)

        return h_mode[-1], h_rand[-1], latent_logits
    
    @torch.no_grad()
    def decode(self, latent):
        N = latent.size(0)
        device = latent.device

        v = torch.empty(N, self.nv, device=device).bernoulli_()
        y = torch.empty(N, self.ny, device=device).normal_()
        h = [torch.empty(N, self.nh[i],
                         device=device).bernoulli_() for i in range(self.L)]
        h[-1] = latent
        v, y, h = self.local_search(v, y, h, True, True)
        v_mode, y_mode, h_mode, _ = self.gibbs_step(v, y, h, T=0)
        v_rand, y_rand, h_rand, _ = self.gibbs_step(v, y, h)

        return v_mode, v_rand
    
    @torch.no_grad()
    def reconstruct(self, v, y):
        N = v.size(0)
        device = v.device

        v = v.flatten(1).float()
        h = [torch.empty(N, self.nh[i],
                         device=device).bernoulli_() for i in range(self.L)]

        v, y, h = self.local_search(v, y, h, True, True)
        v_mode, y_mode, h_mode, _ = self.gibbs_step(v, y, h, T=0)
        v_rand, y_rand, h_rand, _ = self.gibbs_step(v, y, h)

        return v_mode, v_rand
