import argparse
import os
import random
import math
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision

from utils import generate_id, visualize_curve


# --------------------------
# Utilities
# --------------------------
def is_dist_env():
    # True if launched by torchrun / SLURM with env vars set
    return ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ)

def get_dist_info():
    if is_dist_env():
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        return True, rank, local_rank, world_size
    else:
        return False, 0, 0, 1

def init_distributed():
    # Init from environment set by torchrun
    dist.init_process_group(backend="nccl", init_method="env://")

def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# --------------------------
# Training / Validation
# --------------------------
def train(rank, args, model, optimizer, scheduler, dataloader, writer, epoch, detect_anomaly=False):
    model.train()
    energies_per_batch = []
    loss_per_batch = []
    device = next(model.parameters()).device

    with torch.autograd.set_detect_anomaly(detect_anomaly):
        for step, (x, y) in enumerate(dataloader):
            v = x.to(device, non_blocking=True)
            r = y.to(device, non_blocking=True).reshape(-1, args.ny)

            optimizer.zero_grad(set_to_none=True)
            loss = model(v, r)
            loss.mean().backward()
            optimizer.step()
            scheduler.step()

            loss_per_batch.append(loss.detach().float().cpu().numpy())
            # model might be DDP or plain; use getattr-safe access to .module
            m = model.module if hasattr(model, "module") else model
            with torch.no_grad():
                energies_per_batch.append(m.marginal_energy(v, r).detach().float().cpu().numpy())

    # return np.array(energies_per_batch).flatten(), np.array(loss_per_batch).flatten()
    return np.concatenate(energies_per_batch), np.concatenate(loss_per_batch)


def valid(rank, args, model, dataloader, writer, epoch):
    model.eval()
    device = next(model.parameters()).device
    if dataloader is not None:
        try:
            x, y = next(iter(dataloader))
        except StopIteration:
            return
        v = x.to(device, non_blocking=True)
        r = y.to(device, non_blocking=True).reshape(-1, args.ny)
        m = model.module if hasattr(model, "module") else model
        with torch.no_grad():
            _ = m.reconstruct(v, r)  # you can add TB images if you like


# --------------------------
# Args / Seed
# --------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Deep Boltzmann Machine (supervised, multinomial)")
    parser.add_argument('--model_type', type=str, default='bernoulli', choices=['bernoulli', 'multinomial'], help='DBM Top latent layer distribution type')
    parser.add_argument('--learning_type', type=str, default='unsupervised', choices=['unsupervised', 'supervised'], help='DBM learning type')
    parser.add_argument('--num_workers', type=int, default=8, help='DataLoader workers per process')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument("--dataset", type=str, default="Mutations")
    parser.add_argument("--nv", type=int, default=4801)
    parser.add_argument("--nh", type=int, nargs='+', default=[4900, 3600])
    parser.add_argument("--ny", type=int, default=1)
    parser.add_argument("--L", type=int, default=2)
    parser.add_argument("--nMulti", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--y_sigma", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--gamma", type=float, default=1e-4)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1000)   # global batch (we’ll split per-rank)
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--detect_anomaly", action='store_true')
    parser.add_argument("--model_path", type=str, default='models')
    parser.add_argument("--distributed", action='store_true',
                        help="Enable DDP if running under torchrun; ignored if not in dist env.")
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


# --------------------------
# Main runner (per process)
# --------------------------
def run(args):
    in_dist_env, rank, local_rank, world_size = get_dist_info()

    # Respect --distributed flag: only initialize if both flag and env are present
    use_ddp = args.distributed and in_dist_env

    if use_ddp:
        init_distributed()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0
        world_size = 1
        local_rank = 0

    # Data
    if args.dataset != "Mutations":
        raise NotImplementedError("Only 'Mutations' is implemented in this snippet.")
    data = torch.load("cell_drug_response_samples.pt")
    full_dataset = torch.utils.data.TensorDataset(data['X'], data['y'])

    # Split
    train_len = int(0.8 * len(full_dataset))
    test_len = len(full_dataset) - train_len
    training_data, test_data = torch.utils.data.random_split(full_dataset, [train_len, test_len])

    # Samplers
    if use_ddp:
        train_sampler = DistributedSampler(training_data, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        test_sampler  = DistributedSampler(test_data,  num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        test_sampler  = None

    # Batch size per process
    assert args.batch_size >= world_size, "batch_size must be >= world_size"
    per_rank_bs = args.batch_size // world_size

    # Dataloader kwargs
    nw = max(0, args.num_workers)
    dl_kwargs = dict(
        batch_size=per_rank_bs,
        pin_memory=True,
        num_workers=nw,
        persistent_workers=(nw > 0),
    )

    train_loader = DataLoader(training_data, sampler=train_sampler, shuffle=(train_sampler is None), **dl_kwargs)
    test_loader  = DataLoader(test_data,  sampler=test_sampler,  shuffle=False, **dl_kwargs)

    # Model
    if (args.model_type == 'bernoulli') and (args.learning_type == 'unsupervised'):
        from model import DBM
    elif (args.model_type == 'multinomial') and (args.learning_type == 'supervised'):
        from model_multinomial_supervised import DBM
    elif (args.model_type == 'bernoulli') and (args.learning_type == 'supervised'):
        from model_supervised import DBM
    elif (args.model_type == 'multinomial') and (args.learning_type == 'unsupervised'):
        from model_multinomial import DBM
    else:
        raise NotImplementedError("Model type and learning type combination not implemented.")
    
    model = DBM(args.nv, args.nh, args.ny, args.L, args.nMulti, args.y_sigma, args.rho).to(device)
    if use_ddp:
        # IMPORTANT: create optimizer *after* moving to device and *before* wrapping in DDP is fine;
        # but model params are the same objects after wrapping.
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    else:
        # Optional: for local multi-GPU debugging you could use DataParallel; here we keep it single-GPU by default.
        pass

    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda t: 1.0 / math.sqrt(1 + args.gamma * t))

    # Logging
    if args.log_dir is None:
        run_id = generate_id(8)
        log_dir = (
            f"./hpc_runs/{args.learning_type}/{args.model_type}/{args.dataset}-{args.learning_type}-{args.model_type}-L:{args.L}-nh:{args.nh}-ySigma:{args.y_sigma}-rho:{args.rho}"
            f"-nMulti:{args.nMulti}-lr:{args.lr}-momentum:{args.momentum}"
            f"-bs:{args.batch_size}-gamma:{args.gamma}-epoch:{args.epoch}-seed:{args.seed}-{run_id}"
        )
    else:
        log_dir = args.log_dir

    writer = SummaryWriter(log_dir) if rank == 0 else None
    if rank == 0:
        os.makedirs(os.path.join(log_dir, args.model_path), exist_ok=True)

    # One warmup valid
    valid(rank, args, model, test_loader, writer, 0)

    energies_mean, energies_std = [], []
    losses_mean, losses_std = [], []

    for epoch in tqdm(range(1, args.epoch + 1)):
        if use_ddp:
            # ensure different shuffles each epoch
            assert isinstance(train_loader.sampler, DistributedSampler)
            train_loader.sampler.set_epoch(epoch)

        energy, loss = train(rank, args, model, optimizer, scheduler, train_loader, writer, epoch, args.detect_anomaly)
        valid(rank, args, model, test_loader, writer, epoch)

        # Only rank 0 aggregates scalars to avoid duplicate writes
        if rank == 0:
            energies_mean.append(float(np.mean(energy)) if energy.size else float('nan'))
            energies_std.append(float(np.std(energy)) if energy.size else float('nan'))
            losses_mean.append(float(np.mean(loss)) if loss.size else float('nan'))
            losses_std.append(float(np.std(loss)) if loss.size else float('nan'))

            if epoch % 10 == 0:
                # Save model (DDP-safe: state_dict of the wrapped module)
                m = model.module if hasattr(model, "module") else model
                torch.save(m.state_dict(), os.path.join(log_dir, args.model_path, f"model-{epoch}.pt"))
                visualize_curve(np.array(energies_mean), np.array(energies_std),
                                np.array(losses_mean), np.array(losses_std),
                                epoch, log_dir, writer)

    if writer is not None:
        writer.close()

    if use_ddp:
        cleanup_distributed()


def load_model(log_dir, device, args):
    if (args.model_type == 'bernoulli') and (args.learning_type == 'unsupervised'):
        from model import DBM
    elif (args.model_type == 'multinomial') and (args.learning_type == 'supervised'):
        from model_multinomial_supervised import DBM
    elif (args.model_type == 'bernoulli') and (args.learning_type == 'supervised'):
        from model_supervised import DBM
    elif (args.model_type == 'multinomial') and (args.learning_type == 'unsupervised'):
        from model_multinomial import DBM
    else:
        raise NotImplementedError("Model type and learning type combination not implemented.")
    model = DBM(args.nv, args.nh, args.ny, args.L, args.nMulti, args.y_sigma, args.rho, False).to(device)
    checkpoint_path = os.path.join(log_dir, args.model_path, f"model-{args.epoch}.pt")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


# --------------------------
# Entrypoint
# --------------------------
if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)

    # Under torchrun: launch exactly once per process; no mp.spawn here.
    # Local (no torchrun): you can still run `python train.py` for single-GPU debugging.
    run(args)
