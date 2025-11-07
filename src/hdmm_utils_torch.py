import torch


def stats_by_label(data: torch.Tensor, labels: torch.Tensor, num_classes: int, eps: float = 1e-8):
    """
    Compute mean, variance, sum of variance, and count for each label.
    Handles both 1D (N,) and 2D (N, D) data tensors, including empty classes.

    Returns:
        means: (num_classes, D)
        variances: (num_classes, D)
        sum_variances: (num_classes,)
        counts: (num_classes,)
    """
    if data.dim() == 1:
        data = data.unsqueeze(1)  # (N, 1)

    N, D = data.shape
    one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()  # (N, C)
    counts = one_hot.sum(dim=0)  # (C,)

    # avoid div-by-zero downstream
    safe_counts = counts.clamp_min(eps)

    # Mean per class
    sums = one_hot.T @ data  # (C, D)
    means = sums / safe_counts.unsqueeze(1)

    # Variance per class
    diff = data.unsqueeze(1) - means.unsqueeze(0)  # (N, C, D)
    sq_diff = diff.pow(2)
    weighted_sq = one_hot.unsqueeze(2) * sq_diff
    var_sums = weighted_sq.sum(dim=0)  # (C, D)
    variances = var_sums / safe_counts.unsqueeze(1)

    # Sum of variances per label (scalar)
    sum_variances = variances.sum(dim=1)

    # Explicitly zero-out stats for empty components
    empty = counts == 0
    means[empty] = 0.0
    variances[empty] = 0.0
    sum_variances[empty] = 0.0

    return means, variances, sum_variances, counts


def get_unique_rows_and_positions(x: torch.Tensor):
    """
    Get all unique rows in data and their positions.

    Args:
        x: (N, D) tensor of data points.

    Returns:
        unique_rows: (N_unique, D) tensor of unique rows.
        positions: list of index tensors; positions[i] contains the indices in x
                   that correspond to unique_rows[i].
    """
    # Ensure float32 or int dtype, no gradients
    x = x.detach()

    # Step 1: Get unique rows and inverse mapping
    unique_rows, inv_idx = torch.unique(x, dim=0, return_inverse=True)

    # Step 2: Group positions for each unique row
    positions = [(inv_idx == i).nonzero(as_tuple=False).flatten() for i in range(unique_rows.size(0))]

    return unique_rows, positions


def mix_weights(beta, axis=-1):
    """
    Fully general stick-breaking mixture weights (PyTorch version)
    Supports arbitrary batch and group dimensions.
    """
    # Ensure floating dtype and numerical stability
    beta = beta.clamp(min=1e-10, max=1-1e-10)

    # Compute cumulative product of remaining sticks
    remaining = torch.cumprod(1 - beta, dim=axis)

    # Construct ones tensor for prepending
    ones_shape = list(beta.shape)
    ones_shape[axis] = 1
    ones = torch.ones(ones_shape, dtype=beta.dtype, device=beta.device)

    # Build shifted version (prepend ones, remove last element along axis)
    idx = [slice(None)] * beta.ndim
    idx[axis] = slice(0, -1)
    shifted = torch.cat([ones, remaining[tuple(idx)]], dim=axis)

    # Final mixture weights
    return beta * shifted


def suffix_sum(x: torch.Tensor) -> torch.Tensor:
    """
    Compute suffix sums along the last dimension of a tensor.
    Each entry is the sum of all elements to its right.
    The last element along that dimension is always 0.

    Example:
        x = torch.tensor([1, 2, 3])
        suffix_sum(x) -> tensor([5, 3, 0])

        x = torch.tensor([[1, 2, 3], [4, 5, 6]])
        suffix_sum(x) -> tensor([[5, 3, 0],
                                 [11, 6, 0]])
    """
    # Flip along the last dimension
    rev = torch.flip(x, dims=[-1])
    # Cumulative sum on the flipped tensor
    rev_cumsum = torch.cumsum(rev, dim=-1)
    # Flip back to original order
    suffix = torch.flip(rev_cumsum, dims=[-1])
    # Subtract original to exclude the current element
    suffix = suffix - x
    # Clip to ensure non-negativity and avoid underflow
    suffix = torch.clamp(suffix, min=1e-10)
    return suffix


def safe_update_scatter(tensor: torch.Tensor,
                        indices: torch.LongTensor,
                        values: torch.Tensor) -> torch.Tensor:
    """
    Vectorized autograd-safe update along the last dimension using torch.scatter.

    tensor: (..., k)
    indices: (num_updates, tensor.ndim - 1)
              Each row gives the coordinates (e.g. (a, b)).
    values:  (num_updates, k)
              Replacement rows.
    """
    out = tensor.clone()
    *prefix_shape, k = out.shape
    n_prefix = int(torch.prod(torch.tensor(prefix_shape, device=tensor.device)))

    flat = out.view(n_prefix, k)

    # Compute strides for flattening all but last dimension
    strides = []
    for i in range(len(prefix_shape)):
        if i + 1 < len(prefix_shape):
            stride = int(torch.prod(torch.tensor(prefix_shape[i+1:], device=tensor.device)))
        else:
            stride = 1
        strides.append(stride)
    strides = torch.tensor(strides, device=tensor.device, dtype=torch.long)

    # Compute flattened indices
    flat_idx = (indices.long() * strides).sum(dim=1)

    # Expand flat indices to match (num_updates, k)
    scatter_index = flat_idx.unsqueeze(1).expand(-1, k)

    # Scatter new values into the flattened view
    flat = flat.scatter(0, scatter_index, values)

    return flat.view(out.shape)


def advanced_multi_index_select(a: torch.Tensor, b: torch.Tensor, dims: torch.Tensor):
    """
    Generalized multi-dimensional indexing.

    Args:
        a: tensor of shape (D0, D1, ..., Dp)
        b: LongTensor of shape (N, n)
           where each row gives indices along the dimensions specified in `dims`
        dims: 1D LongTensor or list of ints of length n
           specifying which dimensions of `a` are being indexed.

    Returns:
        Tensor of shape (N, remaining_dims_of_a)
        where dimensions not in `dims` are kept.
    """
    assert b.shape[1] == len(dims), "b.shape[1] must match len(dims)"

    dims = torch.as_tensor(dims, dtype=torch.long, device=b.device)
    N, n = b.shape
    total_dims = a.ndim

    # 1️⃣ Move indexed dims to the front so we can flatten them easily
    permute_order = torch.cat([dims, torch.tensor([d for d in range(total_dims) if d not in dims], device=b.device)])
    a_perm = a.permute(*permute_order)
    
    prefix_shape = [a.shape[d] for d in dims]
    suffix_shape = [a.shape[d] for d in range(total_dims) if d not in dims]
    flat_a = a_perm.reshape(int(torch.prod(torch.tensor(prefix_shape))), *suffix_shape)

    # 2️⃣ Compute flat indices corresponding to b[:, dims]
    strides = torch.tensor(
        [int(torch.prod(torch.tensor(prefix_shape[i+1:]))) for i in range(n)],
        device=b.device
    )
    flat_idx = (b * strides).sum(dim=1)

    # 3️⃣ Gather the indexed rows
    result = flat_a[flat_idx]
    return result



