import torch


def safe_positive(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x_safe = torch.where(x > 0, x, torch.full_like(x, eps))
    return x_safe


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
    return suffix


def safe_update_scatter(x: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor, dim: int):
    """
    Autograd-safe overwrite update of `x` at `indices` with `weights`.
    Supports both single and batched updates on arbitrary dimension.
    """
    dim = dim % x.ndim

    # Normalize shapes
    if indices.ndim == 1:
        indices = indices.unsqueeze(0)
        weights = weights.unsqueeze(0)
    B = indices.shape[0]
    assert indices.shape[1] == x.ndim - 1
    assert weights.shape == (B, x.shape[dim])

    # Check duplicates
    if torch.unique(indices, dim=0).shape[0] != B:
        raise ValueError("Duplicate index rows found; updates must be distinct.")

    # Move target dim to last for uniform scatter shape
    x_t = x.movedim(dim, -1).clone()  # (N0,…,N_{d-1},N_{d+1},…,N_{n-1}, Ndim)
    shape_except = x_t.shape[:-1]
    D = x_t.shape[-1]
    flat_x = x_t.reshape(-1, D)

    # Compute flat positions for the provided index rows
    strides = torch.tensor(
        [int(torch.prod(torch.tensor(shape_except[i + 1:]))) if i < len(shape_except) - 1 else 1
         for i in range(len(shape_except))],
        device=indices.device,
        dtype=torch.long,
    )
    flat_pos = (indices * strides).sum(dim=1)  # (B,)

    # Create new flat tensor with updated rows
    update_flat = torch.zeros_like(flat_x)
    update_flat.index_copy_(0, flat_pos, weights)

    # Merge: overwrite the selected rows, keep others
    mask = torch.zeros(flat_x.size(0), dtype=torch.bool, device=flat_x.device)
    mask[flat_pos] = True
    flat_x = torch.where(mask.unsqueeze(1), update_flat, flat_x)

    # Reshape back and move dimension to original position
    x_new = flat_x.view(*shape_except, D).movedim(-1, dim)
    return x_new


def advanced_multi_index_select(a: torch.Tensor, b: torch.Tensor, dims):
    """
    Generalized multi-dimensional indexing (autograd-safe).

    Args:
        a: Tensor of shape (D0, D1, ..., Dp)
        b: LongTensor of shape (N, n)
           Each row gives indices along the dimensions specified in `dims`
        dims: 1D LongTensor, list, or tuple of length n
           Which dimensions of `a` are being indexed.

    Returns:
        Tensor of shape (N, remaining_dims_of_a)
    """
    assert b.ndim == 2, "b must be 2D"
    assert b.shape[1] == len(dims), f"b.shape[1] ({b.shape[1]}) must match len(dims) ({len(dims)})"

    # ensure dims is a list of ints
    dims = [int(d) for d in dims]
    N, n = b.shape
    total_dims = a.ndim

    # 1️⃣ Move indexed dims to the front
    permute_order = dims + [d for d in range(total_dims) if d not in dims]
    a_perm = a.permute(*permute_order)  # convert list -> unpack to ints

    prefix_shape = [a.shape[d] for d in dims]
    suffix_shape = [a.shape[d] for d in range(total_dims) if d not in dims]
    flat_a = a_perm.reshape(int(torch.prod(torch.tensor(prefix_shape))), *suffix_shape)

    # 2️⃣ Compute flat indices corresponding to b[:, dims]
    strides = torch.tensor(
        [int(torch.prod(torch.tensor(prefix_shape[i+1:]))) if i < n-1 else 1
         for i in range(n)],
        device=b.device,
        dtype=torch.long
    )
    flat_idx = (b * strides).sum(dim=1)

    # 3️⃣ Gather the indexed rows
    result = flat_a[flat_idx]
    return result




