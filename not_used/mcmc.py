import torch
import pyro
import pyro.poutine as poutine


def gibbs_step(model, current_state, variable_names):
    """
    Perform one Gibbs step over all variables.
    model: your pyro model
    current_state: dict of variable_name -> current tensor
    variable_names: list of discrete latent variable names
    """
    new_state = current_state.copy()
    for var in variable_names:
        cond_state = {k: v for k, v in new_state.items() if k != var}

        # Try both possible values (e.g., for binary variables 0 and 1)
        log_probs = []
        for val in [torch.tensor(0.0), torch.tensor(1.0)]:
            temp_state = cond_state.copy()
            temp_state[var] = val
            trace = poutine.trace(poutine.condition(model, data=temp_state)).get_trace()
            log_probs.append(trace.log_prob_sum())

        log_probs = torch.stack(log_probs)
        probs = torch.softmax(log_probs, dim=0)

        # Sample new value
        new_val = torch.multinomial(probs, num_samples=1).float()
        new_state[var] = new_val
    return new_state
