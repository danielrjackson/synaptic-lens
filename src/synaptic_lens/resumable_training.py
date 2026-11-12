"""Deterministic fit-prefix continuation shared by study controllers.

The caller owns source/input identity, leases, resource admission and atomic
publication. This primitive owns optimizer/model/sampler continuation only.
"""

from copy import deepcopy

import torch


def cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_state(item) for item in value)
    return deepcopy(value)


def run_prefix(*, model, optimizer, generator, identity, updates, step,
               publish, initial=None, checkpoint_every=500, endpoints=(),
               observe=None, observe_every=50):
    """Continue a fixed chain, publishing independent durable CPU snapshots.

    ``step`` receives the model, optimizer and CPU replay generator and returns
    the loss. Endpoints are fixed before execution; observation never selects
    a different endpoint. Models must use no additional stochastic operations.
    """
    if updates < 1 or checkpoint_every < 1 or observe_every < 1:
        raise ValueError("Update and checkpoint intervals must be positive")
    endpoints = set(endpoints)
    if any(not isinstance(item, int) or item < 1 or item > updates for item in endpoints):
        raise ValueError("Endpoints must lie within the fixed chain")
    start = 0
    history = []
    if initial is not None:
        if initial["identity"] != identity:
            raise ValueError("Resume source/input/fit identity differs")
        start = initial["update"]
        if not isinstance(start, int) or not 0 <= start <= updates:
            raise ValueError("Resume update lies outside the fixed chain")
        model.load_state_dict(initial["model_state"])
        optimizer.load_state_dict(initial["optimizer_state"])
        generator.set_state(initial["sampler_generator_state"])
        history = deepcopy(initial["history"])

    def snapshot(update):
        return {"identity": deepcopy(identity), "update": update,
                "model_state": cpu_state(model.state_dict()),
                "optimizer_state": cpu_state(optimizer.state_dict()),
                "sampler_generator_state": generator.get_state().clone(),
                "history": deepcopy(history)}

    for update in range(start + 1, updates + 1):
        model.train()
        loss = step(model, optimizer, generator)
        if update % observe_every == 0 or update in endpoints or update == updates:
            if not bool(torch.isfinite(loss).all()):
                raise ValueError(f"Nonfinite loss at update {update}")
            if observe is not None:
                model.eval()
                with torch.no_grad():
                    metrics = observe(model)
                if any(not bool(torch.isfinite(torch.as_tensor(v)).all()) for v in metrics.values()):
                    raise ValueError(f"Nonfinite observation at update {update}")
                history.append({"update": update, **metrics})
        if update % checkpoint_every == 0 or update in endpoints or update == updates:
            publish(snapshot(update))
    return snapshot(updates)
