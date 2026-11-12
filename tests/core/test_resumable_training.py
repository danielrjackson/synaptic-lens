import pytest
import torch

from synaptic_lens.resumable_training import run_prefix


def fit(updates, initial=None, identity=None):
    torch.manual_seed(7)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    generator = torch.Generator().manual_seed(8)
    saved = []

    def step(model, optimizer, generator):
        # The public fixture uses a plain CPU replay sampler.
        rows = torch.randint(8, (4,), generator=generator)
        x = torch.arange(16, dtype=torch.float32).reshape(8, 2) / 16
        loss = model(x[rows]).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return loss.detach()

    result = run_prefix(model=model, optimizer=optimizer, generator=generator,
                        identity=identity or {"fit": "same"}, updates=updates,
                        step=step, publish=saved.append, initial=initial,
                        checkpoint_every=3)
    return result, saved


def test_interrupted_fit_matches_uninterrupted_model_optimizer_and_replay():
    full, snapshots = fit(9)
    resumed, _ = fit(9, snapshots[0])
    for key in full["model_state"]:
        assert torch.equal(full["model_state"][key], resumed["model_state"][key])
    assert torch.equal(full["sampler_generator_state"], resumed["sampler_generator_state"])
    for slot in full["optimizer_state"]["state"]:
        for key, tensor in full["optimizer_state"]["state"][slot].items():
            assert torch.equal(tensor, resumed["optimizer_state"]["state"][slot][key])
    assert not torch.equal(snapshots[0]["model_state"]["weight"], full["model_state"]["weight"])


def test_mismatched_identity_cannot_resume():
    initial, _ = fit(3)
    with pytest.raises(ValueError, match="identity"):
        fit(9, initial, {"fit": "different"})
