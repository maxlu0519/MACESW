import torch

from phasei_ml.gnn_model import PhaseIGNN, losses, parameter_count


def _empty_batch(batch_size=2):
    dims = {"scifi": 9, "ttc": 8, "ecal": 9}
    return {name: {"x": torch.zeros((0, dims[name])),
                   "channel": torch.zeros((0,), dtype=torch.long),
                   "batch": torch.zeros((0,), dtype=torch.long),
                   "ptr": torch.zeros((batch_size + 1,), dtype=torch.long),
                   "edge_index": torch.zeros((2, 0), dtype=torch.long),
                   "edge_attr": torch.zeros((0, dims[name])),
                   "counts": torch.zeros((batch_size,))}
            for name in dims}


def test_model_contract_parameter_count_and_zero_hit_prior_gradient():
    torch.manual_seed(4)
    model = PhaseIGNN()
    assert 2_700_000 <= parameter_count(model) <= 3_200_000
    output = model(_empty_batch())
    assert output["pid"].shape == (2, 2)
    assert output["energy"].shape == (2, 15)
    assert output["position"].shape == (2, 40)
    assert output["direction"].shape == (2, 20)
    assert output["is_zero"].tolist() == [True, True]
    target = {"pid": torch.tensor([0, 1]), "energy": torch.zeros(2),
              "position": torch.zeros((2, 3)), "direction": torch.tensor([[1., 0., 0.], [0., 1., 0.]])}
    total = sum(value.mean() for value in losses(output, target).values())
    total.backward()
    assert model.zero_prior.grad is not None
    assert torch.isfinite(model.zero_prior.grad).all()
    assert model.zero_prior.grad.abs().sum() > 0


def test_checkpoint_readback_preserves_probabilistic_predictions(tmp_path):
    torch.manual_seed(7)
    model = PhaseIGNN().eval()
    batch = _empty_batch(batch_size=1)
    with torch.no_grad():
        expected = model(batch)
    path = tmp_path / "gnn_state.pt"
    torch.save(model.state_dict(), path)
    restored = PhaseIGNN().eval()
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    with torch.no_grad():
        actual = restored(batch)
    for head in ("pid", "energy", "position", "direction"):
        torch.testing.assert_close(actual[head], expected[head])
