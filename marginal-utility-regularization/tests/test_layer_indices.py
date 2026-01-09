from models.hidden_state_utils import resolve_layer_indices


def test_resolve_layer_indices_fractional() -> None:
    indices = resolve_layer_indices(12, 0.25, 0.75)
    assert indices[0] >= 0
    assert indices[-1] < 12
    assert len(indices) > 0
