from backend.monitoring.drift import compute_psi


def test_compute_psi_basic():
    expected = [0.25, 0.25, 0.25, 0.25]
    observed_same = [0.25, 0.25, 0.25, 0.25]
    observed_shift = [0.10, 0.15, 0.35, 0.40]

    psi_same = compute_psi(observed_same, expected)
    psi_shift = compute_psi(observed_shift, expected)

    assert psi_same >= 0.0 and psi_same < 0.01
    assert psi_shift > 0.0

