import pytest

import rose.hpo.bootstrap_hpo as bootstrap_hpo


def test_parse_args_valid():
    # If you have a parse_args function
    if not hasattr(bootstrap_hpo, "parse_args"):
        pytest.skip("bootstrap_hpo.parse_args not implemented")

    args = bootstrap_hpo.parse_args([
        "--strategy", "grid",
        "--learning_rate", "0.001",
        "--num_layers", "2",
        "--hidden_units", "64",
        "--batch_size", "64",
        "--l2_reg", "0.0",
    ])

    assert args.strategy == "grid"
    assert args.learning_rate == pytest.approx(0.001)
    assert args.num_layers == 2
    assert args.hidden_units == 64
    assert args.batch_size == 64
    assert args.l2_reg == pytest.approx(0.0)


def test_parse_args_invalid_strategy():
    if not hasattr(bootstrap_hpo, "parse_args"):
        pytest.skip("bootstrap_hpo.parse_args not implemented")

    with pytest.raises(SystemExit):
        # Assuming argparse exits on invalid choice
        bootstrap_hpo.parse_args([
            "--strategy", "invalid_strategy",
        ])
