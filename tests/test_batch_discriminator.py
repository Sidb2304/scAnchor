import torch

from scanchor.model.batch_discriminator import BatchDiscriminator, dann_lambda_schedule, gradient_reversal


def test_gradient_reversal_forward_is_identity():
    x = torch.randn(4, 3)
    assert torch.equal(gradient_reversal(x, lambd=2.0), x)


def test_gradient_reversal_negates_gradient():
    x = torch.randn(5, requires_grad=True)
    (x * 3).sum().backward()
    grad_plain = x.grad.clone()
    x.grad = None

    (gradient_reversal(x, lambd=1.0) * 3).sum().backward()

    assert torch.allclose(x.grad, -grad_plain)


def test_gradient_reversal_lambd_scales_gradient():
    x = torch.randn(5, requires_grad=True)

    (gradient_reversal(x, lambd=0.5) * 2).sum().backward()

    assert torch.allclose(x.grad, torch.full_like(x, -1.0))


def test_batch_discriminator_output_shape():
    disc = BatchDiscriminator(embed_dim=16, n_batches=4)
    embeddings = torch.randn(6, 16)

    logits = disc(embeddings)

    assert logits.shape == (6, 4)


def test_batch_discriminator_default_capacity_matches_correction_head():
    """Locks in the capacity fix: 2 hidden layers at >=128 units, not the
    original 1-layer/64-unit default that let the discriminator reach
    equilibrium without removing batch structure a kNN metric could detect.
    """
    disc = BatchDiscriminator(embed_dim=512, n_batches=9)
    linear_layers = [m for m in disc.net if isinstance(m, torch.nn.Linear)]

    assert len(linear_layers) == 3  # input->hidden, hidden->hidden, hidden->output
    assert linear_layers[0].out_features >= 128


def test_batch_discriminator_grl_flows_gradient_to_upstream_input():
    disc = BatchDiscriminator(embed_dim=8, n_batches=3)
    embeddings = torch.randn(5, 8, requires_grad=True)
    labels = torch.randint(0, 3, (5,))

    logits = disc(embeddings, lambd=1.0)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()

    assert disc.net[0].weight.grad is not None
    assert embeddings.grad is not None


def test_dann_lambda_schedule_starts_at_zero():
    assert dann_lambda_schedule(progress=0.0, max_lambda=1.0) == 0.0


def test_dann_lambda_schedule_approaches_max_at_progress_one():
    assert dann_lambda_schedule(progress=1.0, max_lambda=2.0) > 1.9


def test_dann_lambda_schedule_monotonically_increases():
    values = [dann_lambda_schedule(p / 10, max_lambda=1.0) for p in range(11)]
    assert values == sorted(values)


def test_dann_lambda_schedule_clamps_out_of_range_progress():
    assert dann_lambda_schedule(progress=-1.0, max_lambda=1.0) == 0.0
    assert dann_lambda_schedule(progress=5.0, max_lambda=1.0) == dann_lambda_schedule(1.0, max_lambda=1.0)
