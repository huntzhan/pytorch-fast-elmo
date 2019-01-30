"""
All tests are copied from AllenNLP.
"""
import torch
import pytest
import numpy
from pytorch_fast_elmo import ScalarMix


def access_scalar_parameters(mixture, k):
    return mixture.named_parameters()['scalar_' + str(k)]


def access_gamma(mixture):
    return mixture.named_parameters()['gamma']


def test_scalar_mix_can_run_forward():
    mixture = ScalarMix(3)
    tensors = [torch.randn([3, 4, 5]) for _ in range(3)]
    for k in range(3):
        access_scalar_parameters(mixture, k).data[0] = 0.1 * (k + 1)
    access_gamma(mixture).data[0] = 0.5
    result = mixture(tensors)

    weights = [0.1, 0.2, 0.3]
    normed_weights = numpy.exp(weights) / numpy.sum(numpy.exp(weights))
    expected_result = sum(normed_weights[k] * tensors[k].data.numpy() for k in range(3))
    expected_result *= 0.5
    numpy.testing.assert_almost_equal(expected_result, result.data.numpy())


def test_scalar_mix_throws_error_on_incorrect_number_of_inputs():
    mixture = ScalarMix(3)
    tensors = [torch.randn([3, 4, 5]) for _ in range(5)]
    with pytest.raises(ValueError):
        _ = mixture(tensors)


def test_scalar_mix_throws_error_on_incorrect_initial_scalar_parameters_length():
    with pytest.raises(ValueError):
        ScalarMix(3, initial_scalar_parameters=[0.0, 0.0])


def test_scalar_mix_trainable_with_initial_scalar_parameters():
    initial_scalar_parameters = [1.0, 2.0, 3.0]
    mixture = ScalarMix(3, initial_scalar_parameters=initial_scalar_parameters, trainable=False)
    for k, initial_scalar_parameter in enumerate(initial_scalar_parameters):
        scalar_mix_parameter = access_scalar_parameters(mixture, k)
        assert scalar_mix_parameter.requires_grad is False
        assert scalar_mix_parameter.item() == initial_scalar_parameter


def test_scalar_mix_layer_norm():
    mixture = ScalarMix(3, do_layer_norm=True)

    tensors = [torch.randn([3, 4, 5]) for _ in range(3)]
    numpy_mask = numpy.ones((3, 4), dtype='int32')
    numpy_mask[1, 2:] = 0
    mask = torch.from_numpy(numpy_mask)

    weights = [0.1, 0.2, 0.3]
    for k in range(3):
        access_scalar_parameters(mixture, k).data[0] = weights[k]
    access_gamma(mixture).data[0] = 0.5
    result = mixture(tensors, mask)

    normed_weights = numpy.exp(weights) / numpy.sum(numpy.exp(weights))
    expected_result = numpy.zeros((3, 4, 5))
    for k in range(3):
        mean = numpy.mean(tensors[k].data.numpy()[numpy_mask == 1])
        std = numpy.std(tensors[k].data.numpy()[numpy_mask == 1])
        normed_tensor = (tensors[k].data.numpy() - mean) / (std + 1E-12)
        expected_result += (normed_tensor * normed_weights[k])
    expected_result *= 0.5

    numpy.testing.assert_almost_equal(expected_result, result.data.numpy(), decimal=6)


def test_scalar_mix_layer_norm_packed_sequence():
    mixture = ScalarMix(3, do_layer_norm=True)
    tensors = [torch.randn([3, 4]) for _ in range(3)]
    mixture(tensors)
