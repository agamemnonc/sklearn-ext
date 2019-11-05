import numpy as np

from sklearn.neural_network import MLPClassifier, MLPRegressor

__all__ = [
    'MLPClassifierPlus',
    'MLPRegressorPlus'
]


def _get_activations(mdl, X, layer):
    """Returns activations of any layer in a neural network.

    Parameters
    ----------
    mdl : estimator
        Neural netwoork model.
    X : array-like, shape = [n_samples, n_features]
        Input data.
    layer : int
        The layer for which activations are queried. If None, activations of
        all layers are returned.

    Returns
    -------
    activations : array-like or list of array-likes
        Activations for a single layer or activations of all layers when
        ``layer`` parameter is set to None.
    """
    hidden_layer_sizes = mdl.hidden_layer_sizes
    if not hasattr(hidden_layer_sizes, "__iter__"):
        hidden_layer_sizes = [hidden_layer_sizes]
    hidden_layer_sizes = list(hidden_layer_sizes)
    layer_units = [X.shape[1]] + hidden_layer_sizes + \
        [mdl.n_outputs_]
    activations = [X]
    for i in range(mdl.n_layers_ - 1):
        activations.append(np.empty((X.shape[0],
                                     layer_units[i + 1])))
    activations = mdl._forward_pass(activations)
    if layer is not None:
        return activations[layer]
    else:
        return activations

class MLPClassifierPlus(MLPClassifier):
    """MLPCLassifier with extensions."""
    def get_activations(self, X, layer=None):
        """Returns activations of any layer in the network."""
        return _get_activations(self, X, layer)

class MLPRegressorPlus(MLPRegressor):
    """MLPRegressor with extensions."""
    def get_activations(self, X, layer=None):
        """Returns activations of any layer in the network."""
        return _get_activations(self, X, layer)
