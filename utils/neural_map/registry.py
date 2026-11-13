from .neural_map import NeuralMap, MapCfg
from .ema_neural_map import EMANeuralMap, EMANeuralMapCfg
from .trust_neural_map import TrustNeuralMap, TrustNeuralMapCfg
from .directional_bayesian_neural_map import (
    DirectionalBayesianNeuralMap,
    DirectionalBayesianNeuralMapCfg,
)
from .cellwise_bayesian_neural_map import (
    CellwiseBayesianNeuralMap,
    CellwiseBayesianNeuralMapCfg,
)

NeuralMapRegistry = {
    "ema_neural_map": (EMANeuralMap, EMANeuralMapCfg()),
    "trust_neural_map": (TrustNeuralMap, TrustNeuralMapCfg()),
    "directional_bayesian_neural_map": (
        DirectionalBayesianNeuralMap,
        DirectionalBayesianNeuralMapCfg(),
    ),
    "cellwise_bayesian_neural_map": (
        CellwiseBayesianNeuralMap,
        CellwiseBayesianNeuralMapCfg(),
    ),
}
