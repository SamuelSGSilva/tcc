"""
test_deep_classifier.py
=======================
Testes unitários para DeepCrackClassifier (ResNet50).
Não requer pesos treinados — valida apenas a estrutura do modelo e inferência básica.

Se o PyTorch não estiver instalado, todos os testes são pulados automaticamente.
"""

import numpy as np
import pytest

# Skip gracioso se PyTorch não estiver instalado
torch = pytest.importorskip("torch", reason="PyTorch não instalado — testes do ResNet50 pulados.")
from src.deep_classifier import DeepCrackClassifier, CLASSES


# --------------------------------------------------------------------------- #
# Fixture: modelo sem pesos fine-tuned (usa ImageNet como base)
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def clf():
    """Instancia o classificador sem caminho de pesos. Usa CPU."""
    return DeepCrackClassifier(weights_path=None, device="cpu")


# --------------------------------------------------------------------------- #
# Testes de inicialização
# --------------------------------------------------------------------------- #

def test_model_loads_without_weights(clf):
    """O modelo deve instanciar sem erros mesmo sem pesos fine-tuned."""
    assert clf is not None
    assert clf.model is not None


def test_model_classes_correct(clf):
    """As classes do modelo devem ser exatamente as 4 categorias definidas."""
    assert clf.CLASSES == ["CRITICA", "SEMI_CRITICA", "NAO_CRITICA", "DESCARTADA"]


def test_model_is_on_cpu(clf):
    """Com device='cpu', o modelo deve estar na CPU."""
    assert clf.device.type == "cpu"


def test_model_in_eval_mode(clf):
    """O modelo deve estar em modo eval após inicialização."""
    import torch.nn as nn
    for module in clf.model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.Dropout)):
            assert not module.training


# --------------------------------------------------------------------------- #
# Testes de inferência
# --------------------------------------------------------------------------- #

def test_predict_returns_valid_class(clf):
    """predict() deve retornar uma classe dentro das 4 categorias."""
    dummy_image = np.zeros((512, 512, 3), dtype=np.uint8)
    crack_class, confidence = clf.predict(dummy_image)
    assert crack_class in CLASSES


def test_predict_returns_confidence_between_0_and_1(clf):
    """A confiança deve estar no intervalo [0, 1]."""
    dummy_image = np.zeros((512, 512, 3), dtype=np.uint8)
    _, confidence = clf.predict(dummy_image)
    assert 0.0 <= confidence <= 1.0


def test_predict_with_blank_image(clf):
    """Imagem completamente preta (sem fissura) não deve levantar exceção."""
    blank = np.zeros((256, 256, 3), dtype=np.uint8)
    crack_class, confidence = clf.predict(blank)
    assert isinstance(crack_class, str)
    assert isinstance(confidence, float)


def test_predict_with_white_image(clf):
    """Imagem completamente branca não deve levantar exceção."""
    white = np.full((256, 256, 3), 255, dtype=np.uint8)
    crack_class, confidence = clf.predict(white)
    assert isinstance(crack_class, str)
    assert isinstance(confidence, float)


def test_predict_with_random_image(clf):
    """Imagem aleatória (ruído) deve produzir resultado válido."""
    rng = np.random.default_rng(42)
    noisy = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    crack_class, confidence = clf.predict(noisy)
    assert crack_class in CLASSES
    assert 0.0 <= confidence <= 1.0


def test_predict_with_small_image(clf):
    """Imagem menor que 224x224 deve ser redimensionada corretamente."""
    small = np.zeros((50, 50, 3), dtype=np.uint8)
    crack_class, confidence = clf.predict(small)
    assert crack_class in CLASSES


def test_predict_batch(clf):
    """predict_batch() deve retornar resultado para cada imagem da lista."""
    images = [np.zeros((128, 128, 3), dtype=np.uint8)] * 3
    results = clf.predict_batch(images)
    assert len(results) == 3
    for crack_class, confidence in results:
        assert crack_class in CLASSES
        assert 0.0 <= confidence <= 1.0
