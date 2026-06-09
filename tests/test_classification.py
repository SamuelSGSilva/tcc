import pytest
from src.classification import CriticalityClassifier

def test_classifier_initialization():
    classifier = CriticalityClassifier(scale_factor=0.1)
    assert classifier.scale_factor == 0.1
    
    # Check default thresholds
    assert classifier.THRESH_CRITICA == 1.00
    assert classifier.THRESH_SEMI_CRITICA_MAX == 1.00
    assert classifier.THRESH_SEMI_CRITICA_MIN == 0.20
    assert classifier.THRESH_NAO_CRITICA_MAX == 0.20
    assert classifier.THRESH_NAO_CRITICA_MIN == 0.01

def test_classifier_descartada():
    classifier = CriticalityClassifier(scale_factor=0.1)
    # Less than 0.01mm (0.1 pixel width)
    crack_class, width_mm = classifier.classify(0.05)
    assert crack_class == "DESCARTADA"
    assert width_mm == pytest.approx(0.005)

def test_classifier_nao_critica():
    classifier = CriticalityClassifier(scale_factor=0.1)
    # Between 0.01mm and 0.20mm
    crack_class, width_mm = classifier.classify(1.0) # 0.1 mm
    assert crack_class == "NAO_CRITICA"
    assert width_mm == pytest.approx(0.10)

    # Just below the NAO_CRITICA upper boundary (0.19mm < 0.20mm)
    crack_class, width_mm = classifier.classify(1.9) # 0.19 mm
    assert crack_class == "NAO_CRITICA"
    assert width_mm == pytest.approx(0.19)

def test_classifier_semi_critica():
    classifier = CriticalityClassifier(scale_factor=0.1)
    # Between 0.30mm and 1.00mm
    crack_class, width_mm = classifier.classify(5.0) # 0.50 mm
    assert crack_class == "SEMI_CRITICA"
    assert width_mm == pytest.approx(0.5)

    # Exactly on max boundary of SEMI_CRITICA
    crack_class, width_mm = classifier.classify(10.0) # 1.00 mm
    assert crack_class == "SEMI_CRITICA"
    assert width_mm == pytest.approx(1.0)
    
    # Well above the SEMI_CRITICA lower boundary (0.31mm >= 0.20mm)
    crack_class, width_mm = classifier.classify(3.1) # 0.31 mm
    assert crack_class == "SEMI_CRITICA"
    assert width_mm == pytest.approx(0.31)

def test_classifier_critica():
    classifier = CriticalityClassifier(scale_factor=0.1)
    # Greater than 1.00mm
    crack_class, width_mm = classifier.classify(20.0)
    assert crack_class == "CRITICA"
    assert width_mm == pytest.approx(2.0)
