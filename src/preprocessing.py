import cv2

class CrackPreprocessor:
    def __init__(self):
        pass

    def process(self, image_path):
        """
        Reads image and applies pre-processing.
        Returns the original BGR image and the enhanced grayscale image.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Failed to read image at {image_path}")
        
        # 1. Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Bilateral Filter instead of Gaussian. 
        # Bilateral preserves sharp edges (like fine cracks) while blurring flat noise (like concrete stains).
        blurred = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)
        
        # O CLAHE foi removido porque transformava sombras naturais (como a do canto esquerdo da imagem)
        # em alto-contraste extremo, forçando o Frangi a detectar ruído como fissura.
        # A fissura fina já tem constraste natural suficiente para o filtro Frangi.
        enhanced = blurred
        
        return img, enhanced
