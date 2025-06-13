"""
Módulo unificado para procesamiento de imágenes
Contiene:
- Images: Carga/manipulación básica
- Kernel: Operaciones de convolución
- CommonKernels: Kernels predefinidos
"""
__version__ = "1.0.0"

from typing import List, Tuple, Optional, Union
from PIL import Image
import numpy as np
import os

class Images:
    """
    Clase para procesamiento de imágenes que soporta múltiples formatos (RGB, escala de grises, etc.)
    y tamaños variables. 
    """
    
    def __init__(self, img_path: str, target_size: Optional[Tuple[int, int]] = None, grayscale: bool = True):
        """
        Inicializador que carga y procesa la imagen automáticamente.
        
        Args:
            img_path: Ruta al archivo de imagen
            target_size: Tamaño objetivo (ancho, alto) o None para tamaño original
            grayscale: Si True, convierte a escala de grises (solo afecta carga inicial)
        """
        self.img_path = img_path
        self.target_size = target_size
        self.original_size = None  # Se establece durante la carga
        self.grayscale = grayscale
        self.matrix = self._Image_to_matrix()  # Matriz normalizada [0-1]

    def _check_dependencies(self) -> None:
        """
        Verificación dinámica de dependencias.
        Importa las librerías necesarias solo cuando se requieren.
        """
        try:
            global Image, os
            from PIL import Image
            import os
        except ImportError as e:
            raise ImportError(
                "Dependencias no encontradas. Instala con: pip install XXXXX"
            ) from e
        
    def _Image_to_matrix(self) -> List[List[Union[float, Tuple[float, float, float]]]]:
        """
        Método principal de carga y conversión de imágenes.
        
        Returns:
            Matriz normalizada donde cada elemento es:
            - float [0-1] para escala de grises
            - tuple (R,G,B) [0-1] para color
            
        Raises:
            FileNotFoundError: Si la imagen no existe
            RuntimeError: Para otros errores de procesamiento
        """
        self._check_dependencies()
        
        if not os.path.isfile(self.img_path):
            raise FileNotFoundError(f"Archivo no encontrado: {self.img_path}")
        
        try:
            with Image.open(self.img_path) as img:
                self.original_size = img.size
                
                # Redimensionamiento condicional
                if self.target_size is not None:
                    img = img.resize(self.target_size)
                    output_size = self.target_size
                else:
                    output_size = self.original_size
                
                # Conversión a modo de color
                img = img.convert('L' if self.grayscale else 'RGB')
                
                # Extracción y normalización de píxeles
                pixels = list(img.getdata())
                width, height = output_size
                
                # Construcción de matriz con manejo de tipos
                matrix = []
                for i in range(height):
                    row = pixels[i*width : (i+1)*width]
                    
                    if self.grayscale:
                        matrix.append([p/255.0 for p in row])
                    else:
                        matrix.append([(r/255.0, g/255.0, b/255.0) for (r,g,b) in row])
                
                # Verificación de matriz no vacía
                if not matrix or not any(matrix):
                    raise ValueError("La matriz generada está vacía")
                    
                return matrix
                
        except Exception as e:
            raise RuntimeError(f"Error procesando imagen: {str(e)}")

    def get_matrix(self, verbose: bool = False) -> list:
        """
        Obtiene matriz de la imagen.
        
        Args:
            verbose: Si True, muestra información de depuración
            
        Returns:
            Matriz de píxeles normalizados
        """
        if not hasattr(self, 'matrix'):
            raise AttributeError("Matriz de imagen no inicializada")
        
        if verbose:
            print(f"[Debug] Tipo matriz: {type(self.matrix)}")
            print(f"[Debug] Dimensiones: {len(self.matrix)}x{len(self.matrix[0]) if self.matrix else 0}")
        
        return [row.copy() for row in self.matrix]

    def create_from_matrix(self, matrix: List[List[Union[float, Tuple[float, float, float]]]]) -> 'Images':
        """
        Crea una nueva instancia de Images a partir de una matriz existente.
        
        Args:
            matrix: Matriz de píxeles normalizados [0-1]
            
        Returns:
            Nueva instancia de Images con la matriz proporcionada
        """
        new_img = Images(self.img_path)
        new_img.matrix = matrix
        new_img.original_size = (len(matrix[0]), len(matrix))  # (width, height)
        new_img.target_size = self.target_size
        
        # Detecta automáticamente el tipo de imagen
        first_pixel = matrix[0][0]
        new_img.grayscale = not isinstance(first_pixel, tuple)
        
        return new_img
    
    def show_img(self, display_size: Optional[Tuple[int, int]] = None, inline: Optional[bool] = None) -> None:
        """
        Muestra la imagen adaptándose al entorno de ejecución.
        
        Args:
            display_size: Tamaño de visualización (ancho, alto) o None para tamaño original
            inline: Forzar modo inline (True) o ventana (False). Si None, se detecta automáticamente
        """
        self._check_dependencies()
        from IPython.display import display  # Importación condicional
        
        pil_img = self._to_pil_image()
        
        if display_size is not None:
            pil_img = pil_img.resize(display_size, Image.NEAREST)
        
        # Detección automática del entorno si no se especifica

        if inline is None:
            try:
                from IPython import get_ipython
                get_ipython()
                inline = True
            except Exception:
                inline = False

        
        if inline:
            # Visualización en notebook
            display(pil_img)
        else:
            # Ventana emergente tradicional
            pil_img.show()

    def _to_pil_image(self) -> 'Image':
        """
        Conversor a objeto PIL.Image con manejo de valores fuera de rango para convoluciones
        
        Returns:
            Image: Objeto PIL.Image listo para visualización.
        Raises:
            ImportError: Si PIL no está disponible
            ValueError: Si los valores RGB están fuera del rango [0,1]
        """
        self._check_dependencies()  # Asegura que PIL está disponible
        
        # Determinar tamaño de salida
        output_size = self.target_size if self.target_size is not None else self.original_size
        
        # Detección automática del tipo de imagen
        first_pixel = self.matrix[0][0]
        is_grayscale = isinstance(first_pixel, (int, float))
        
        pixels = []
        for row in self.matrix:
            for pixel in row:
                if is_grayscale:
                    # Manejo especial para valores fuera de rango (resultado de convoluciones)
                    clamped_pixel = max(0.0, min(1.0, float(pixel)))
                    pixels.append(int(clamped_pixel * 255))
                else:
                    # Validación y conversión para RGB
                    if not all(0 <= c <= 1 for c in pixel):
                        raise ValueError(f"Valores RGB {pixel} fuera de rango [0,1]")
                    pixels.append(tuple(int(c * 255) for c in pixel))
        
        # Crear y retornar la imagen
        mode = 'L' if is_grayscale else 'RGB'
        img = Image.new(mode, output_size)
        img.putdata(pixels)
        return img
    

class Kernel:
    """
    Clase para aplicar operaciones de convolución robustas a matrices de imágenes.
    Soporta múltiples formatos (escala de grises, RGB) y maneja automáticamente los bordes.
    """
    
    def __init__(self, img_matrix, kernel_matrix):
        """
        Inicializa la convolución con validación automática.
        
        Args:
            img_matrix: Matriz de imagen (lista de listas de floats o tuplas RGB)
            kernel_matrix: Matriz del kernel (lista de listas de floats)
        """
        self._check_dependencies()
        self.img_matrix = self._prepare_matrix(img_matrix, 'Imagen')
        self.kernel_matrix = self._prepare_matrix(kernel_matrix, 'Kernel')
        self._validate_inputs()
        self.result = self._apply_convolution()

    def _check_dependencies(self):
        """Verifica e importa numpy dinámicamente."""
        try:
            global np
            import numpy as np
        except ImportError:
            raise ImportError("Se requiere numpy. Instale con: pip install numpy")

    def _prepare_matrix(self, matrix, name):
        """Convierte matriz a numpy array con validación básica."""
        try:
            arr = np.array(matrix, dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError(f"{name} debe ser matriz 2D, set grayscale=True")
            return arr
        except Exception as e:
            raise ValueError(f"{name} inválida: {str(e)}")

    def _validate_inputs(self):
        """Valida dimensiones y tipos de las matrices."""
        if self.kernel_matrix.shape[0] != self.kernel_matrix.shape[1]:
            raise ValueError("Kernel debe ser cuadrado")
        if self.kernel_matrix.shape[0] % 2 == 0:
            raise ValueError("Kernel debe tener tamaño impar (3x3, 5x5, etc.)")
        if self.img_matrix.size == 0:
            raise ValueError("Matriz de imagen vacía")

    def _apply_convolution(self):
        """Aplica convolución con manejo de bordes."""
        k_size = self.kernel_matrix.shape[0]
        pad = k_size // 2
        
        # Padding con reflexión para mejores bordes
        padded = np.pad(self.img_matrix, pad, mode='reflect')
        output = np.zeros_like(self.img_matrix)
        
        # Convolución 
        for i in range(pad, padded.shape[0]-pad):
            for j in range(pad, padded.shape[1]-pad):
                region = padded[i-pad:i+pad+1, j-pad:j+pad+1]
                output[i-pad,j-pad] = np.sum(region * self.kernel_matrix)
        
        return output.tolist()

    def get_result(self):
        """Devuelve el resultado de la convolución como lista de listas."""
        return self.result

    @staticmethod
    def normalize_kernel(kernel):
        """Normaliza el kernel para preservar el rango dinámico."""
        try:
            kernel_sum = sum(sum(row) for row in kernel)
            return [[val/kernel_sum for val in row] for row in kernel]
        except:
            return kernel  # Si no se puede normalizar, retorna original
        
class CommonKernels:
    @staticmethod
    def blur(size=3):
        """Kernel de desenfoque normalizado"""
        kernel = [[1/(size**2) for _ in range(size)] for _ in range(size)]
        return kernel

    @staticmethod
    def edge_detection():
        """Kernel para detectar bordes"""
        return [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]

    @staticmethod
    def sharpen():
        """Kernel para enfocar"""
        return [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]