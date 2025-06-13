# Image Processing with Convolution Kernels

![Example Pipeline](https://via.placeholder.com/500x200?text=Original+%E2%86%92+Sharpen+%E2%86%92+Edge+Detection+%E2%86%92+Blur)

A Python project demonstrating image processing using common convolution kernels for sharpening, edge detection, and blurring effects.

## Features

- Load and prepare images (grayscale conversion, resizing)
- Apply multiple convolution kernels sequentially
- Supported operations:
  - Sharpening (enhance image details)
  - Edge detection (highlight boundaries)
  - Blurring (smoothing effect)
- Visualize each processing step
- Matrix inspection for debugging

## Example Code

```python
# 1. Cargar y preparar imagen original
img = Images("2.jpg", target_size=(500, 500), grayscale=True)
print("\n=== Imagen Original ===")
img.show_img((500, 500))  # Mostrar ampliada
original_matrix = img.get_matrix(verbose=True)

# 2. Aplicar primer kernel (sharpen)
print("\n=== Aplicando Kernel Sharpen ===")
kernel_sharpen = CommonKernels.sharpen()
convoluted_sharp = Kernel(img_matrix=original_matrix, kernel_matrix=kernel_sharpen).get_result()

# Crear y mostrar imagen sharpen
img_sharp = img.create_from_matrix(convoluted_sharp)
img_sharp.show_img((500, 500))
print("Matriz después de sharpen:", [row[:3] for row in convoluted_sharp[:3]])  # Muestra porción

# 3. Aplicar segundo kernel (edge detection) SOBRE EL RESULTADO ANTERIOR
print("\n=== Aplicando Kernel Edge Detection ===")
kernel_edge = CommonKernels.edge_detection()
convoluted_edge = Kernel(img_matrix=convoluted_sharp, kernel_matrix=kernel_edge).get_result()

# Crear y mostrar imagen con bordes
img_edge = img.create_from_matrix(convoluted_edge)
img_edge.show_img((500, 500))
print("Matriz después de edge detection:", [row[:3] for row in convoluted_edge[:3]])

# 4. Opcional: Aplicar desenfoque después (para ver efecto combinado)
print("\n=== Aplicando Kernel Blur ===")
kernel_blur = CommonKernels.blur(size=5)
convoluted_blur = Kernel(img_matrix=convoluted_edge, kernel_matrix=kernel_blur).get_result()

# Crear y mostrar imagen final
img_final = img.create_from_matrix(convoluted_blur)
img_final.show_img((500, 500))
print("Matriz final después de blur:", [row[:3] for row in convoluted_blur[:3]])
