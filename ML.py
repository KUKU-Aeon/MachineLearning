from tensorflow import keras
from keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings

'''------------------------------------------------------------------------------------------------------------------'''

# Функция для показа изображения
def show_image(tensor):
    plt.figure(figsize=(6, 6)) # Размер рисунка (ширина и высота) в дюймах
    plt.imshow(tf.squeeze(tensor)) # tf.squeeze() removes dimensions of size 1 from the shape of a tensor; plt.imshow() displays data as an image.
    plt.axis('off') # Hide all axis decorations, i.e. axis labels, spines, tick marks, tick labels, and grid lines
    plt.show()

'''
Слой свертки (Convolutional Layer)
Слой свертки является основным строительным блоком сверточной нейронной сети.
Он состоит из набора фильтров (ядер), которые применяются к входным данным для выделения различных признаков.
Каждый фильтр проходит через входные данные, выполняя операцию свертки, которая вычисляет скалярное произведение
между фильтром и соответствующей областью входных данных. Результатом операции свертки является карта признаков,
которая представляет собой активации фильтров на различных местоположениях входных данных.

Слой активации (Activation Layer)
Слой активации применяет нелинейную функцию активации к выходам слоя свертки.
Это позволяет модели улавливать более сложные зависимости между признаками и делает ее более гибкой в аппроксимации сложных функций.

Слой пулинга (Pooling Layer)
Слой пулинга уменьшает размерность карты признаков, удаляя избыточную информацию и улучшая инвариантность к малым изменениям входных данных.
Самый распространенный тип пулинга – это операция максимального пулинга, которая выбирает максимальное значение из каждой области карты признаков.
'''

# Функция свертки
def convolution(tensor, kernel):
    return tf.nn.conv2d(
        input=tensor,
        filters=kernel,
        strides=1,
        padding='SAME'
    )

# Функция пулинга
def pooling(tensor):
    return tf.nn.pool(
        input=tensor, # предобработанное изображение
        window_shape=(2, 2),
        pooling_type='MAX',
        strides=(2, 2),
        padding='SAME',
    )

# Тестирование переданного фильтра на свертку, активацию и пулинг
def CAP_kernel_test(image, kernel):
    # Свертка
    image_filter = convolution(image, kernel)
    show_image(image_filter)

    # Активация
    image_detect = tf.nn.relu(image_filter)
    show_image(image_detect)

    # Пулинг
    image_condense = pooling(image_detect)
    show_image(image_condense)

# Тестирование ядра, реагирующего на вертикальные линии
def kernel_vert_lines_test(image):
    kernel_vert_lines = tf.constant([
        [-1, 2, -1],
        [-1, 2, -1],
        [-1, 2, -1],
    ], dtype=tf.float32)
    kernel_vert_lines = tf.reshape(kernel_vert_lines, [*kernel_vert_lines.shape, 1, 1])
    CAP_kernel_test(image, kernel_vert_lines)

# Тестирование ядра, реагирующего на контуры
def kernel_contours_test(image):
    kernel_contours = tf.constant([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1],
    ], dtype=tf.float32)
    kernel_contours = tf.reshape(kernel_contours, [*kernel_contours.shape, 1, 1])
    CAP_kernel_test(image, kernel_contours)

# Тестирование ядра, реагирующего на горизонтальные линии
def kernel_horiz_lines_test(image):
    kernel_horiz_lines = tf.constant([
        [-1, -1, -1],
        [ 2,  2,  2],
        [-1, -1, -1],
    ], dtype=tf.float32)
    kernel_horiz_lines = tf.reshape(kernel_horiz_lines, [*kernel_horiz_lines.shape, 1, 1])
    CAP_kernel_test(image, kernel_horiz_lines)
    
'''------------------------------------------------------------------------------------------------------------------'''

plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')
warnings.filterwarnings("ignore")

# чтение изображения
image_path = 'palace.jpg'
image = tf.io.read_file(image_path)
image = tf.io.decode_jpeg(image, channels = 1) # Decode a JPEG-encoded image to a uint8 tensor.
                                               # channels указывает желаемое количество цветовых каналов для декодированного изображения.
                                               # 1: вывод изображения в оттенках серого.

# изменение формата изображения
image = tf.image.convert_image_dtype(image, dtype=tf.float32) # Convert image to dtype, scaling its values if needed.
image = tf.expand_dims(image, axis=0)

print("Проверка фильтра, реагирующего на вертикальные линии")
kernel_vert_lines_test(image)
print("Проверка фильтра, реагирующего на контуры")
kernel_contours_test(image)
print("Проверка фильтра, реагирующего на горизонтальные линии")
kernel_horiz_lines_test(image)

