import math
import numpy as np
import cv2
from numpy.linalg import pinv, inv
from matplotlib import pyplot as plt
from matplotlib.pyplot import title
from skimage.io import imread, imshow, show
from PIL import Image


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    yiq_from_rgb = np.array([[0.299, 0.587, 0.114],
                             [0.59590059, -0.27455667, -0.32134392],
                             [0.21153661, -0.52273617, 0.31119955]])
    OrigShape = imgRGB.shape
    Matrix_YIQ = np.dot(imgRGB.reshape(-1, 3), yiq_from_rgb.transpose()).reshape(OrigShape)
    return Matrix_YIQ


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    yiq_from_rgb = np.array([[0.299, 0.587, 0.114],
                [0.596, -0.275, -0.321],
                [0.212, -0.523, 0.311]])
    OrigShape = imgYIQ.shape
    return np.dot(imgYIQ.reshape(-1, 3), inv(yiq_from_rgb).transpose()).reshape(OrigShape)


def getting_matrix_Tk(Teta_k):
    result = np.array([[1, 0, 0],
                       [0, math.cos(Teta_k), -math.sin(Teta_k)],
                       [0, math.sin(Teta_k), math.cos(Teta_k)]])
    return result


if __name__ == "__main__":
    k = 60

    img = imread('test3.jpg', plugin='pil')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    """отрисовка исходника"""

    fig = plt.figure(figsize=(20, 10))
    fig.add_subplot(2, 3, 1)
    title('Исходное изображение(rgb)')
    plt.imshow(img)

    # T_YIQ = transformRGB2YIQ(img)
    # fig.add_subplot(2, 3, 2)
    # title('После преобразования цвета(yiq)')
    # imshow(T_YIQ)

    """реализация первого шага алгоритма"""

    Teta_k = (3.14 * k) / 180
    T_k = getting_matrix_Tk(Teta_k)
    T_YIQ_arr = np.array([[0.299, 0.587, 0.114],
                [0.596, -0.275, -0.321],
                [0.212, -0.523, 0.311]])
    T_YIQ_arr_inv = inv(T_YIQ_arr)

    g_k_arr = np.dot(T_YIQ_arr, T_k, T_YIQ_arr_inv)
    # print(g_k)
    img_ = img

    """реализация второго пункта алгоритма"""

    m, n, q = img.shape
    fig.add_subplot(2, 3, 2)
    title('v1 = g_k * v_transpose')
    img = img.reshape(-1,3)
    m, n = img.shape
    for i in range(0, m):
            img[i] = img[i].transpose()
    v = np.dot(img, g_k_arr).reshape(img_.shape)
    imshow(v)

    """наложение"""

    img_ = np.asarray(img_, np.float64)
    result = cv2.addWeighted(img_, 1, v, 0.000001, -165)
    fig.add_subplot(2, 3, 3)
    title('исходное I c водяным знаком V с подписью k')
    imshow(result)

    show()
