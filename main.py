import math
import numpy as np
import cv2
import tf as tf
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
    img = imread('test2.jpg', plugin='pil')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    """отрисовка исходника"""
    fig = plt.figure(figsize=(20, 10))
    fig.add_subplot(2, 3, 1)
    title('Исходное изображение(rgb)')
    plt.imshow(img)

    """реализация первого шага алгоритма"""
    # T_YIQ = transformRGB2YIQ(img)
    # fig.add_subplot(2, 3, 2)
    # title('После преобразования цвета(yiq)')
    # imshow(T_YIQ)

    """реализация второго шага алгоритма"""
    k = 60
    Teta_k = (3.14 * k) / 180
    T_k = getting_matrix_Tk(Teta_k)

    """реализация третьего шага алгоритма"""
    #____________________________________________________________________
    # T_YIQ_inv = pinv(T_YIQ)

    # T_YIQ_ = T_YIQ.reshape(-1, 3)
    # T_YIQ_inv_ = T_YIQ_inv.reshape(-1, 3)
    # g_k = np.dot(T_YIQ_, T_k, T_YIQ_inv_)
    #________________________________________________________________________
    T_YIQ_arr = np.array([[0.299, 0.587, 0.114],
                [0.596, -0.275, -0.321],
                [0.212, -0.523, 0.311]])
    T_YIQ_arr_inv = inv(T_YIQ_arr)

    g_k_arr = np.dot(T_YIQ_arr, T_k, T_YIQ_arr_inv)
    # print(g_k)

    """проверка развертывания изображния"""
    # fig.add_subplot(2, 3, 3)
    img_ = img
    # img_v = img_.reshape(img.shape)
    # imshow(img_v)

    """реализация четвертого пункта алгоритма"""
    m, n, q = img.shape
    # v = np.empty((m,n,q))
    fig.add_subplot(2, 3, 2)
    title('v1 = g_k * v_transpose')
    img = img.reshape(-1,3)
    m, n = img.shape
    for i in range(0, m):
            img[i] = img[i].transpose()
    v = np.dot(img, g_k_arr).reshape(img_.shape)
    imshow(v)

    # v_result = tf.image.yiq_to_rgb(v)
    # v_result = transformYIQ2RGB(v)
    # v_result = np.empty((m, n, q))
    # v_result = np.dot(v.reshape(-1, 3), T_YIQ_arr_inv.transpose()).reshape(img_.shape)
    # fig.add_subplot(2, 3, 3)
    # title('YIQ - > RGB')
    # for i in range(0, m):
    #     for j in range(0, n):
    #         v_result[i][j] = np.dot(v[i][j], inv(T_YIQ_arr).transpose())
    #
    # imshow(v_result)

    """попытка вычленить массивы, отвечающие за цвет"""
    # p = np.zeros((m, n, q))
    # p = img.reshape(1, 3)
    # k = 0
    # for i in range(0, n):
    #     img_[n][0] = img_[n][0].transpose()
    #     k += 1
    #
    # print(p)
    # n, q = np.shape(img_)
    # im1 = Image.open('test3.jpg')
    # imag = im1.convert('RGB')
    # X, Y = 0, 0
    # pixelRGB = imag.getpixel((X, Y))
    # R, G, B = pixelRGB
    # r1, g1, b1 = im1.split()
    # print(R, G, B)

    ## еще один способ перехода из rgb -> yiq
    # I = imread('test.jpg');
    # imshow(I);
    # J = rgb2ntsc(I);
    # imshow(J);

    show()
