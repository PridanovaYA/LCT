import math
import numpy as np
import cv2
from numpy.linalg import pinv
from matplotlib import pyplot as plt
from matplotlib.pyplot import title
from skimage.io import imread, imshow, show


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    yiq_from_rgb = np.array([[0.299, 0.587, 0.114],
                             [0.59590059, -0.27455667, -0.32134392],
                             [0.21153661, -0.52273617, 0.31119955]])
    OrigShape = imgRGB.shape
    tmp = imgRGB.reshape(-1, 3)
    Matrix_YIQ = np.dot(imgRGB.reshape(-1, 3), yiq_from_rgb.transpose()).reshape(OrigShape)
    return Matrix_YIQ

    pass


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    yiq_from_rgb = np.array([[0.299, 0.587, 0.114],
                             [0.59590059, -0.27455667, -0.32134392],
                             [0.21153661, -0.52273617, 0.31119955]])
    OrigShape = imgYIQ.shape
    return np.dot(imgYIQ.reshape(-1, 3), np.linalg.inv(yiq_from_rgb).transpose()).reshape(OrigShape)

    pass


def getting_matrix_Tk(Teta_k):
    result = np.array([[1, 0, 0],
                       [0, math.cos(Teta_k), -math.sin(Teta_k)],
                       [0, math.sin(Teta_k), math.cos(Teta_k)]])
    return result


if __name__ == "__main__":
    img = cv2.imread('test.jpg')
    #print(img[0][0])
    #print(img[0][0].transpose())
    fig = plt.figure(figsize=(20, 10))
    fig.add_subplot(2, 3, 1)
    # title('Исходное изображение(rgb)')
    # imshow(img)
    k = 60

    T_YIQ = transformRGB2YIQ(img)

    T_YIQ_inv = pinv(T_YIQ)

    Teta_k = (math.pi * k) / 180

    T_k = getting_matrix_Tk(Teta_k)

    T_YIQ_ = T_YIQ.reshape(-1, 3)
    T_YIQ_inv_ = T_YIQ_inv.reshape(-1, 3)

    g_k = np.dot(T_YIQ_, T_k, T_YIQ_inv_)
    print(g_k)

    fig.add_subplot(2, 3, 2)
    # title('После преобразования цвета(yiq)')
    # imshow(T_YIQ)
    # show()
    img_ = img.reshape(-1, 3)
    imshow(img_)

    show()
    m,n,q = np.shape(img)
    p = np.zeros((m, n, q))
    # for i in range(0,m):
    #     for j in range(0,n):


    print()

    # I = imread('test.jpg');
    # imshow(I);
    # J = rgb2ntsc(I);
    # imshow(J);