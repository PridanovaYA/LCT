from skimage.io import imread, imshow, show


if __name__ == "__main__":
    img = imread('test3.jpg', plugin='pil')

    print(img.mode)
    print('_______________________________________________________')
    print(img.shape)
    print('_______________________________________________________')
    print(img[0].shape)
    print('_______________________________________________________')
    print(img[1].shape)
    print('_______________________________________________________')
    print(img[1])
    print('_______________________________________________________')