from cv2 import cv2
import numpy as np
from numpy import linalg

# Gaussian convolution weights


def Gaussian(p):
    Gaussian_core = []
    for x in range(-1, 2):
        for y in range(-1, 2):
            Gaussian_core.append(1./(2*np.pi*(p**2)) *
                                 np.e**(-(x**2+y**2)/(2*(p**2))))
    return Gaussian_core

# Non-maximum suppression


def non_maximum_suppression(R, x, y, r_min, width, height):
    for i in range(max(0, x-r_min), min(x+r_min, width)):
        for j in range(max(0, y-r_min), min(y+r_min, height)):
            if R[i*height+j] > R[x*height+y]:
                return False
    return True

# Harris corner detection


def harris_detect(img, size):
    # intialize image
    height, width, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sobel operator calculates the gradient
    x_sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    y_sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

    # intialize to store medium reusult
    max_show = np.zeros((height-2, width-2, 3), np.uint8)
    min_show = np.zeros((height-2, width-2, 3), np.uint8)
    R_show = np.zeros((height-2, width-2, 3), np.uint8)

    max_list = []
    min_list = []
    R_list = []
    k = 0.05  # Constant coefficient
    Gaussian_core = Gaussian((size-1)/6)

    for x in range(1, width-1):
        for y in range(1, height-1):
            M = np.mat([[0., 0.], [0., 0.]])
            for i in range(x-1, x+2):
                for j in range(y-1, y+2):
                    M += Gaussian_core[(i-(x-1))*(j-(y-1))+(j-(y-1))]*np.mat(
                        [[(x_sobel[j][i])**2, x_sobel[j][i]*y_sobel[j][i]], [x_sobel[j][i]*y_sobel[j][i], (y_sobel[y][x])**2]])
            lamda, _ = linalg.eig(M)
            lamda1 = max(lamda)
            lamda2 = min(lamda)
            max_list.append(lamda1)
            min_list.append(lamda2)
            R_list.append(lamda1*lamda2-k*(lamda1+lamda2)**2)

    normalize_1 = max(max_list)
    normalize_2 = max(min_list)
    normalize_R = max(R_list)

    for x in range(0, width-2):
        for y in range(0, height-2):
            record1 = int((max_list[x * (height - 2) + y] / normalize_1) * 255)
            record2 = int((min_list[x * (height - 2) + y] / normalize_2) * 255)
            record_R = int((R_list[x * (height - 2) + y] / normalize_R) * 255)
            max_show[y][x] = (record1, record1, record1)
            min_show[y][x] = (record2, record2, record2)
            if(R_list[x * (height - 2) + y] > 0.005*normalize_R and non_maximum_suppression(R_list, x, y, 3, width-2, height-2)):
                cv2.circle(img, (x+1, y+1), 1, (0, 0, 255), -1)
            R_show[y][x] = (record_R, record_R, record_R)

    cv2.imshow("max", max_show)
    cv2.imwrite('./img/max_lamda.png', max_show)
    cv2.imshow("min", min_show)
    cv2.imwrite('./img/min_lamda.png', min_show)
    cv2.imshow("r", R_show)
    cv2.imwrite('./img/R.png', R_show)

    cv2.imshow('Harris', img)
    cv2.imwrite('./img/Harris.png', img)
    cv2.waitKey(0)

# main


def run():
    cap = cv2.VideoCapture(0)

    while(1):
        ret, frame = cap.read()
        if ret is None:
            break
        cv2.namedWindow("capture", 0)
        cv2.imshow("capture", frame)
        if cv2.waitKey(1) & 0xFF == 32:
            cv2.imwrite('./img/frame.png', frame)
            harris_detect(frame, 3)
            cv2.waitKey(0)
    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
