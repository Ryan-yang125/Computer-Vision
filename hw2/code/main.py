from cv2 import cv2
import numpy as np
import math
import time

# aggregate thershold


def get_thershold(hough_space, width, height, x, y, space):
    thershold = hough_space[x][y]
    if x - space > 0 and x + space < width and y - space > 0 and y + space < height:
        tmp = []
        for i in range(x-space, x + space):
            tmp.append(max(hough_space[i][y-space:y+space]))
        thershold = max(tmp)
    return thershold


def line_detecting(path, thershold, rMin, rMax):
    # canny
    cannyMin = 200
    cannyMax = 400
    image = cv2.imread(path)
    # to gray
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # median blur
    gauss = cv2.GaussianBlur(gray, (5, 5), 0)
    # median = cv2.medianBlur(gray, 5)
    # edges detection
    edges = cv2.Canny(gauss, cannyMin, cannyMax)
    cv2.imwrite('./results/line_edges_'+str(time.time())+'_.bmp', edges)
    cv2.imshow('img-canny', edges)
    # Copy edges to the images that will display the results in BGR

    # Hough Space initialize
    height = edges.shape[0]
    width = edges.shape[1]
    rMax = int(math.hypot(height, width))
    thetaMax = 360
    points = [[[] for k in range(thetaMax)]for theta in range(rMax)]
    hough_space = [[0 for k in range(thetaMax)] for theta in range(rMax)]
    # vote
    for x in range(width):
        for y in range(height):
            if edges[y][x] == 0:
                continue
            for theta in range(0, thetaMax-1):
                r = int(x*math.cos(theta)+y*math.sin(theta))
                if rMin < r < rMax:
                    hough_space[r][theta] += 1
                    points[r][theta].append((x, y))
    # find max
    for r in range(rMax):
        for theta in range(thetaMax):
            if hough_space[r][theta] >= thershold:
                if hough_space[r][theta] == get_thershold(hough_space, thetaMax, rMax, r, theta, 20):
                    k = (-math.cos(theta)/math.sin(theta))
                    b = (r/math.sin(theta))
                    points[r][theta].sort()
                    x0 = points[r][theta][0][0]
                    x1 = points[r][theta][-1][0]
                    y0 = int(x0*k+b)
                    y1 = int(x1*k+b)
                    cv2.line(image, (x0, y0), (x1, y1),
                             (0, 0, 255), thickness=2)
    cv2.imshow("line-detection", image)
    cv2.imwrite("results/line_detection_" + str(time.time()) + "_.bmp", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def circle_detecting(path, threshold, minRadius, maxRadius):
    # read img
    img = cv2.imread(path)
    size = img.shape
    height = size[0]
    width = size[1]
    # to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # filter
    gauss = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gauss, 200, 400, 3)
    cv2.imshow("canny_edge.bmp", edges)
    cv2.imwrite('./results/circle_edges_'+str(time.time()) + '_.bmp', edges)
    # sobel
    sobel_dx = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=5)
    sobel_dy = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=5)

    # initialize
    accumulator = [[[] for j in range(height)] for i in range(width)]
    hough = [[0 for j in range(height)] for i in range(width)]

    # loop to vote
    for x in range(width):
        for y in range(height):
            if edges[y][x] != 0:
                if sobel_dx[y][x] != 0:
                    # compute theta
                    tan_theta = sobel_dy[y][x] / sobel_dx[y][x]
                    # circle center
                    for a in range(0, width):
                        b = int(a * tan_theta - x * tan_theta + y)
                        if 0 < b < height:
                            if minRadius < math.hypot(a-x, b-y) < maxRadius:
                                hough[a][b] += 1
                                accumulator[a][b].append((x, y))

                if sobel_dx[y][x] == 0 and sobel_dy[y][x] != 0:
                    if sobel_dy[y][x] > 0:
                        for b in range(y+minRadius, y+maxRadius):
                            # circle center in y
                            if 0 <= b < height:
                                # vote
                                hough[x][b] += 1
                                accumulator[x][b].append((x, y))

                    else:
                        for b in range(y-maxRadius, y-minRadius):
                            if 0 <= b < height:
                                # vote
                                hough[x][b] += 1
                                accumulator[x][b].append((x, y))

    for a in range(0, width):
        for b in range(0, height):
            if hough[a][b] > threshold:
                maxth = get_thershold(hough, width, height, a, b, 50)
                if hough[a][b] == maxth:
                    list = []
                    area_count = [0, 0, 0, 0]
                    for i in range(0, len(accumulator[a][b])):
                        dx = accumulator[a][b][i][0] - a
                        dy = accumulator[a][b][i][1] - b
                        radius = int(math.hypot(dx, dy))
                        # get the points of four spaces in one circle
                        if dx >= 0 and dy >= 0:
                            area_count[0] += 1
                        if dx >= 0 and dy < 0:
                            area_count[1] += 1
                        if dx < 0 and dy >= 0:
                            area_count[2] += 1
                        if dx < 0 and dy < 0:
                            area_count[3] += 1
                        list.append(radius)

                    if(len(list) > 0) and area_count[0] != 0 and area_count[1] != 0 and area_count[2] != 0 and area_count[3] != 0 and max(area_count)-min(area_count) < 10:
                        radius = list[len(list)//2]
                        if minRadius < radius < maxRadius:
                            cv2.circle(img, center=(a, b), radius=radius,
                                       color=(0, 0, 255), thickness=2)
                            for k in accumulator[a][b]:
                                cv2.circle(img, k, 4, (255, 0, 0), 1, 8, 0)
    cv2.imshow("circle-detection", img)
    cv2.imwrite("results/circle_"+str(time.time())+"_.bmp", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    line_img = './assets/hw-highway.jpg'
    circle_img = './assets/hw-coin.jpg'
    img = './assets/hw-seal.jpg'
    # line_detecting(line_img, 120, 30, 500)
    # circle_detecting(circle_img, 8, 30, 70)
    # line_detecting(img, 120, 30, 500)
    circle_detecting(img, 18, 50, 200)
