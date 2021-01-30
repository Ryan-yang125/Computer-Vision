import numpy as np
from cv2 import cv2
import random


class AnimateDraw:
    @staticmethod
    def drawPacman(img, pacmanBodyCenter):
        cv2.ellipse(img, pacmanBodyCenter, (80, 80),
                    0, 30, 330, (0, 255, 255), -1)
        pacmanEyeCenter = (110, 200)
        cv2.circle(img, pacmanEyeCenter, 12, (255, 255, 255), -1)

    @staticmethod
    def drawBeans(img, beanHeadCenter, bodyColor):
        # Draw head
        bodySize = (25, 22)
        cv2.ellipse(img, beanHeadCenter, bodySize,
                    180, 0, 180, bodyColor, -1)
        # Draw eyes
        eyeLeft = (beanHeadCenter[0]-8, beanHeadCenter[1]-5)
        eyeRight = (beanHeadCenter[0]+8, beanHeadCenter[1]-5)
        cv2.circle(img, eyeLeft, 3, (255, 255, 255), -1)
        cv2.circle(img, eyeRight, 3, (255, 255, 255), -1)
        # Draw body
        bodyLeft = [beanHeadCenter[0]-bodySize[0], beanHeadCenter[1]]
        bodyRight = [beanHeadCenter[0]+bodySize[0], beanHeadCenter[1]]
        bodyLength = 50
        pts = np.array(
            [bodyLeft, [bodyLeft[0]-2, bodyLeft[1]+bodyLength],
             [bodyLeft[0]+7, bodyRight[1]+bodyLength-10],
             [bodyLeft[0]+16, bodyRight[1]+bodyLength],
             [bodyLeft[0]+25, bodyRight[1]+bodyLength-10],
             [bodyRight[0]-16, bodyRight[1]+bodyLength],
             [bodyRight[0]-7, bodyRight[1]+bodyLength-10],
             [bodyRight[0]+2, bodyRight[1]+bodyLength], bodyRight], np.int32)
        # 顶点个数：9，矩阵变成9*1*2维
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(img, [pts], bodyColor)

    @staticmethod
    def drawAnimate(video, fps):
        # Create a black image
        img = np.zeros((640, 960, 3), np.uint8)
        # Draw pacman
        pacmanBodyCenter = (100, 256)
        AnimateDraw.drawPacman(img, pacmanBodyCenter)
        for _ in range(0, fps):
            video.write(img)
        # Draw beans
        colors = (
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 0, 255),
            (255, 255, 0),
            (0, 255, 255),
        )
        for i in range(0, 7*fps):
            if(i % 30 == 0 and i < 180):
                AnimateDraw.drawBeans(img, (250+i*3, 256), colors[int(i/30)])
            video.write(img)
            cv2.imshow('image', img)
            if cv2.waitKey(30) & 0xFF == ord(' '):
                cv2.waitKey(0)
        cv2.line(img, pacmanBodyCenter, (250+450, 256), (0, 0, 255), 5)
        for _ in range(0, fps):
            video.write(img)
            cv2.imshow('image', img)
            if cv2.waitKey(30) & 0xFF == ord(' '):
                cv2.waitKey(0)
        return img


def run():
    # 读取图片
    inPath = './img/zju.jpg'
    outPath = './output.mp4'
    image = cv2.imread(inPath)
    if image is None:
        print("The image is null")
        image = np.zeros((640, 960, 3), np.uint8)
    # 初始化视频
    rows = 640
    cols = 960
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(outPath, fourcc, fps,
                            (cols, rows))  # (cols,rows)
    # 读入图片ZJU
    image = cv2.resize(image, (960, 640))
    for _ in range(1, 2*fps):
        video.write(image)
        cv2.imshow('image', image)
        if cv2.waitKey(30) & 0xFF == ord(' '):
            cv2.waitKey(0)
    # [1,0,x],[0,1,y]
    M = np.float32([[1, 0, 0], [0, 1, +5]])
    for _ in range(1, 4*fps):
        video.write(image)
        image = cv2.warpAffine(image, M, (cols, rows))
        cv2.imshow('image', image)
        if cv2.waitKey(30) & 0xFF == ord(' '):
            cv2.waitKey(0)
    # Read img person and move to center
    inPath = './img/person.jpg'
    image = cv2.imread(inPath)
    if image is None:
        print("The image is null")
        image = np.zeros((640, 960, 3), np.uint8)
    M = np.float32([[1, 0, 180], [0, 1, 0]])
    image = cv2.warpAffine(image, M, (cols, rows))
    for _ in range(1, fps):
        video.write(image)
        cv2.imshow('image', image)
        if cv2.waitKey(30) & 0xFF == ord(' '):
            cv2.waitKey(0)
    # Add some word
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, 'YangRui 3180101941', (100, 550), font,
                2, (255, 255, 255), 3, lineType=cv2.LINE_AA)
    for _ in range(1, 2*fps):
        video.write(image)
        cv2.imshow('image', image)
        if cv2.waitKey(30) & 0xFF == ord(' '):
            cv2.waitKey(0)
    M = np.float32([[1, 0, -5], [0, 1, 0]])
    for _ in range(1, 6*fps):
        video.write(image)
        image = cv2.warpAffine(image, M, (cols, rows))
        cv2.imshow('image', image)
        if cv2.waitKey(30) & 0xFF == ord(' '):
            cv2.waitKey(0)
    # Add pacman animate
    image = AnimateDraw.drawAnimate(video, fps)
    # Add ending
    W = 960
    W3 = W*3
    H = 640
    H3 = H*3
    delay = 30
    # Random ray
    for _ in range(0, 3*fps):
        pt1 = (int(random.random()*W3-W), int(random.random()*H3-H))
        pt2 = (int(random.random()*W3-W), int(random.random()*H3-H))
        pt3 = (int(random.random()*W3-W), int(random.random()*H3-H))
        pt4 = (int(random.random()*W3-W), int(random.random()*H3-H))
        pt5 = (int(random.random()*W3-W), int(random.random()*H3-H))
        pt6 = (int(random.random()*W3-W), int(random.random()*H3-H))
        pts = np.array([pt1, pt2, pt3, pt4, pt5, pt6], np.int32)
        pts = pts.reshape((-1, 1, 2))
        color = (random.randint(0, 255), random.randint(
            0, 255), random.randint(0, 255))
        cv2.polylines(image, [pts], True, color, 10, 0)
        video.write(image)
        cv2.imshow('image', image)
        if cv2.waitKey(delay) & 0xFF == ord(' '):
            cv2.waitKey(0)
    cv2.putText(image, 'Thanks', (350, 300), font,
                2, (255, 255, 255), 3, lineType=cv2.LINE_AA)
    for _ in range(0, 2*fps):
        video.write(image)
        cv2.imshow('image', image)
        if cv2.waitKey(delay) & 0xFF == ord(' '):
            cv2.waitKey(0)
    # release
    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
