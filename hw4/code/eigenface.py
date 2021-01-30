'''
Author: Yang Rui
Date: 2020-12-24 13:52:22
LastEditTime: 2020-12-29 15:25:53
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /code/eigenface.py
'''

import sys

import click
import matplotlib.pyplot as plt
import numpy as np
from cv2 import cv2


class EigenFace(object):
    @staticmethod
    def mytrainer():
        faces = []
        for i in range(1, 42):
            # train/s1/
            person_path = f'./train/s{str(i)}'
            # person_path = dataset_path + '/s' + str(i)
            for j in range(1, 6):
                # s1/masked_1.jpg
                face_path = f'{person_path}/masked_{str(j)}.jpg'
                # face_path = person_path + '/masked_' + str(j) + '.jpg'
                img = cv2.imread(face_path)
                if img is None:
                    print(f'{face_path} is Empty!')
                    return -1
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Normalization
                img_norm = cv2.equalizeHist(img_gray)
                height, width = img_norm.shape
                # 2D img to 1D array
                img_array = img_norm.reshape(
                    height * width, 1).astype('float64')
                faces.append(img_array)
        print("Normalization finished...")
        # 平均脸
        avg_face = np.mean(faces, axis=0)
        avg_img = avg_face.reshape((height, width))
        cv2.imwrite("./train_result/avg_face.jpg", avg_img.astype('uint8'))
        cv2.imshow("avgface", avg_img.astype('uint8'))
        print("Avg Face finished...")
        # 减去平均脸，方便求协方差矩阵
        fai_faces = []
        for face in faces:
            fai_face = face - avg_face
            fai_faces.append(fai_face)
        print("Delta face finished")
        # 求Cov矩阵
        cov = np.zeros((height*width, height*width))
        for delta_face in fai_faces:
            cov += np.transpose(delta_face) * delta_face
        cov = cov / len(fai_faces)
        # 求特征值和特征向量并降序排列
        eigenvalue, eigenvector = np.linalg.eig(cov)
        sorted_ev = []
        for i in range(height*width):
            sorted_ev.append((eigenvalue[i].real, i))
        sorted_ev.sort(reverse=True, key=lambda e: e[0])
        print("Cov finished")
        v = []
        # 保存前100个特征向量
        for i in range(100):
            v.append(eigenvector[:, sorted_ev[i][1]].real)
        v_img = []

        # 取前十个特征向量
        plt.figure()
        for i in range(1, 10 + 1):
            v_img.append(np.array(v[i-1]).reshape(height, width))
            cv2.normalize(v_img[i-1], v_img[i-1], 0, 255, cv2.NORM_MINMAX)
            plt.subplot(2, 5, i)
            plt.imshow(v_img[i-1].astype('uint8'), cmap=plt.cm.gray)
            plt.xticks(())
            plt.yticks(())
        plt.savefig("./train_result/eigenfaces.jpg")
        plt.show()
        # 将特征向量结果保存至model文件，用于识别和重构
        np.savetxt('./model/model.txt', np.array(v), fmt="%s")

    @staticmethod
    def mytester(test_path, pcs):
        # 识别结果
        test_result = False

        # 读取model
        A = np.loadtxt('./model/model.txt')
        A = A[0:(pcs-1)]

        # 计算测试图片在特征空间的坐标
        test_img = cv2.imread(test_path)
        test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        test_norm = cv2.equalizeHist(test_gray)
        height, width = test_norm.shape
        test_array = test_norm.reshape((height*width), 1).astype('float64')
        test_y = np.matmul(A, test_array)

        # 遍历训练库，寻找特征空间下欧氏距离最近的图片
        min = sys.maxsize
        result_path = ''
        dataset_path = './train'
        for i in range(1, 42):
            person_path = f'{dataset_path}/s{i}'
            for j in range(1, 6):
                face_path = f'{person_path}/masked_{j}.jpg'
                img = cv2.imread(face_path)
                if img is None:
                    print(f'{face_path} is None!')
                    return -1
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_norm = cv2.equalizeHist(img_gray)
                img_array = img_norm.reshape(
                    height * width, 1).astype('float64')
                # 将训练库中的图片映射到特征空间
                y = np.matmul(A, img_array)

                # 计算欧氏距离最小的图片即最相似
                distance = np.linalg.norm(y - test_y)
                if distance < min:
                    min = distance
                    result_path = face_path
        if result_path is not None:
            # ./train/s{i}/masked_{j}.jpg
            # 识别是否正确
            (_, _, result_person, _) = result_path.split('/')
            (_, _, test_person, test_face) = test_path.split('/')
            output_path = f'./test_result_{pcs}/{test_person}/test_{test_face}'
            if result_person == test_person:
                test_result = True
            test_info = 'Success!' if test_result else 'Failed!'
            print(result_path, min, test_info)
            result = cv2.imread(result_path)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            plt.figure()
            # 原图
            plt.subplot(1, 3, 1)
            plt.imshow(test_gray, cmap=plt.cm.gray)
            plt.title("Target Image")
            plt.xlabel(test_path)
            plt.xticks(())
            plt.yticks(())

            # 叠加图
            add = test_gray.astype('float64') + result.astype('float64')
            cv2.normalize(add, add, 0, 255, cv2.NORM_MINMAX)
            plt.subplot(1, 3, 2)
            plt.imshow(add.astype('uint8'), cmap=plt.cm.gray)
            plt.title("Blend Image")
            plt.xlabel(test_info)
            plt.xticks(())
            plt.yticks(())

            # 最相似图片
            plt.subplot(1, 3, 3)
            plt.imshow(result, cmap=plt.cm.gray)
            plt.title("Recognized Image")
            plt.xlabel(result_path)
            plt.xticks(())
            plt.yticks(())
            # 存储绘制结果
            # plt.show()
            # plt.savefig(output_path)
            plt.close()
        return test_result

    @staticmethod
    def myreconstruct(model_path, face_path, p):
        # 读取model
        A = np.loadtxt(model_path)

        img = cv2.imread(face_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_norm = cv2.equalizeHist(img_gray)
        height, width = img_norm.shape
        img_array = img_norm.reshape(height * width, 1)
        avg_face = cv2.imread('./train_result/avg_face.jpg', flags=-1)
        avg_array = avg_face.reshape(height * width, 1)
        img_float_array = img_array.astype(
            'float64') - avg_array.astype('float64')

        plt.figure()
        rec_result = []
        for i in range(len(p)):
            # 求得特征脸空间下的k维向量坐标
            y = np.matmul(A[0:p[i]], img_float_array)

            # 重构
            rec_result.append(np.matmul(A[0:p[i]].T, y).astype('float64'))
            rec_result[i] = np.array(rec_result[i]).reshape(height, width)
            rec_result[i] += avg_face
            cv2.normalize(rec_result[i], rec_result[i],
                          0, 255, cv2.NORM_MINMAX)
            # cv2.imwrite("./result/reconstruct_"+str(p[i])+".jpg", rec_result[i].astype('uint8'))
            print(f'PCs {p[i]}:Successfully reconsrtuct!')

            plt.subplot(1, len(p), i+1)
            plt.imshow(rec_result[i].astype('uint8'), cmap=plt.cm.gray)
            plt.title(str(p[i])+"PCs")
            plt.xticks(())
            plt.yticks(())
        plt.savefig("./reconstruct_result/reconstruct.jpg")
        plt.show()
        return


@click.group()
def cli():
    pass


@cli.command()
def train():
    # 使用40份AT&T人脸+一份自己的人脸中的前5张做训练
    EigenFace.mytrainer()


@cli.command()
# @click.option('--energy', required=True, type=int, help='energy propotion to use')
def test():
    # 使用40份AT&T人脸+一份自己的人脸中的后5张做识别测试
    dataset_path = './train'
    res_list = []
    for pcs in [10, 25, 50, 100]:
        result_cnt = 0
        for i in range(1, 42):
            person_path = f'{dataset_path}/s{i}'
            for j in range(6, 11):
                face_path = f'{person_path}/masked_{j}.jpg'
                result_cnt += 1 if EigenFace.mytester(face_path, pcs) else 0
        res = f'PCs: {pcs}\nSuccess: {result_cnt}\nTotal: 205\nrate: {result_cnt/205}%\n\n'
        print(res)
        res_list.append(res)
    with open(f'./test_ans.txt', 'w+') as file:
        file.writelines(res_list)
        file.close()


@cli.command()
@click.option('--img', required=True, type=str, help='img path to reconstruct like s12/masked_1.jpg')
def reconstruct(img):
    EigenFace.myreconstruct("./model/model.txt",
                            f'./train/{img}', (10, 25, 50, 100))


if __name__ == '__main__':
    cli()
