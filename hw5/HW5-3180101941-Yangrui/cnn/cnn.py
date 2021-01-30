import click
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras import datasets

from model import create_better_model, create_model


class cifar_10(object):
    @staticmethod
    def train():
        # 加载训练数据
        (train_images, train_labels), (test_images,
                                       test_labels) = datasets.cifar10.load_data()

        # 将像素的值标准化至0到1的区间内
        train_images, test_images = train_images / 255.0, test_images / 255.0

        # 加载模型
        # model = create_model(input_size=(32, 32, 3), kernel_size=(3, 3))
        model = create_better_model()
        # 编译模型
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(
                          from_logits=True),
                      metrics=['accuracy'])

        # 训练模型 epochs = 10
        history = model.fit(train_images, train_labels, epochs=20,
                            validation_data=(test_images, test_labels))

        # 保存权重到本地
        model.save_weights('./cifar_models/cifar10_model')

        # 绘制训练结果
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.3, 1])
        plt.legend(loc='lower right')
        plt.show()

        # 测试训练结果
        test_loss, test_acc = model.evaluate(
            test_images,  test_labels, verbose=2)
        print(test_acc)

    @staticmethod
    def test():
        # 加载训练数据
        (train_images, train_labels), (test_images,
                                       test_labels) = datasets.cifar10.load_data()

        # 将像素的值标准化至0到1的区间内。
        train_images, test_images = train_images / 255.0, test_images / 255.0

        # 加载模型
        # model = create_model(input_size=(32, 32, 3))
        model = create_better_model()
        # 编译模型
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(
                          from_logits=True),
                      metrics=['accuracy'])
        model.load_weights('./cifar_models/cifar10_model')

        # 测试训练结果
        test_loss, test_acc = model.evaluate(
            test_images,  test_labels, verbose=2)
        print(test_acc)


class mnist(object):
    @staticmethod
    def pre_process(train_images, test_images):
        # 将像素的值标准化至0到1的区间内。
        train_images, test_images = train_images / 255.0, test_images / 255.0
        # 将28 x 28的图片填充为32 x 32
        train_images = np.pad(
            train_images, ((0, 0), (2, 2), (2, 2)), 'constant')
        test_images = np.pad(test_images, ((0, 0), (2, 2), (2, 2)), 'constant')
        # 将32 x 32展开为32 x 32 x 1
        tmp_train = []
        tmp_test = []
        for train_image in train_images:
            tmp_train.append(train_image.reshape(32, 32, 1))
        for test_image in test_images:
            tmp_test.append(test_image.reshape(32, 32, 1))
        return np.array(tmp_train), np.array(tmp_test)

    @staticmethod
    def train():
        # 加载训练数据
        (train_images, train_labels), (test_images,
                                       test_labels) = datasets.mnist.load_data()

        # 预处理输入
        train_images, test_images = mnist.pre_process(
            train_images, test_images)
        # 将训练数据打乱
        train_images, train_labels = shuffle(train_images, train_labels)

        # 加载模型
        model = create_model((32, 32, 1))

        # 编译模型
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(
                          from_logits=True),
                      metrics=['accuracy'])

        # 训练模型 epochs = 10
        history = model.fit(train_images, train_labels, epochs=10,
                            validation_data=(test_images, test_labels))

        # 保存权重到本地
        model.save_weights('./mnist_models/mnist_model')

        # 绘制训练结果
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        plt.show()

        # 测试训练结果
        test_loss, test_acc = model.evaluate(
            test_images,  test_labels, verbose=2)
        print(test_acc)

    @staticmethod
    def test():
        model = create_model()

        # 编译模型
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(
                          from_logits=True),
                      metrics=['accuracy'])

        # 载入模型权重
        model.load_weights('./mnist_models/mnist_model')

        # 加载数据
        (train_images, train_labels), (test_images,
                                       test_labels) = datasets.mnist.load_data()

        # 预处理输入
        train_images, test_images = mnist.pre_process(
            train_images, test_images)

        # 测试模型
        test_loss, test_acc = model.evaluate(
            test_images, test_labels, verbose=2)

        print(test_acc, test_loss)


@click.group()
def cli():
    pass


@cli.command()
@click.option('--option', required=True, type=str, help='train or test')
def Mnist(option):
    if option == 'train':
        mnist.train()
    elif option == 'test':
        mnist.test()
    else:
        print('train or test')


@cli.command()
@click.option('--option', required=True, type=str, help='train or test')
def Cifar_10(option):
    if option == 'train':
        cifar_10.train()
    elif option == 'test':
        cifar_10.test()
    else:
        print('train or test')


if __name__ == "__main__":
    cli()
