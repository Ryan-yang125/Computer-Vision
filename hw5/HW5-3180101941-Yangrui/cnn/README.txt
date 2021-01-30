To run this script:
1.ensure you have `python >= 3.7.6` in your pc
2.run `pip install`
# 训练mnist数据，将模型保存到本地，并测试数据，输出测试准确率
python cnn.py mnist --option train
# 读取本地已训练好的模型，测试数据，输出测试准确率
python cnn.py mnist --option test
# 训练cifar-10数据，将模型保存到本地，并测试数据，输出测试准确率
python cnn.py cifar_10 --option train
# 读取本地已训练好的模型，测试数据，输出测试准确率
python cnn.py cifar_10 --option test
