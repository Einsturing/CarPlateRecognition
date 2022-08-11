import sys
import os
import numpy as np
import cv2
import tensorflow.compat.v1 as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphbets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z']
chinese = ['zh_cuan', 'zh_e', 'zh_gan', 'zh_gan1', 'zh_gui', 'zh_gui1', 'zh_hei', 'zh_hu', 'zh_ji', 'zh_jin',
           'zh_jing', 'zh_jl', 'zh_liao', 'zh_lu', 'zh_meng', 'zh_min', 'zh_ning', 'zh_qing', 'zh_qiong',
           'zh_shan', 'zh_su', 'zh_sx', 'zh_wan', 'zh_xiang', 'zh_xin', 'zh_yu', 'zh_yu1', 'zh_yue', 'zh_yun',
           'zh_zang', 'zh_zhe']
tf.disable_v2_behavior()



class char_cnn_net:  # 初始化
    def __init__(self):
        self.dataset = numbers + alphbets + chinese
        self.dataset_len = len(self.dataset)
        self.img_size = 20
        self.y_size = len(self.dataset)
        self.batch_size = 100

        self.x_place = tf.placeholder(dtype=tf.float32, shape=[None, self.img_size, self.img_size], name='x_place')
        self.y_place = tf.placeholder(dtype=tf.float32, shape=[None, self.y_size], name='y_place')
        self.keep_place = tf.placeholder(dtype=tf.float32, name='keep_place')

    def cnn_construct(self):  # 构建神经网络
        x_input = tf.reshape(self.x_place, shape=[-1, 20, 20, 1])  # 变换格式，与图片相联系
        # 卷积层1
        cw1 = tf.Variable(tf.random_normal(shape=[3, 3, 1, 32], stddev=0.01), dtype=tf.float32)  # 初始化变量，卷积核，定义权重
        cb1 = tf.Variable(tf.random_normal(shape=[32]), dtype=tf.float32)  # 定义偏差
        #             激励层，非线性映射        卷积，卷积核为cw1，考虑边界，并激活（大于cb1的不变，小于cb1的置为cb1）
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x_input, filter=cw1, strides=[1, 1, 1, 1], padding='SAME'), cb1))
        # 池化层                  需要池化的输入  池化窗口大小      窗口在每一维度上滑动的步长  全0填充（去除杂余信息，保留关键信息）
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME')  # 2*2的卷积核，目的为增加感受野，进一步特征抽样

        conv1 = tf.nn.dropout(conv1, self.keep_place)  # 防止或减轻过拟合，随机丢弃一部分神经元
        # 卷积层2
        cw2 = tf.Variable(tf.random_normal(shape=[3, 3, 32, 64], stddev=0.01), dtype=tf.float32)
        cb2 = tf.Variable(tf.random_normal(shape=[64]), dtype=tf.float32)
        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, filter=cw2, strides=[1, 1, 1, 1], padding='SAME'), cb2))
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.dropout(conv2, self.keep_place)
        # 卷积层3
        cw3 = tf.Variable(tf.random_normal(shape=[3, 3, 64, 128], stddev=0.01), dtype=tf.float32)
        cb3 = tf.Variable(tf.random_normal(shape=[128]), dtype=tf.float32)
        conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, filter=cw3, strides=[1, 1, 1, 1], padding='SAME'), cb3))
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv3 = tf.nn.dropout(conv3, self.keep_place)

        conv_out = tf.reshape(conv3, shape=[-1, 3 * 3 * 128])

        fw1 = tf.Variable(tf.random_normal(shape=[3 * 3 * 128, 1024], stddev=0.01), dtype=tf.float32)
        fb1 = tf.Variable(tf.random_normal(shape=[1024]), dtype=tf.float32)
        #          激活       矩阵加      矩阵乘
        fully1 = tf.nn.relu(tf.add(tf.matmul(conv_out, fw1), fb1))
        fully1 = tf.nn.dropout(fully1, self.keep_place)

        fw2 = tf.Variable(tf.random_normal(shape=[1024, 1024], stddev=0.01), dtype=tf.float32)
        fb2 = tf.Variable(tf.random_normal(shape=[1024]), dtype=tf.float32)
        fully2 = tf.nn.relu(tf.add(tf.matmul(fully1, fw2), fb2))
        fully2 = tf.nn.dropout(fully2, self.keep_place)

        fw3 = tf.Variable(tf.random_normal(shape=[1024, self.dataset_len], stddev=0.01), dtype=tf.float32)
        fb3 = tf.Variable(tf.random_normal(shape=[self.dataset_len]), dtype=tf.float32)
        fully3 = tf.add(tf.matmul(fully2, fw3), fb3, name='out_put')

        return fully3

    def train(self, data_dir, save_model_path):  # 训练
        print('ready load train dataset')
        X, y = self.init_data(data_dir)
        print('success load' + str(len(y)) + 'datas')
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)  # 划分样本

        out_put = self.cnn_construct()
        predicts = tf.nn.softmax(out_put)  # 归一化，得分值转化为概率，输出层
        predicts = tf.argmax(predicts, axis=1)  # 记录每一行最大的索引，预测结果
        actual_y = tf.argmax(self.y_place, axis=1)  # 记录每一行最大的索引，实际结果
        #           求均值          数据类型转换   判断是否相等
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predicts, actual_y), dtype=tf.float64))
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_put, labels=self.y_place))  # 计算代价
        #                             学习率
        opt = tf.train.AdamOptimizer(learning_rate=0.001)  # adam优化算法，寻找全局最优点的优化算法，引入二次方梯度校正
        train_step = opt.minimize(cost)  # 最大限度地最小化cost

        with tf.Session() as sess:  # 启动图运行操作
            init = tf.global_variables_initializer()
            sess.run(init)  # 初始化tf中建立地变量变量
            step = 0
            saver = tf.train.Saver()  # 用于训练网络之后保存训练好的模型，以及在程序中读取已保存好的模型
            while True:
                train_index = np.random.choice(len(train_x), self.batch_size, replace=False)  # 从train_x中选取随机的样本
                train_randx = train_x[train_index]
                train_randy = train_y[train_index]
                # 对变量初始化，真正的赋值操作
                _, loss = sess.run([train_step, cost],
                                   feed_dict={self.x_place: train_randx, self.y_place: train_randy,
                                              self.keep_place: 0.75})  # 赋值，当次有效
                step += 1

                if step % 10 == 0:  # 每训练10张图测试1次
                    test_index = np.random.choice(len(test_x), self.batch_size, replace=False)  # 从test_x中选取随机的样本
                    test_randx = test_x[test_index]
                    test_randy = test_y[test_index]
                    acc = sess.run(accuracy, feed_dict={self.x_place: test_randx, self.y_place: test_randy,
                                                        self.keep_place: 1.0})
                    print(step, loss, file=f)

                    if step % 50 == 0:
                        print('accuracy:', acc)  # 输出训练结果
                    if step % 500 == 0:
                        saver.save(sess, save_model_path, global_step=step)  # 保存模型
                    if acc > 0.99 and step > 10000:  # 正确率大于0.99且训练过500张图
                        saver.save(sess, save_model_path, global_step=step)  # 保存模型
                        break

    def test(self, x_images, model_path):  # 测试
        text_list = []
        output = self.cnn_construct()
        predicts = tf.nn.softmax(output)  # 归一化
        predicts = tf.argmax(predicts, axis=1)  # 记录每一行最大的索引
        saver = tf.train.Saver()  # 实例化一个saver对象
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())  # 初始化tf中建立的变量
            saver.restore(sess, model_path)  # 调用模型
            preds = sess.run(predicts, feed_dict={self.x_place: x_images, self.keep_place: 1.0})
            for i in range(len(preds)):
                pred = preds[i].astype(int)  # 把列表转化为int数组
                text_list.append(self.dataset[pred])  # 添加对象
            return text_list

    def list_all_files(self, root):
        files = []
        list = os.listdir(root)
        for i in range(len(list)):
            element = os.path.join(root, list[i])
            if os.path.isdir(element):
                temp_dir = os.path.split(element)[-1]
                if temp_dir in self.dataset:
                    files.extend(self.list_all_files(element))
            elif os.path.isfile(element):
                files.append(element)
        return files

    def init_data(self, dir):
        X = []
        y = []
        if not os.path.exists(data_dir):
            raise ValueError('没有找到文件夹')
        files = self.list_all_files(dir)

        for file in files:
            src_img = cv2.imread(file, cv2.COLOR_BGR2GRAY)
            if src_img.ndim == 3:
                continue
            resize_img = cv2.resize(src_img, (20, 20))
            X.append(resize_img)
            # 获取图片文件全目录
            dir = os.path.dirname(file)
            # 获取图片文件上一级目录名
            dir_name = os.path.split(dir)[-1]
            vector_y = [0 for i in range(len(self.dataset))]
            index_y = self.dataset.index(dir_name)
            vector_y[index_y] = 1
            y.append(vector_y)

        X = np.array(X)
        y = np.array(y).reshape(-1, self.dataset_len)
        return X, y

    def init_testData(self, dir):
        test_X = []
        if not os.path.exists(dir):
            raise ValueError('没有找到文件夹')
        files = self.list_all_files(dir)
        for file in files:
            src_img = cv2.imread(file, cv2.COLOR_BGR2GRAY)
            if src_img.ndim != 3:
                continue
            resize_img = cv2.resize(src_img, (20, 20))
            test_X.append(resize_img)
        test_X = np.array(test_X)
        return test_X


if __name__ == '__main__':
    cur_dir = sys.path[0]
    data_dir = os.path.join(cur_dir, 'images/cnn_char_train')
    test_dir = os.path.join(cur_dir, 'images/cnn_char_test')
    train_model_path = os.path.join(cur_dir, 'model/char_recongnize/model.ckpt')
    model_path = os.path.join(cur_dir, 'model/char_recongnize/model.ckpt-550')

    train_flag = 1
    net = char_cnn_net()

    if train_flag == 1:
        # 训练模型
        f = open("normal.txt", "w")
        net.train(data_dir, train_model_path)
        f.close()
    else:
        test_X = net.init_testData(test_dir)
        text = net.test(test_X, model_path)
        print(text)
