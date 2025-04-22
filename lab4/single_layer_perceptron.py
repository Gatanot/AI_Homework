import numpy as np

# 激活函数（阶跃函数）


# 单层感知机类
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        # 初始化权重和偏置
        self.weights = np.random.rand(input_size)  # 随机初始化权重
        self.bias = np.random.rand(1)  # 随机初始化偏置
        self.learning_rate = learning_rate  # 学习率

    def step_function(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        # 计算加权和
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        # 通过激活函数
        return self.step_function(weighted_sum)

    def train(self, inputs, labels, epochs=100):
        for epoch in range(epochs):
            for i in range(len(inputs)):
                # 前向传播
                prediction = self.predict(inputs[i])
                # 计算误差
                error = labels[i] - prediction
                # 更新权重和偏置
                self.weights += self.learning_rate * error * inputs[i]
                self.bias += self.learning_rate * error


def main():

    # 数据准备
    # 输入：学习时间（小时），复习次数
    inputs = np.array([[2, 1], [3, 2], [5, 3], [6, 4]])
    # 标签：1表示通过，0表示未通过
    labels = np.array([0, 0, 1, 1])

    # 初始化感知机
    perceptron = Perceptron(input_size=2, learning_rate=0.1)

    # 训练感知机
    perceptron.train(inputs, labels, epochs=100)

    # 测试感知机
    print("测试结果：")
    for i in range(len(inputs)):
        prediction = perceptron.predict(inputs[i])
        print(f"输入: {inputs[i]}, 预测: {prediction}, 实际: {labels[i]}")

    # 测试新数据
    new_input = np.array([4, 2])  # 新学生：学习4小时，复习2次
    prediction = perceptron.predict(new_input)
    print(f"新输入: {new_input}, 预测: {prediction}")


main()
