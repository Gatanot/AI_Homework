import numpy as np
import matplotlib.pyplot as plt


class MLP:
    def __init__(self, input_size, hidden_sizes, output_size):
        # 初始化网络参数
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.weights = []
        self.biases = []

        # He初始化权重
        for i in range(len(self.layer_sizes) - 1):
            limit = np.sqrt(2 / self.layer_sizes[i])
            self.weights.append(
                np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * limit
            )
            self.biases.append(np.zeros((1, self.layer_sizes[i + 1])))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def forward(self, x):
        self.activations = [x]
        self.z_values = []

        # 前向传播
        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            if i != len(self.weights) - 1:  # 隐藏层使用ReLU激活
                a = self.relu(z)
            else:  # 输出层不使用激活函数
                a = z
            self.activations.append(a)

        return self.activations[-1]

    def backward(self, x, y, learning_rate):
        # 计算梯度
        m = x.shape[0]
        delta = (self.activations[-1] - y) / m  # MSE的导数

        for i in reversed(range(len(self.weights))):
            # 计算权重和偏置的梯度
            dW = np.dot(self.activations[i].T, delta)
            db = np.sum(delta, axis=0, keepdims=True)

            # 如果不是第一层，计算前一层的delta
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(
                    self.z_values[i - 1]
                )

            # 更新参数
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db

    def train(self, x, y, learning_rate, batch_size=32):
        losses = []
        m = x.shape[0]
        largest_loss_in100 = 0
        smallest_loss_in100 = 10
        epoch = 0
        while True:
            # 随机mini-batch
            indices = np.random.permutation(m)
            x_shuffled = x[indices]
            y_shuffled = y[indices]

            epoch_loss = 0

            for i in range(0, m, batch_size):
                x_batch = x_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]

                # 前向传播
                y_pred = self.forward(x_batch)

                # 计算损失
                loss = np.mean((y_pred - y_batch) ** 2)
                epoch_loss += loss * x_batch.shape[0]

                # 反向传播
                self.backward(x_batch, y_batch, learning_rate)

            epoch_loss /= m
            losses.append(epoch_loss)
            if epoch_loss > largest_loss_in100:
                largest_loss_in100 = epoch_loss
            if epoch_loss < smallest_loss_in100:
                smallest_loss_in100 = epoch_loss
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
                if abs(largest_loss_in100 - smallest_loss_in100) < 0.0015:
                    print(
                        largest_loss_in100,
                        smallest_loss_in100,
                        abs(largest_loss_in100 - smallest_loss_in100),
                    )
                    print("Loss is stable")
                    break
            epoch += 1
        return losses


# 数据预处理
def load_and_preprocess_data():
    # 加载数据 (这里假设数据已经下载并放在同一目录下)
    data = np.loadtxt("MLP_data.csv", delimiter=",", skiprows=1)

    # 分离特征和标签
    X = data[:, :4]  # longitude, latitude, housing_age, homeowner_income
    y = data[:, 4].reshape(-1, 1)  # house_price

    # 数据标准化
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X = (X - X_mean) / X_std

    y_mean = np.mean(y)
    y_std = np.std(y)
    y = (y - y_mean) / y_std

    # 打乱数据
    np.random.seed(42)
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    # 分割训练集和测试集
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, y_train, X_test, y_test, y_mean, y_std


# 数据可视化
def plot_features(X, y):
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.scatter(X[:, 0], y, alpha=0.3)
    plt.xlabel("Longitude (normalized)")
    plt.ylabel("House Price")

    plt.subplot(2, 2, 2)
    plt.scatter(X[:, 1], y, alpha=0.3)
    plt.xlabel("Latitude (normalized)")
    plt.ylabel("House Price")

    plt.subplot(2, 2, 3)
    plt.scatter(X[:, 2], y, alpha=0.3)
    plt.xlabel("Housing Age (normalized)")
    plt.ylabel("House Price")

    plt.subplot(2, 2, 4)
    plt.scatter(X[:, 3], y, alpha=0.3)
    plt.xlabel("Homeowner Income (normalized)")
    plt.ylabel("House Price")

    plt.tight_layout()
    plt.show()


# 主程序
def main():
    # 加载和预处理数据
    X_train, y_train, X_test, y_test, y_mean, y_std = load_and_preprocess_data()

    # 数据可视化
    plot_features(X_train, y_train)

    # 创建MLP模型
    input_size = X_train.shape[1]
    mlp = MLP(input_size, hidden_sizes=[64, 32], output_size=1)

    # 训练模型
    learning_rate = 0.001
    losses = mlp.train(X_train, y_train, learning_rate, batch_size=64)

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss Curve")
    plt.show()

    # 测试模型
    y_pred = mlp.forward(X_test)

    # 反标准化预测结果
    y_pred_orig = y_pred * y_std + y_mean
    y_test_orig = y_test * y_std + y_mean

    # 计算测试集上的RMSE
    rmse = np.sqrt(np.mean((y_pred_orig - y_test_orig) ** 2))
    print(f"Test RMSE: {rmse:.2f}")

    # 绘制预测结果 vs 真实值
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test_orig, y_pred_orig, alpha=0.3)
    plt.plot(
        [y_test_orig.min(), y_test_orig.max()],
        [y_test_orig.min(), y_test_orig.max()],
        "k--",
    )
    plt.xlabel("True Prices")
    plt.ylabel("Predicted Prices")
    plt.title("True vs Predicted House Prices")
    plt.show()


if __name__ == "__main__":
    main()
