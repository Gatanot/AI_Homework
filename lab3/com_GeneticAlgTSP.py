from GeneticAlgTSP import GeneticAlgTSP
import matplotlib.pyplot as plt


def test_iteration():
    iterations = [i for i in range(100, 1600, 100)]
    print(iterations)
    best_distances = []
    for i in iterations:
        ga = GeneticAlgTSP("dj38.tsp")
        ga.iterate(i)
        best_distances.append(ga.best_distance)
    plot_connected_points(iterations, best_distances)


def test_mutation_rate():
    mutation_rates = [i for i in range(1, 11, 1)]
    print(mutation_rates)
    best_distances = []
    for i in mutation_rates:
        ga = GeneticAlgTSP("dj38.tsp", mutation_rate=i / 10)
        ga.iterate(100)
        best_distances.append(ga.best_distance)
    plot_connected_points(mutation_rates, best_distances)


def plot_connected_points(x_coords, y_coords):

    # 创建图形
    plt.figure(figsize=(8, 6))

    # 绘制连线图，红色线条带圆圈标记
    plt.plot(x_coords, y_coords, "r-o", linewidth=2, markersize=8)

    # 添加标题和标签
    plt.title("compare", fontsize=14)
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)

    # 添加网格
    plt.grid(True, linestyle="--", alpha=0.6)

    # 显示图形
    plt.show()


test_iteration()
test_mutation_rate()
