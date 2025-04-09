from GeneticAlgTSP import GeneticAlgTSP
from time import strftime
from time import localtime
import matplotlib.pyplot as plt


test_files = ["wi29.tsp", "dj38.tsp", "qa194.tsp"]
result_file = address = strftime("%Y_%m_%d_%H_%M", localtime()) + ".md"


def save(testname, var, distance):

    with open(result_file, "a") as f:
        f.write(f"# {testname}\n")
        f.write(f"|var|distance|\n")
        f.write(f"|---|---|\n")
        for i in range(len(var)):
            f.write(f"|{var[i]}|{distance[i]}|\n")


def test_iteration(filename):
    iterations = [i for i in range(100, 2550, 50)]
    print(iterations)
    best_distances = []
    for i in iterations:
        ga = GeneticAlgTSP(filename)
        ga.iterate(i)
        best_distances.append(ga.best_distance)
    return iterations, best_distances


def test_mutation_rate(filename):
    mutation_rates = [i for i in range(1, 11, 1)]
    print(mutation_rates)
    best_distances = []
    for i in mutation_rates:
        ga = GeneticAlgTSP(filename, mutation_rate=i / 10)
        ga.iterate(100)
        best_distances.append(ga.best_distance)
    return mutation_rates, best_distances


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


def test_save_iteration():
    for file in test_files:
        vars, distance = test_iteration("data\\" + file)
        save("data\\" + file, vars, distance)


test_save_iteration()
