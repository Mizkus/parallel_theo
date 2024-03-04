import matplotlib.pyplot as plt

stat = list()
dyn = list()
gui = list()

with open("data/stat.txt") as file:
    stat = list(map(float, file.readline().split()))

with open("data/stat_dyn.txt") as file:
    dyn = list(map(float, file.readline().split()))

with open("data/stat_gui.txt") as file:
    gui = list(map(float, file.readline().split()))

min_len = min([len(stat), len(dyn), len(gui)])

plt.xlabel("chunk size")
plt.ylabel("seconds")
x = [i for i in range(1, min_len * 100, 100)]
plt.plot(x, stat[:min_len], label="static")
plt.plot(x, dyn[:min_len], label="dynamic")
plt.plot(x, gui[:min_len], label="guided")
plt.legend()
plt.show()