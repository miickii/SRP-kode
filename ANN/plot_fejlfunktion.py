import numpy as np
import matplotlib.pyplot as plt

def cross_entropy_loss():
    a = np.linspace(0.01, 0.99, 99)

    tab = -np.log(a) # når y = 1

    plt.plot(a, tab)
    plt.xlabel('Forudsigelsen af sandsynligheden for den forventede kategori')
    plt.ylabel('Krydsentropitabet')
    plt.show()

def plot_convex():
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')

    range = np.arange(-10, 10, 0.1)
    w, b = np.meshgrid(range, range)
    error = np.square(w) + np.square(b)

    ax.plot_surface(w, b, error, cmap="plasma")
    ax.set_xlabel("w", fontsize=20)
    ax.set_ylabel("b", fontsize=20)
    ax.set_zlabel("error", fontsize=20)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_zticklabels([])
    ax.set_title("Fejlfunktion", size=15)
    plt.show()
