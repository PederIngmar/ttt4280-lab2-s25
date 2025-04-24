
import matplotlib . patches as mpatches
import numpy as np
import matplotlib.pyplot as plt
import config

def vinkeloppsett(testvinkel):
    fig, ax = plt.subplots()
    theta = np.linspace(0, 180, 1000)
    x = np.cos(np.radians(theta))
    y = np.sin(np.radians(theta))
    ax.plot(x, y, color='black')

    mikrofon_1 = plt.Circle((0, 0.6), 0.05, color=config.colors['mic1'])
    mikrofon_2 = plt.Circle((-0.5, -0.3), 0.05, color=config.colors['mic2'])
    mikrofon_3 = plt.Circle((0.5, -0.3), 0.05, color=config.colors['mic3'])
    ax.add_patch(mikrofon_1)
    ax.add_patch(mikrofon_2)
    ax.add_patch(mikrofon_3)

    label_mic1 = mpatches.Patch(color=config.colors['mic1'], label='Mikrofon 1')
    label_mic2 = mpatches.Patch(color=config.colors['mic2'], label='Mikrofon 2')
    label_mic3 = mpatches.Patch(color=config.colors['mic3'], label='Mikrofon 3')

    vinkelskive = np.arange(0, 181, 30)
    for vinkel in vinkelskive:
        x_tick = np.cos(np.radians(vinkel))
        y_tick = np.sin(np.radians(vinkel))
        ax.plot([0, x_tick], [0, y_tick], color='black', linestyle='dashed')
        ax.text(x_tick * 1.1, y_tick * 1.1, str(vinkel), ha='center', va='center')

    ax.plot([0, np.cos(np.radians(testvinkel))], [0.6, np.sin(np.radians(testvinkel))], color=config.colors['mic1'], linestyle='-')
    ax.plot([-0.5, np.cos(np.radians(testvinkel))], [-0.3, np.sin(np.radians(testvinkel))], color=config.colors['mic2'], linestyle='-')
    ax.plot([0.5, np.cos(np.radians(testvinkel))], [-0.3, np.sin(np.radians(testvinkel))], color=config.colors['mic3'], linestyle='-')

    ax.set_aspect('equal', 'box')
    ax.axis('off')
    plt.legend(handles=[label_mic1, label_mic2, label_mic3])
    plt.show()

if __name__ == "__main__":
    testvinkel = 0
    vinkeloppsett(testvinkel)