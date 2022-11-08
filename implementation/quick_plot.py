from matplotlib import pyplot as plt

def gen_plot(ds):
    plt.subplots_adjust(wspace=0.5)
    for i, d in enumerate(ds):
        plt.subplot(1, len(ds), i+1, adjustable='box', aspect=1)
        plt.plot([0,2], [d, 1], marker='o', color='k')
        plt.plot([0,2], [-d, -1], marker='o', color='k')
        plt.xticks([])
        plt.yticks([-1, -0.5, 0, 0.5, 1])
    # plt.savefig('implementation/process_visualization.png')
    plt.show()

if __name__ == '__main__':
    gen_plot([0.5, 0.1, 0])