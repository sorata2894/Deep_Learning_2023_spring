from utiles import generate_linear, generate_XOR_easy, show_result
from nets import nn_net_func
import matplotlib.pyplot as plt


if __name__ == '__main__':
    x1, y1 = generate_linear()
    x2, y2 = generate_XOR_easy()

    loss_list1 = []
    loss_list2 = []

    net = nn_net_func(lr=0.01, epoch=500, ground_truth=y1, random_seed=0, hidden_node=10)
    parameters, loss_list1 = net.train([x1, y1])
    net.test([x1, y1], parameters=parameters)
    plot1 = plt.figure(1)
    plt.plot(loss_list1)
    plt.title('linear_train_loss')
    plt.show()


    net2 = nn_net_func(lr=0.01, epoch=300, ground_truth=y2, random_seed=0, hidden_node=100)
    parameters2, loss_list2 = net2.train([x2, y2])
    net2.test([x2, y2], parameters=parameters2)
    plot2 = plt.figure(2)
    plt.plot(loss_list2)
    plt.title('XOR_train_loss hidden layer unit = 10')
    plt.show()