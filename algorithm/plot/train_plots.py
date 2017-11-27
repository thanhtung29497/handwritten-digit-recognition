import matplotlib.pyplot as plt


def loss_per_epoch(epochs, eval_every, train_loss):
    eval_indices = range(0, epochs, eval_every)
    plt.plot(eval_indices, train_loss, 'k-')
    plt.title('Softmax Loss per epochs')
    plt.xlabel('epochs')
    plt.ylabel('Softmax Loss')
    plt.show()


def train_test_accuracy(epochs, eval_every, train_accuracy, test_accuracy):
    eval_indices = range(0, epochs, eval_every)
    plt.plot(eval_indices, train_accuracy, 'k-', label='Train Set Accuracy')
    plt.plot(eval_indices, test_accuracy, 'r--', label='Test Set Accuracy')
    plt.title('Train and Test Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()