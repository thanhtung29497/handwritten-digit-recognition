from tensorflow import reduce_mean, train, placeholder, reshape, float32, global_variables_initializer, argmax, equal, cast, InteractiveSession
from ..plot import loss_per_epoch, train_test_accuracy


class CNNModel:
    def __init__(self, image_size=[28, 28], char_number=10, channel=1):
        self.image_size = image_size
        self.char_number = char_number
        self.channel = channel
        # self.learning_rate = learning_rate
        # self.eval_every = eval_every
        # self.epochs = epochs
        # self.evaluation_size = evaluation_size
        # self.batch_size = batch_size
        # self.optimizer = optimizer

        self.inputs = placeholder(
            float32,
            shape=[None, image_size[0] * image_size[1]],
            name="inputs")

        self.model = reshape(
            self.inputs, [-1, image_size[0], image_size[1], channel])
        self.labels = placeholder(float32, shape=[None, 10], name="labels")
        self.sess = InteractiveSession()

    def addLayer(self, layer):
        self.model = layer.implement(self.model)

    def addOutputLayer(self, layer):
        self.loss = layer.implement(self.model, self.labels)

    def train(self,
              dataset,
              learning_rate=0.005,
              eval_every=5,
              epochs=500,
              evaluation_size=500,
              batch_size=100,
              optimizer=train.MomentumOptimizer(0.005, 0.9)):
        train_step = optimizer.minimize(self.loss)
        prediction = argmax(self.model, 1, name="prediction")
        result = equal(argmax(self.labels, 1), prediction, name="result")
        accuracy = reduce_mean(cast(result, float32))
        train_loss = []
        train_accuracy = []
        test_accuracy = []

        global_variables_initializer().run()

        for i in range(epochs):
            # Lay ra batch_size hinh anh tu tap train
            train_batch = dataset.train.next_batch(batch_size)
            train_dict = {
                self.inputs: train_batch[0],
                self.labels: train_batch[1]
            }
            if i % eval_every == 0:

                # Cu sau eval_every buoc lap thi test mot lan
                test_batch = dataset.test.next_batch(evaluation_size)
                temp_train_loss = self.loss.eval(feed_dict=train_dict)
                temp_train_accuracy = accuracy.eval(feed_dict=train_dict)
                temp_test_accuracy = accuracy.eval(feed_dict={
                    self.inputs: test_batch[0],
                    self.labels: test_batch[1]
                })

                print(
                    'Epoch # %d, Train Loss: %g. Train Accuracy (Test Accuracy): %g (%g)'
                    % (i, temp_train_loss, temp_train_accuracy,
                       temp_test_accuracy))

                # Luu cac gia tri de ve bieu do
                train_loss.append(temp_train_loss)
                train_accuracy.append(temp_train_accuracy)
                test_accuracy.append(temp_test_accuracy)

            # Chay thuat toan toi uu ham mat mat
            self.sess.run(train_step, feed_dict=train_dict)

        # Show plots
        loss_per_epoch(epochs, eval_every, train_loss)
        train_test_accuracy(epochs, eval_every, train_accuracy, test_accuracy)

    def save(self, model_path):
        """
            save model to folder in model_path
        """
        saver = train.Saver()
        saver.save(self.sess, model_path)