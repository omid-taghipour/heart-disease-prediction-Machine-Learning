from sklearn import svm


class SVM_Linear:
    def __init__(self, x_train, y_train):
        self.X_train = x_train
        self.Y_train = y_train
        self.model_training()

    def model_training(self):
        return svm.SVC(kernel='linear').fit(self.X_train, self.Y_train)

    def predict_result(self, x):
        return self.model_training().predict(x)
