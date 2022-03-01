from sklearn.linear_model import LogisticRegression


class logistic_regression:
    def __init__(self, x_train, y_train):
        self.X_train = x_train
        self.Y_train = y_train
        self.model = self.model_training()

    def model_training(self):
        return LogisticRegression().fit(self.X_train, self.Y_train)

    def predict_result(self, x_train):
        return self.model.predict(x_train)
