import pandas as pd
import matplotlib.pyplot as plt
from preparator import Preparator
from sklearn.svm import SVR
import numpy as np
from error_metric_calculator import Metrics
import printer




class SVRModel:
    def __init__(self, attribute, test_from_date, test_to_date, horizon):
        self.preparator = Preparator(attribute, test_from_date)
        self.y_train, self.y_test = self.preparator.y_train, self.preparator.y_test
        self.prediction = self.multistep_forecast(test_from_date=test_from_date, test_to_date=test_to_date, horizon=horizon)
        self.individual_error_scores, self.overall_error_scores = Metrics().calculate_errors(self.preparator.y_test, self.prediction)
        self.individual_error_scores.index = self.prediction.index
        self.std_error = self.individual_error_scores.std()

        printer.print_error(self.individual_error_scores)
        printer.print_single_forecast(self.y_train, self.y_test, self.prediction)


    def fit_and_predict(self, test_from_date, horizon):
        x_train, x_test, y_train, y_test = self.preparator.get_scaled_data(test_from_date=test_from_date)
        model = SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.05, gamma=0.5, kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=True)
        model = model.fit(x_train, y_train.ravel())
        prediction = model.predict(x_test[:horizon])
        inverse_scaled_prediction = self.preparator.inverse_scaler(prediction)
        return inverse_scaled_prediction.ravel()


    def format_prediction(self, prediction):
        prediction = pd.Series(prediction)
        prediction.index = self.preparator.y_test[:len(prediction)].index
        prediction.index = pd.DatetimeIndex(prediction.index)
        return prediction

    def multistep_forecast(self, test_from_date, test_to_date, horizon=1):
        date_range = pd.date_range(test_from_date, test_to_date, freq=str(horizon) + "H")
        prediction = []
        for date in date_range:
            prediction = np.append(prediction, self.fit_and_predict(test_from_date=date, horizon=horizon))
        return self.format_prediction(prediction)


model = SVRModel("solar_absolute", test_from_date="2020-06-01 00:00", test_to_date="2020-06-02 00:00", horizon=3)



# %%


# scaled_prediction = format_prediction(scaled_prediction[0][:1200], preparator.y_test)

#print_forecast(model.preparator.y_train, model.preparator.y_test, model.prediction3,model.prediction1,"")
# print(model.model.score(x_test, y_test))
