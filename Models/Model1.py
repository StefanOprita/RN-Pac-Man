from Models.PacManModel import PacManModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from hyperparameters import input_size, number_actions


class Model1(PacManModel):
    def __init__(self):
        super(Model1, self).__init__()
        self.__initialize_model()

    def __initialize_model(self):
        self.model = Sequential()

        self.model.add(Dense(32, input_dim=input_size, activation='relu'))
        # self.model.add(Dense(100, activation='relu'))

        self.model.add(Dense(number_actions))

        self.model.compile(optimizer='adam', loss='mean_squared_error')
