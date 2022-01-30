from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.models import Sequential

from Models.PacManModel import PacManModel
from hyperparameters import number_actions, input_shape


class MainModel(PacManModel):
    def __init__(self):
        super(MainModel, self).__init__()
        self.__initialize_model()

    def __initialize_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(2, 3, activation='relu', input_shape=input_shape))

        self.model.add(Flatten())

        self.model.add(Dense(32, activation='relu'))
        # self.model.add(Dense(100, activation='relu'))

        self.model.add(Dense(number_actions))

        self.model.compile(optimizer='adam', loss='mean_squared_error')
