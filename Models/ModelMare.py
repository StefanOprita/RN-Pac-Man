from keras.layers import Dropout
from keras.optimizer_v2.adam import Adam

from Models.PacManModel import PacManModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from hyperparameters import input_size, number_actions
from tensorflow.keras.optimizers import RMSprop


class ModelMare(PacManModel):
    def __init__(self):
        super(ModelMare, self).__init__()
        self.__initialize_model()

    def __initialize_model(self):
        self.model = Sequential()

        self.model.add(Dense(256, input_dim=input_size, activation='relu', kernel_initializer='he_uniform'))
        self.model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        self.model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        self.model.add(Dense(9, activation='linear', kernel_initializer='he_uniform'))

        self.model.compile(optimizer=Adam(0.001), loss='mse')
