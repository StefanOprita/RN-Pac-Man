from keras.optimizer_v2.adam import Adam

from Models.PacManModel import PacManModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from hyperparameters import input_size, number_actions
from tensorflow.keras.optimizers import RMSprop


class ModelulLuiNenea(PacManModel):
    def __init__(self):
        super(ModelulLuiNenea, self).__init__()
        self.__initialize_model()

    def __initialize_model(self):
        self.model = Sequential()

        self.model.add(Dense(512, input_dim=input_size, activation='relu', kernel_initializer='he_uniform'))
        self.model.add(Dense(512, activation='relu', kernel_initializer='he_uniform'))
        self.model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        # self.model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        # self.model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
        self.model.add(Dense(9, activation='softmax', kernel_initializer='he_uniform'))

        self.model.compile(optimizer=RMSprop(0.1), loss='mse')

        # self.model.add(Dense(32, input_dim=input_size, activation='relu', kernel_initializer='he_uniform'))
        # self.model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        # self.model.add(Dense(number_actions,  activation='linear'))
        # # self.model.add(Dense(100, activation='relu'))
        #
        # self.model.compile(optimizer='adam', loss='mean_squared_error')
