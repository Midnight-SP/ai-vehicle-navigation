import numpy as np

class ReinforcementLearningModel:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = self.build_model()

    def build_model(self):
        from tensorflow.keras.models import Sequential # type: ignore
        from tensorflow.keras.layers import Dense, Input # type: ignore

        model = Sequential()
        model.add(Input(shape=self.input_shape))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.output_shape, activation='linear'))

        model.compile(optimizer='adam', loss='mse')
        print(model.summary())  # Print model architecture
        return model

    def train(self, x_train, y_train, epochs=1, batch_size=32):
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, state):
        return self.model.predict(state)