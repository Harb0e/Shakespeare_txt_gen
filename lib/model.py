from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.optimizers import Adam

def createModel(input_size,df,total_words):
    max_len = max([len(x) for x in df])
    model = Sequential()
    model.add(InputLayer((input_size)))
    model.add(Embedding(total_words,125 ,input_length = max_len))
    model.add(Bidirectional(LSTM(225)))
    model.add(Dense(total_words,activation='softmax'))
    model.summary()

    model.compile(optimizer=Adam(learning_rate=0.01),loss='categorical_crossentropy',metrics=['accuracy'])
    return model

