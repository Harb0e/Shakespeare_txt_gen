from tensorflow.keras.callbacks import ModelCheckpoint

class ShTrainer(object):
    def trainMe(self,model,X,y):
        self.cp = ModelCheckpoint('.\\lib\\model',monitor='accuracy',save_best_only = True)
        self.hist = model.fit(X,y,epochs=50,verbose=1,batch_size = 400,callbacks = [self.cp])