from data_load import *
from deefnet import *
from sklearn.metrics import confusion_matrix
import keras

model = lipi( classes=36)
model.summary()

model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_split = 0.2, epochs=150, batch_size=8)

y_pred = model.predict(x_test)

print(accuracy_score(y_test, y_pred.round()))

matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

print(matrix.diagonal()/matrix.sum(axis=1))

