from keras.models import load_model
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

(x_train,y_train),(x_test,y_test) = mnist.load_data()

model = load_model("Hand_Written_Digit_RecoModel.h5")

image = x_test[1220]
plt.imshow(image.reshape(28, 28))
y_pred = model.predict(image.reshape(1,28,28,1))
print('Number:',np.argmax(y_pred))