import tensorflow as tf
from tensorflow import keras # API for tensorflow
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

# split data into training and test sets

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress','Coat',
               'Scandal', 'Shirt', 'Sneaker','Bag', 'Ankle boot']
#train_labels are between 0 to 9 and each refer to a
#different type of clothing item

train_images = train_images / 255.00  #rescaling the image data pixel in order to avoid large numbers
test_images = test_images /255.00


# print(train_images[7])

#plt.imshow(train_images[7], cmap=plt.cm.binary)
#plt.show()

# vectorizing the input array and setting up the model

######################################################################
model = keras.Sequential([keras.layers.Flatten(input_shape=(28,28)),
                          keras.layers.Dense(128, activation="relu"), #setting up the hidden layer and its activation function
                          keras.layers.Dense(10, activation="softmax")
                          ]) #setting up the output layer and its activation function

#setting up the loss/cost function and optimizer + accuracy
model.compile(optimizer="adam", loss= "sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels,
          epochs=5
          ) #order of training matters in the outcome. play with it and twik it


test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Tested Acc:", test_acc)
#######################################################################
prediction = model.predict(test_images)
print(class_names[np.argmax(prediction[0])])
#######################################################################

#plt.imshow(test_images[0], cmap=plt.cm.binary)
#plt.show()


for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction " + class_names[np.argmax(prediction[i])])
    plt.show()
#print(np.size(test_images))







