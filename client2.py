import flwr as fl
import tensorflow as tf
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow_federated as tff

from keras.applications.vgg19 import VGG19
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras.layers import GlobalAveragePooling2D

#from tensorflow_federated.proto.v0 import common_pb2




#grpc_max_message_length = tff.framework.common_pb2.MAX_SUPPORTED_MESSAGE_BYTES


#controller code
from controller import client2_weights
from controller import include_top
from controller import input_shape
from controller import client2_number_of_classes

from controller import client2_training_dir
from controller import client2_testing_dir
from controller import client2_batch_size
from controller import client2_epochs
from controller import client2_verbose
from controller import client2_grpc_max_message_length
from controller import server_address

# Auxiliary methods
def getDist(y):
    ax = sns.countplot(x=y)
    ax.set(title="Count of data classes")
    plt.show()

# Load and compile Keras model
inception = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)

for layer in inception.layers[:10]:
    layer.trainable = False

x = inception.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(client2_number_of_classes, activation='softmax')(x)

model = Model(inputs=inception.input, outputs=predictions)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load dataset
train_dir = client2_training_dir
test_dir =  client2_testing_dir
batch_size = client2_batch_size
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=batch_size, class_mode='sparse')
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=batch_size, class_mode='sparse')

# Visualize data distribution
getDist(train_generator.classes)

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self,config):
        print("Get parameters")
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        r = model.fit(train_generator, epochs=client2_epochs, validation_data=test_generator, verbose=client2_verbose)
        hist = r.history
        print("Fit history : " ,hist)
        return model.get_weights(), train_generator.n, {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(test_generator, verbose=0)
        print("Eval accuracy : ", accuracy)
        return loss, test_generator.n, {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client(
        server_address=server_address, #not added in the controller
        client=FlowerClient(), 
        grpc_max_message_length = client2_grpc_max_message_length * client2_grpc_max_message_length * client2_grpc_max_message_length
)