# import the nececessary lib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# data augmentation for the training variable
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D,Flatten,Dense

train_datagen = ImageDataGenerator(rescale =1./255,zoom_range=0.2,horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale =1./255)

# data augmentation on the training data
x_train = train_datagen.flow_from_directory('D:\PROJECTS\Login System\Data\Training',
                                            target_size=(224,224),
                                            class_mode = 'categorical',
                                            batch_size = 100)
# data augmentation on the testing data
x_test = test_datagen.flow_from_directory('D:\PROJECTS\Login System\Data\Testing',
                                            target_size=(224,224),
                                            class_mode = 'categorical',
                                            batch_size = 100)
# adding layers
model = Sequential()
model.add(Convolution2D(32,(3,3),activation='relu',input_shape=(224,224,3)))  #convolution layer
model.add(MaxPooling2D(pool_size =(2,2)))  # maxpooling layer
model.add(Flatten())  # flatten layer
model.add(Dense(300,activation ='relu')) # hidden layer 1
model.add(Dense(150,activation ='relu')) # hidden layer 2
model.add(Dense(14,activation ='softmax')) # output layer

# compile the model
model.compile(optimizer = 'adam',loss= 'categorical_crossentropy',metrics =['accuracy'])

# training the model
model.fit_generator(x_train, steps_per_epoch=len(x_train), epochs=10, validation_data=x_test, validation_steps=len(x_test))

model.save('symbols.h5')