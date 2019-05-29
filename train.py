
from keras.applications import MobileNetV2
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input
import numpy as np
import tensorflow as tf


train_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory( directory="/Users/naveekum/Desktop/Codes/Projects/Grab/Data/car_data/train", target_size=(224, 224), 
	color_mode="rgb" , 
	batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

def train():
	input_image_tensor = Input(shape=(224, 224, 3)) 
	base_model = MobileNetV2(input_tensor=input_image_tensor, weights='imagenet', include_top=False)
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	x = Dense(1024, activation='relu')(x)
	predictions = Dense(196, activation='softmax')(x)
	model = Model(inputs=base_model.input, outputs=predictions)
	for layer in base_model.layers:
	    layer.trainable = True

	model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
	print(model.summary())
	
	model.fit_generator(generator=train_generator,steps_per_epoch=8142/32,epochs=10)
	model.save('model.h5')

def test():
	img = image.load_img('/Users/naveekum/Desktop/data_sleeves/sleeveless/im__0.jpg', target_size=(224, 224))
	img = image.img_to_array(img)
	img = np.expand_dims(img, axis=0)
	img = preprocess_input(img)
	model = load_model('model.h5')
	op = model.predict(img)
	print(op)
train()
