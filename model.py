import numpy
import numpy as np
from PIL import Image
from keras import layers, models, preprocessing
from keras.applications.densenet import preprocess_input
from keras.saving.model_config import model_from_json
from keras_preprocessing import image


def train():
    datagen = preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        directory="dataset",
        target_size=(24, 24),
        color_mode="grayscale",
        batch_size=32,
        class_mode="binary",
        shuffle=True,
        seed=42,
        subset='training'
    )

    test_generator = datagen.flow_from_directory(
        directory="dataset",
        target_size=(24, 24),
        color_mode="grayscale",
        batch_size=32,
        class_mode="binary",
        shuffle=True,
        seed=42,
        subset='validation'
    )

    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VALID = test_generator.n // test_generator.batch_size

    model = models.Sequential()

    model.add(layers.Conv2D(6, (3, 3), activation='relu', input_shape=(24, 24, 1)))
    model.add(layers.AveragePooling2D())

    model.add(layers.Conv2D(16, (3, 3), activation='relu'))
    model.add(layers.AveragePooling2D())

    model.add(layers.Flatten())

    model.add(layers.Dense(120, activation='relu'))

    model.add(layers.Dense(84, activation='relu'))

    model.add(layers.Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(train_generator,
              steps_per_epoch=STEP_SIZE_TRAIN,
              validation_data=test_generator,
              validation_steps=STEP_SIZE_VALID,
              epochs=20
              )

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")


def predict(img, model):
    img = Image.open(img)
    img = numpy.array(img)
    img = Image.fromarray(img, 'RGB').convert('L')
    img = numpy.array(img.resize((24, 24))).astype('float32')
    img /= 255
    img = img.reshape(1, 24, 24, 1)
    prediction = model.predict(img)

    print(prediction[0][0])
    if prediction > 0.5:
        prediction = 'open'
    else:
        prediction = 'closed'
    return prediction


if __name__ == '__main__':
    # train()
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(predict('tmp/1.jpg', loaded_model))
    print(predict('tmp/right-77.jpg', loaded_model))