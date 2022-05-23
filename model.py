import numpy
from PIL import Image
from keras import layers, models, preprocessing

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


if __name__ == '__main__':
    train()
