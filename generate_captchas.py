import numpy as np
import tensorflow as tf
import base64

# Set up the neural network architecture
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

# Load the training data and labels
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Train the model
model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))

# Generate and save the CAPTCHAs as PNG files in base64 format
num_captchas = 1
captchas = []
captcha_answers = []
for i in range(num_captchas):
    captcha = np.random.randint(0, 9, size=(28, 28))
    captcha = captcha.astype('float32') / 255
    captcha = np.expand_dims(captcha, axis=2)
    captcha = np.expand_dims(captcha, axis=0)
    prediction = model.predict(captcha)
    captcha_label = np.argmax(prediction)
    captchas.append(captcha)
    captcha_answers.append(captcha_label)

# Save the CAPTCHAs as PNG files in base64 format
for i, captcha in enumerate(captchas):
    png = tf.keras.preprocessing.image.array_to_img(captcha)
    encoded_str = base64.b64encode(png.save())
    print("answer: ", captcha_answers[i])
    print(encoded_str)

