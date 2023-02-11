








import numpy as np
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Concatenate
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model, Sequential
from keras.optimizers import Adam

# Load MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train / 127.5 - 1
X_test = X_test / 127.5 - 1

# Text description input
text_input = Input(shape=(1,))

# Embedding layer to transform text descriptions into numerical representations
text_embeddings = Embedding(input_dim=10, output_dim=100)(text_input)

# Flatten the embedded text representation
flatten_text = Flatten()(text_embeddings)

# Generator model
generator = Sequential()
generator.add(Concatenate([flatten_text, Input(shape=(100,))]))
generator.add(Dense(256, activation="relu"))
generator.add(Dense(512, activation="relu"))
generator.add(Dense(1024, activation="relu"))
generator.add(Dense(28 * 28, activation="tanh"))
generator.add(Reshape((28, 28)))

# Discriminator model
discriminator = Sequential()
discriminator.add(Flatten(input_shape=(28, 28)))
discriminator.add(Dense(512, activation="relu"))
discriminator.add(Dense(256, activation="relu"))
discriminator.add(Dense(1, activation="sigmoid"))

# GAN model
gan = Model([text_input, generator.input], discriminator(generator.output))
gan.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.0002, beta_1=0.5))

# Train the GAN model
def train(gan, discriminator, X_train, y_train, noise_dim=100, batch_size=128, epochs=10000):
    for epoch in range(epochs):
        for i in range(len(X_train) // batch_size):
            # Get a batch of real images and their labels
            real_images = X_train[i * batch_size:(i + 1) * batch_size]
            real_labels = y_train[i * batch_size:(i + 1) * batch_size]
            # Generate fake images and labels
            noise = np.random.normal(0, 1, (batch_size, noise_dim))
            fake_labels = np.random.randint(0, 10, (batch_size, 1))
            fake_images = generator.predict([fake_labels, noise])
            # Train the discriminator
            d_loss_real = discriminator
            d_loss_real = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
            d_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((batch_size, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            # Train the generator
            g_loss = gan.train_on_batch([fake_labels, noise], np.ones((batch_size, 1)))
        # Print the losses every 100 epochs
        if epoch % 100 == 0:
            print("Epoch: %d, Discriminator loss: %f, Generator loss: %f" % (epoch, d_loss, g_loss))

# Train the model
train(gan, discriminator, X_train, y_train)


# Define a function to generate images from text descriptions
def generate_images_from_text(model, embedding, text_labels, noise):
    text_embeddings = embedding(text_labels)
    generated_images = model.predict([text_embeddings, noise])
    generated_images = generated_images.reshape(generated_images.shape[0], 28, 28)
    return generated_images

# Generate images from random text descriptions and random noise
random_labels = np.random.randint(0, 10, (10,))
random_noise = np.random.normal(0, 1, (10, 100))
generated_images = generate_images_from_text(gan, embedding, random_labels, random_noise)

# Plot the generated images
for i in range(10):
    plt.imshow(generated_images[i], cmap='gray')
    plt.axis('off')
    plt.show()

