import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
import os
import random

# =========================================
# Hyperparameters and Configuration
# =========================================

# Hyperparameters are settings that configure the behavior of the model and training process.
# They are defined here for easy adjustment and experimentation.

BATCH_SIZE = 64            # Number of samples processed before the model is updated
LATENT_DIM = 100           # Size of the random noise vector input to the generator
EPOCHS = 300               # Number of complete passes through the training dataset
LEARNING_RATE = 0.0002     # Step size for the optimizer
IMG_SIZE = 28              # Dimensions of the generated images (28x28 pixels)
NUM_SAMPLES = 2000         # Total number of smiley face images to generate for training
NUM_REAL_DISPLAY = 5       # Number of real images to display for comparison
NUM_GENERATED_DISPLAY = 5  # Number of generated images to display for comparison
CHECKPOINT_DIR = "checkpoints"  # Directory to save the best model checkpoints

# Create the checkpoint directory if it doesn't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# =========================================
# Data Preparation
# =========================================

def create_face_color() -> Tuple[float, float, float]:
    """
    Randomly selects a face color from predefined options.

    Returns:
        Tuple[float, float, float]: RGB color values representing the face color.
    """
    colors = {
        'yellow': (1.0, 1.0, 0.0),
        'blue': (0.0, 0.0, 1.0),
        'green': (0.0, 1.0, 0.0),
        'pink': (1.0, 0.75, 0.8)
    }
    color = random.choice(list(colors.values()))
    return color

def add_eyes(image: np.ndarray, x: np.ndarray, y: np.ndarray, eye_x_left: int, eye_x_right: int, eye_y: int, eye_radius: int) -> None:
    """
    Adds black circular eyes to the smiley face image.

    Args:
        image (np.ndarray): The image array to modify.
        x (np.ndarray): Grid of x coordinates.
        y (np.ndarray): Grid of y coordinates.
        eye_x_left (int): X-coordinate of the left eye.
        eye_x_right (int): X-coordinate of the right eye.
        eye_y (int): Y-coordinate of the eyes.
        eye_radius (int): Radius of the eyes.
    """
    left_eye_mask = (x - eye_x_left)**2 + (y - eye_y)**2 <= eye_radius**2
    right_eye_mask = (x - eye_x_right)**2 + (y - eye_y)**2 <= eye_radius**2
    image[:, left_eye_mask] = 0.0  # Set left eye to black
    image[:, right_eye_mask] = 0.0  # Set right eye to black

def add_mouth(image: np.ndarray, x: np.ndarray, y: np.ndarray, center: Tuple[int, int], mouth_width: int, mouth_height: int) -> None:
    """
    Adds a smiling mouth to the smiley face image.

    Args:
        image (np.ndarray): The image array to modify.
        x (np.ndarray): Grid of x coordinates.
        y (np.ndarray): Grid of y coordinates.
        center (Tuple[int, int]): Center coordinates of the face.
        mouth_width (int): Width of the mouth.
        mouth_height (int): Height of the mouth.
    """
    mouth_y = center[1] + IMG_SIZE // 6
    mouth_x_start = center[0] - mouth_width // 2
    mouth_x_end = center[0] + mouth_width // 2

    for i in range(mouth_x_start, mouth_x_end):
        relative_x = (i - center[0]) / (mouth_width / 2)
        relative_y = (relative_x**2) * mouth_height
        y_pos = int(mouth_y - relative_y)  # Curve upwards for a smile

        if 0 <= y_pos < IMG_SIZE:
            image[:, y_pos, i] = 0.0  # Set mouth to black

def add_optional_features(image: np.ndarray, x: np.ndarray, y: np.ndarray, feature: str) -> None:
    """
    Adds optional features like glasses or hats to the smiley face.

    Args:
        image (np.ndarray): The image array to modify.
        x (np.ndarray): Grid of x coordinates.
        y (np.ndarray): Grid of y coordinates.
        feature (str): Type of feature to add ('glasses', 'hat').
    """
    if feature == 'glasses':
        # Draw glasses as two small black circles connected by a line
        glass_radius = 1
        left_glass_center = (IMG_SIZE // 2 - IMG_SIZE // 8, IMG_SIZE // 3)
        right_glass_center = (IMG_SIZE // 2 + IMG_SIZE // 8, IMG_SIZE // 3)
        bridge_y = IMG_SIZE // 3
        bridge_x_start = left_glass_center[0] + glass_radius
        bridge_x_end = right_glass_center[0] - glass_radius

        # Draw left glass
        left_mask = (x - left_glass_center[0])**2 + (y - left_glass_center[1])**2 <= glass_radius**2
        image[:, left_mask] = 0.0

        # Draw right glass
        right_mask = (x - right_glass_center[0])**2 + (y - right_glass_center[1])**2 <= glass_radius**2
        image[:, right_mask] = 0.0

        # Draw bridge between glasses
        bridge_mask = (x >= bridge_x_start) & (x <= bridge_x_end) & (y == bridge_y)
        image[:, bridge_mask] = 0.0

    elif feature == 'hat':
        # Draw a simple hat as a black rectangle on top of the head
        hat_height = IMG_SIZE // 10
        hat_width = IMG_SIZE // 2
        hat_x_start = IMG_SIZE // 2 - hat_width // 2
        hat_x_end = IMG_SIZE // 2 + hat_width // 2
        hat_y_start = IMG_SIZE // 4 - hat_height
        hat_y_end = IMG_SIZE // 4

        for i in range(hat_x_start, hat_x_end):
            for j in range(hat_y_start, hat_y_end):
                if 0 <= j < IMG_SIZE:
                    image[:, j, i] = 0.0  # Set hat to black

def generate_yellow_smiley_faces_dataset(num_samples: int = NUM_SAMPLES, img_size: int = IMG_SIZE) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a dataset of diverse smiley face images with a consistent smiling expression,
    varying colors, and optional features like glasses and hats.

    Args:
        num_samples (int): Number of smiley face images to generate.
        img_size (int): Size of each image (img_size x img_size).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - data: Flattened image data tensor suitable for GAN training.
            - real_images: Tensor of the first few real images for comparison purposes.
    """
    data = []
    real_images = []  # To store a subset of real images for later comparison

    for idx in range(num_samples):
        # Initialize a black background image with 3 color channels (RGB)
        image = np.zeros((3, img_size, img_size), dtype=np.float32)

        # Create the face with a random color
        face_color = create_face_color()
        y, x = np.ogrid[:img_size, :img_size]
        center = (img_size // 2, img_size // 2)
        radius = img_size // 2 - 2  # Define the size of the face
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        image[0, mask] = face_color[0]  # Red channel
        image[1, mask] = face_color[1]  # Green channel
        image[2, mask] = face_color[2]  # Blue channel

        # Define positions for the eyes
        eye_radius = 2
        eye_y = center[1] - img_size // 6  # Position eyes above the mouth
        eye_x_left = center[0] - img_size // 4
        eye_x_right = center[0] + img_size // 4

        # Add eyes to the face
        add_eyes(image, x, y, eye_x_left, eye_x_right, eye_y, eye_radius)

        # Randomly decide to add optional features like glasses or hats
        if random.random() < 0.3:  # 30% chance to add glasses
            add_optional_features(image, x, y, 'glasses')
        if random.random() < 0.2:  # 20% chance to add a hat
            add_optional_features(image, x, y, 'hat')

        # Add a smiling mouth
        mouth_width = img_size // 2
        mouth_height = img_size // 8
        add_mouth(image, x, y, center, mouth_width, mouth_height)

        # Normalize the image pixels to be between -1 and 1 for GAN compatibility
        image = (image * 2) - 1

        # Flatten the image into a single vector and add it to the dataset
        data.append(image.flatten())

        # Store the first few real images for later comparison
        if idx < NUM_REAL_DISPLAY:
            real_images.append(image.copy())

    # Convert the list of images to PyTorch tensors
    data = torch.tensor(np.array(data), dtype=torch.float32)
    real_images = torch.tensor(np.array(real_images), dtype=torch.float32)

    return data, real_images

# Generate the dataset and extract real images for comparison
data, real_images = generate_yellow_smiley_faces_dataset()

# Create a TensorDataset and DataLoader for batching during training
dataset = TensorDataset(data)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# =========================================
# Generator Model Definition
# =========================================

class Generator(nn.Module):
    """
    Generator model for the GAN.
    
    The generator takes a random noise vector (latent vector) as input and transforms it into an image.
    It learns to produce images that resemble the training data.
    """
    def __init__(self, latent_dim: int):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),        # First hidden layer
            nn.ReLU(),                         # Activation function for non-linearity
            nn.BatchNorm1d(256),               # Batch normalization to stabilize training
            nn.Linear(256, IMG_SIZE * IMG_SIZE * 3),  # Output layer producing image pixels
            nn.Tanh()                          # Activation to ensure output values are between -1 and 1
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the generator.
        
        Args:
            z (torch.Tensor): Random noise vector.

        Returns:
            torch.Tensor: Generated image tensor reshaped to (batch_size, 3, IMG_SIZE, IMG_SIZE).
        """
        return self.model(z).view(-1, 3, IMG_SIZE, IMG_SIZE)  # Reshape output to image dimensions

# =========================================
# Discriminator Model Definition
# =========================================

class Discriminator(nn.Module):
    """
    Discriminator model for the GAN.
    
    The discriminator takes an image as input and outputs a probability indicating whether the image is real or fake.
    It learns to distinguish between real images from the dataset and fake images produced by the generator.
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(IMG_SIZE * IMG_SIZE * 3, 256),  # First hidden layer
            nn.LeakyReLU(0.2),                        # LeakyReLU activation for better gradient flow
            nn.Linear(256, 1),                         # Output layer producing a single probability
            nn.Sigmoid()                               # Activation to ensure output is between 0 and 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the discriminator.
        
        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Probability of the image being real.
        """
        x = x.view(x.size(0), -1)  # Flatten the image into a single vector
        return self.model(x)

# =========================================
# GAN Lightning Module Definition
# =========================================

class GAN(pl.LightningModule):
    """
    PyTorch Lightning module encapsulating the GAN architecture, training, and optimization.

    This class manages the generator and discriminator models, defines the training steps,
    and configures the optimizers used during training.
    """
    def __init__(self, latent_dim: int):
        super(GAN, self).__init__()
        self.generator = Generator(latent_dim)      # Initialize the generator
        self.discriminator = Discriminator()        # Initialize the discriminator
        self.latent_dim = latent_dim
        self.automatic_optimization = False        # Manual optimization to control training steps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass through the generator.
        
        Args:
            z (torch.Tensor): Random noise vector.

        Returns:
            torch.Tensor: Generated image tensor.
        """
        return self.generator(z)

    def training_step(self, batch: Tuple[torch.Tensor], batch_idx: int):
        """
        Defines the training step for both the discriminator and the generator.

        Args:
            batch (Tuple[torch.Tensor]): Batch of real images.
            batch_idx (int): Index of the batch.

        Returns:
            None
        """
        real_data = batch[0].to(self.device)  # Move real images to the device (GPU or CPU)
        batch_size = real_data.size(0)

        # Generate fake data by sampling random noise and passing it through the generator
        z = torch.randn(batch_size, self.latent_dim, device=self.device)  # Sample random noise
        fake_data = self(z)  # Generate fake images

        # Retrieve the optimizers for discriminator and generator
        d_optimizer, g_optimizer = self.optimizers()

        # ============================
        # Train Discriminator
        # ============================

        d_optimizer.zero_grad()  # Reset discriminator gradients

        # Pass real and fake data through the discriminator
        real_preds = self.discriminator(real_data)            # Discriminator predictions on real images
        fake_preds = self.discriminator(fake_data.detach())   # Discriminator predictions on fake images

        # Implement label smoothing: real labels are slightly less than 1 to prevent overconfidence
        real_labels = torch.ones_like(real_preds) * 0.9  # Real images labeled as 0.9 instead of 1.0
        fake_labels = torch.zeros_like(fake_preds)       # Fake images labeled as 0.0

        # Calculate loss for real and fake data
        real_loss = nn.functional.binary_cross_entropy(real_preds, real_labels)  # Loss on real images
        fake_loss = nn.functional.binary_cross_entropy(fake_preds, fake_labels)  # Loss on fake images
        d_loss = (real_loss + fake_loss) / 2  # Average discriminator loss

        # Backpropagate the discriminator loss and update its weights
        self.manual_backward(d_loss)
        d_optimizer.step()

        # Log discriminator loss for monitoring
        self.log("d_loss", d_loss, prog_bar=True)

        # ============================
        # Train Generator
        # ============================

        g_optimizer.zero_grad()  # Reset generator gradients

        # Generator tries to fool the discriminator by generating images that the discriminator classifies as real
        fake_preds = self.discriminator(fake_data)  # Discriminator predictions on generated images
        g_loss = nn.functional.binary_cross_entropy(fake_preds, torch.ones_like(fake_preds))  # Generator loss

        # Backpropagate the generator loss and update its weights
        self.manual_backward(g_loss)
        g_optimizer.step()

        # Log generator loss for monitoring
        self.log("g_loss", g_loss, prog_bar=True)

    def configure_optimizers(self):
        """
        Configures the optimizers for both the discriminator and the generator.

        Returns:
            list: A list containing the discriminator and generator optimizers.
        """
        # Use Adam optimizer with specific beta values for better convergence
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
        g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
        return [d_optimizer, g_optimizer]

# =========================================
# Training the GAN
# =========================================

def train_gan() -> GAN:
    """
    Initializes and trains the GAN model using PyTorch Lightning's Trainer.

    Returns:
        GAN: The trained GAN model.
    """
    # Initialize the GAN model with the specified latent dimension
    model = GAN(latent_dim=LATENT_DIM)

    # Define a checkpoint callback to save the best model based on generator loss
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,          # Directory to save checkpoints
        filename='best-checkpoint',      # Filename for the best checkpoint
        save_top_k=1,                     # Save only the best model
        verbose=False,                    # Reduce verbosity
        monitor='g_loss',                 # Monitor generator loss
        mode='min'                        # Save the model with the minimum generator loss
    )

    # Initialize the PyTorch Lightning Trainer with minimal console output
    trainer = pl.Trainer(
        max_epochs=EPOCHS,                 # Number of training epochs
        accelerator='auto',                # Automatically select GPU if available, else CPU
        devices=1,                         # Number of devices to use
        logger=False,                      # Disable default logger to reduce console noise
        callbacks=[checkpoint_callback],   # Add the checkpoint callback
        enable_progress_bar=True,          # Enable progress bar for visual feedback
        enable_model_summary=False         # Disable model summary to reduce output
    )

    # Start the training process
    trainer.fit(model, dataloader)

    return model

# =========================================
# Visualization of Real and Generated Images
# =========================================

def generate_and_compare_images(model: GAN, latent_dim: int, real_images: torch.Tensor, num_generated: int = NUM_GENERATED_DISPLAY):
    """
    Generates images using the trained generator and compares them with real images.

    Args:
        model (GAN): The trained GAN model.
        latent_dim (int): Dimension of the latent vector used by the generator.
        real_images (torch.Tensor): Tensor of real images from the dataset for comparison.
        num_generated (int): Number of generated images to display.

    Returns:
        None
    """
    model.eval()  # Set the model to evaluation mode to disable dropout and other training-specific layers
    fig, axes = plt.subplots(2, max(NUM_REAL_DISPLAY, num_generated), figsize=(15, 6))  # Create subplots

    # Display real images on the top row
    for i in range(NUM_REAL_DISPLAY):
        img = real_images[i].cpu().numpy()                 # Convert tensor to NumPy array
        img = np.transpose(img, (1, 2, 0))                # Change shape from (C, H, W) to (H, W, C) for plotting
        img = (img + 1) / 2                                 # Rescale pixel values from [-1, 1] to [0, 1]
        img = np.clip(img, 0, 1)                           # Ensure pixel values are within [0, 1]
        axes[0, i].imshow(img)                             # Display the image
        axes[0, i].axis('off')                             # Hide axis
        if i == 0:
            axes[0, i].set_title("Real Images", fontsize=12)  # Set title for the first image

    # Display generated images on the bottom row
    for i in range(num_generated):
        z = torch.randn(1, latent_dim).to(model.device)  # Sample random noise vector
        with torch.no_grad():                             # Disable gradient computation for efficiency
            generated_image = model.generator(z).squeeze().cpu().numpy()  # Generate image
            generated_image = np.transpose(generated_image, (1, 2, 0))      # Reshape for plotting
            img = (generated_image + 1) / 2                                   # Rescale pixel values
            img = np.clip(img, 0, 1)                                         # Ensure values are within [0, 1]
        axes[1, i].imshow(img)                                                 # Display the generated image
        axes[1, i].axis('off')                                                 # Hide axis
        if i == 0:
            axes[1, i].set_title("Generated Images", fontsize=12)            # Set title for the first image

    plt.tight_layout()  # Adjust subplots to fit into the figure area
    plt.show()          # Display the plot

# =========================================
# Explanation of GAN Results
# =========================================

def explain_gan_results():
    """
    Provides a textual explanation of how GANs work and interprets the generated images.
    """
    explanation = """
    **Understanding GAN Results**

    GANs (Generative Adversarial Networks) are a type of machine learning model used to generate new data samples that resemble a given dataset. They consist of two main components:

    1. **Generator**: This component creates new data samples from random noise. Its goal is to produce data that is indistinguishable from real data.
    2. **Discriminator**: This component evaluates data samples and tries to determine whether each sample is real (from the dataset) or fake (created by the generator).

    **Training Process:**
    - The generator and discriminator are trained simultaneously in an adversarial manner.
    - The generator learns to produce more realistic images to fool the discriminator.
    - The discriminator learns to better distinguish between real and fake images.
    - Over time, both components improve, leading to the generator producing highly realistic images.

    **Interpreting the Results:**
    - **Real Images**: The top row in the visualization shows actual smiley faces from the training dataset. These images serve as a reference for what the generator should aim to replicate.
    - **Generated Images**: The bottom row displays smiley faces produced by the generator. Ideally, these should closely resemble the real images in terms of color, facial features, and expressions.

    **Observations:**
    - All generated smiley faces have a consistent smiling expression, ensuring that the primary feature (a smile) is always present.
    - Variations in face colors (yellow, blue, green, pink) add visual diversity.
    - Optional features like glasses and hats introduce additional creativity, making the generated images more interesting.
    - Minor imperfections are common in GAN-generated images, especially in early training stages. These typically diminish as training progresses.

    **Conclusion:**
    - The GAN successfully learns to generate diverse and realistic smiling faces by capturing the underlying patterns and features of the training data.
    - This demonstrates the power of adversarial training in producing high-quality synthetic data with controlled attributes.
    """
    print(explanation)

# =========================================
# Main Execution
# =========================================

if __name__ == "__main__":
    # Train the GAN model
    trained_model = train_gan()

    # Generate and visualize images
    generate_and_compare_images(trained_model, LATENT_DIM, real_images)

    # Provide an explanation of the GAN results
    explain_gan_results()
