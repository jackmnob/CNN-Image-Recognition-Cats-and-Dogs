"""
About This Dataset:

- The "Cats and Dogs Image Classification" dataset was downloaded from Kaggle on 05/25/2026 by Jack Noble.

- This dataset contains over 1,000 images of dogs and cats in a .jpeg format
(images range from 100x100 - 2,000x1,000 pixels).

- Originally, the images were web-scraped off of Google Images (according to the original author)

- You can learn more about the dataset here:
https://www.kaggle.com/datasets/samuelcortinhas/cats-and-dogs-image-classification

Purpose of Analysis:

- Solve a simple binary image classification problem by using an images' pixel data to determine the
correct classification (cat or dog) of a given image.

- Develop a Convolution Neural Network model (CNN) that can accurately predict future images
by learning from a subset of image metadata.

Now, let's begin!
"""

# ===================== Modules Setup =====================
# Read in the required modules and change our default pandas settings:
try:  #  Use the "try" flow control argument to "try" and import all necessary packages
    # Basic operating system packages
    # 'os' - Provides functions for interacting with the operating system
    import os

    # Math and statistics packages
    # 'pandas' - Data manipulation and analysis library, renamed to 'pd'
    import pandas as pd
    # 'numpy' - Numerical computations library, renamed to 'np'
    import numpy as np
    # 'statistics' - a built-in python library, used to compute basic and advanced statistics
    import statistics

    # Graphing and visualization packages
    # 'matplotlib.pyplot' - Plotting graphs and figures, renamed to 'plt'
    import matplotlib.pyplot as plt
    # 'ticker' from 'matplotlib' - Provides tools to configure axis ticks
    import matplotlib.ticker as ticker
    # 'seaborn' - Statistical data visualization library, renamed to 'sns'
    import seaborn as sns

    # Machine Learning packages (from PyTorch)
    # 'torch' - Core PyTorch library used to build and train neural networks
    import torch
    # 'torchvision' - PyTorch computer vision library containing image datasets, transforms, and models
    import torchvision
    # 'transforms' - Provides image preprocessing functions such as resizing, normalization, and tensor conversion
    from torchvision import transforms
    # 'ImageFolder' - Automatically loads images from folders and assigns labels based on folder names
    from torchvision.datasets import ImageFolder
    # 'Image' - Opens, reads, and manipulates image files
    from PIL import Image
    # 'DataLoader' - Loads datasets into batches for efficient model training and testing
    from torch.utils.data import DataLoader
    # ????
    import torch.nn as nn
    # ????
    import torch.nn.functional as F

    # 'train_test_split' - Splits data into training and testing subsets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    # Print success message indicating successful imports
    print('All modules successfully imported.')

#  Use an "except" clause to catch specific import errors and rename the error to 'moderror'
except ModuleNotFoundError as moderror:
    print('Module failed to import: ' + str(moderror.name))
#  Use an "except" clause to catch any unexpected errors
except Exception as ex:
    print('Unexpected error occurred during imports: ' + str(ex))

# ===================== Pandas: Display Configuration =====================
def configure_pd_display():
    """Configures pandas display settings for better visibility of datasets within the IDE:
    - Sets 'display.max_columns' to None to ensure all columns are shown when printing DataFrames
    - Sets 'display.width' to a large value to prevent line breaks in wide DataFrames
    - Sets 'display.max_rows' to 55 to limit the number of rows shown when printing DataFrames (useful for sampling)
    - Disables scientific notation for better readability of numerical data
    - Includes error handling to catch issues with pandas configuration settings
    Args:
        var = None (this function does not take any arguments)
    Returns:
        var = None (this function does not return any values)
    """
    #  Use the "try" flow control argument to "try" and adjust the display settings for pandas within the IDE
    try:
        # Configure pandas display settings for better visibility of datasets within the IDE
        # 'display.max_columns' - Ensures all columns are displayed when printing DataFrames
        pd.set_option('display.max_columns', None)
        # 'display.width' - Sets the display width to accommodate large DataFrames without truncation
        pd.set_option('display.width', 2000)
        # 'display.max_rows' - Limits the number of rows displayed when printing DataFrames (useful for sampling)
        pd.set_option('display.max_rows', 55)
        # Disable scientific notation for better readability
        pd.set_option('display.float_format', '{:.2f}'.format)

        # Print success message indicating display settings were applied
        print('Display settings successfully applied.')

    #  Use an "except" clause to catch if pandas ('pd') is not defined or imported incorrectly
    except NameError:
        print('Error: Pandas (\'pd\') is not defined. Ensure that pandas is imported using: import pandas as pd.')

    #  Use an "except" clause to catch if the provided 'pd.set_option()' settings are incorrect or invalid
    except ValueError as valErr:
        print('Error: Invalid value provided to \'pd.set_option()\'. Details: ' + str(valErr))

    #  Use an "except" clause to catch unexpected errors during the settings configuration process
    except Exception as ex:
        print('Unexpected error occurred while configuring display settings: ' + str(ex))

# Call the function to configure pandas display settings
configure_pd_display()

# ===================== Directory Initialization =====================
def initialize_directory():
    """Initializes the directory path for the dataset and checks if it exists:
    - Attempts to locate the directory of the dataset using the provided path
    - If the directory is found, it lists the subfolders and establishes paths for training and testing datasets
    Returns:
        var = A variable containing the directory path, training path, testing path, and list of folders.
    """
    # Initialize our directory path where the dataset is stored:
    directory_path = r'C:\Users\jackn\Desktop\Projects\Portfolio\ML\Image Recognition\data'

    # Attempt to locate the *directory* of the dataset (using os.path.isdir()) via the provided path (located above)
    if not os.path.isdir(directory_path):
        # Print an error message for our user if the file path is NOT found (concatenate our filePath variable to str())
        print('Error - the directory at: ' + directory_path + ' was not found!')
    # Otherwise, if our file is located:
    else:
        # Print our success message for user feedback
        print('Directory successfully located.')

    # Use pandas (and our directory_path path variable) to read our directory and list the subfolders
    folder_list = []  # Initialize an empty list to store the names of each directory object

    # Loop over each folder in the directory (using os.listdir())
    for folder in os.listdir(directory_path):
        folder_list.append(folder)  # Append the name of each folder to our folder_list variable
    print('List of folders in the directory: ' + str(folder_list))

    # Establish the directory paths together
    train_path = os.path.join(directory_path, folder_list[1])
    test_path = os.path.join(directory_path, folder_list[0])

    # Print the paths of the folders within the directory
    print(train_path + '\n' + test_path)

    return directory_path, train_path, test_path, folder_list

# Call the function to initialize our directory and store the returned variables
directory_path, train_path, test_path, folder_list = initialize_directory()

# ===================== EDA: Viewing Image Samples =====================
"""
# %%
# Set the directories for the training set on each category
cat_folder = os.path.join(train_path, 'cats')
dog_folder = os.path.join(train_path, 'dogs')


# Initialize an empty list to store the filepaths of each sample
sample_paths = []

# Grab three samples from the training set (for cats)
for file in os.listdir(cat_folder)[:3]:
    sample_paths.append(os.path.join(cat_folder, file))

# Grab three samples from the training set (for dogs)
for file in os.listdir(dog_folder)[:3]:
    sample_paths.append(os.path.join(dog_folder, file))

# Create a subplot area with 2 vertical axes, and three horizontal
fig, axes = plt.subplots(2, 3, figsize=(10, 6))

# For each image in the range of sample images (3)
for image in range(len(sample_paths)):
    # Open the image from the path
    img = Image.open(sample_paths[image])
    # Display the image on the axis (.imshow())
    axes[image // 3, image % 3].imshow(img)
    axes[image // 3, image % 3].axis('off')
# Show the plot of images
plt.show()
# %%
"""

# ===================== EDA: Checking for Class Imbalances =====================
# Check how many images exist in each set of our data (class imbalances)
print("Train Cats: " + str(len(os.listdir(train_path + '\\cats'))))  # Check the training set
print("Train Dogs: " + str(len(os.listdir(train_path + '\\dogs'))))
print("Test Cats: " + str(len(os.listdir(test_path + '\\cats'))))    # Check the testing set
print("Test Dogs: " + str(len(os.listdir(test_path + '\\dogs'))))

# ===================== EDA: Analyzing Image Metadata =====================
def image_quality_check(train_path):
    """Perform an image quality check by analyzing the metadata of each image in the training set:
    - Extract image dimensions (width and height) and color modes (RGB, L, etc.)
    - Identify any non-RGB images and corrupt/unreadable files
    - Summarize the findings with statistics and lists of any problematic files
    Args:
        var = "train_path" (the file path to the training dataset)
        Returns: None (this function does not return any values, but prints the results to the console)
    """
    # Initialize lists to store image metadata and quality checks
    widths = []          # Store image widths
    heights = []         # Store image heights
    modes = []           # Store image color modes (RGB, L, etc.)
    non_rgb_files = []   # Store any files that are not RGB
    bad_files = []       # Store any corrupt/unreadable image files

    # Loop through each image category
    for category in ['cats', 'dogs']:
        # Build the full folder path for the current category
        folder = os.path.join(train_path, category)

        # Loop through every image file in the current category
        for file in os.listdir(folder):
            # Build the full path to the image
            img_path = os.path.join(folder, file)
            try:
                # Open the image
                img = Image.open(img_path)
                # Store image dimensions for later analysis
                widths.append(img.size[0])
                heights.append(img.size[1])
                # Store image color mode
                modes.append(img.mode)
                # Check if the image is not RGB
                if img.mode != 'RGB':
                    # Save the file information for review
                    non_rgb_files.append(
                        category + ': ' + file + ' | mode: ' + img.mode
                    )
                # Verify the image is not corrupt
                img.verify()
            # If an error occurs, store the file path as a bad file
            except Exception as error:
                bad_files.append(img_path)

    # Display image summary statistics (quality check)
    print('Image summary statistics: ' + '\n')

    # Display image color mode summary
    print(str(pd.Series(modes).value_counts()))

    # Display any non-RGB files that were found
    print('\n' + 'Non-RGB Files: ' + str(non_rgb_files) + '\n')

    # Display any corrupt image files that were found
    print('Bad Files Found: ' + str(len(bad_files)) + '\n' + 'Bad Files: ' + str(bad_files) + '\n')

    # Display image width statistics
    print('Min Width: ' + str(min(widths)))
    print('Max Width: ' + str(max(widths)))
    print('Mean Width: ' + str(round(np.mean(widths), 2)))
    print('Median Width: ' + str(round(np.median(widths), 2)) + '\n')

    # Display image height statistics
    print('Min Height: ' + str(min(heights)))
    print('Max Height: ' + str(max(heights)))
    print('Mean Height: ' + str(round(np.mean(heights), 2)))
    print('Median Height: ' + str(round(np.median(heights), 2)))

# Call the function to perform the image quality check and store the returned metadata
image_quality_check(train_path)

# ===================== EDA: Analyzing Image Metadata (Sizing/Pixel Map) =====================
"""
# %%
# Visualize pixel distribution across the dataset (training set)
fig, axes = plt.subplots(1, 2, figsize=(8, 4))  # Create a figure area with 1 row and 2 columns

# Width distribution (first column on the plot area)
axes[0].hist(widths, bins=20, color='black')
axes[0].axvline(np.mean(widths), color='red', linestyle='--', linewidth=2, label='Mean')
axes[0].axvline(np.median(widths), color='green', linestyle='-.', linewidth=2,label='Median')
axes[0].set_title('Training Set: Image Width Distribution')
axes[0].set_xlabel('Width (pixels)')
axes[0].set_ylabel('Frequency')
axes[0].legend()

# Height distribution (first column on the plot area)
axes[1].hist(heights, bins=20, color='black')
axes[1].axvline(np.mean(heights), color='red', linestyle='--', linewidth=2, label='Mean')
axes[1].axvline(np.median(heights), color='green', linestyle='-.', linewidth=2, label='Median')
axes[1].set_title('Training Set: Image Height Distribution')
axes[1].set_xlabel('Height (pixels)')
axes[1].set_ylabel('Frequency')
axes[1].legend()

# Prevent overlap
plt.tight_layout()

# Display plots
plt.show()
# %%
"""

# ===================== Pre-Processing: Image Standardization =====================
# Define a function to convert images to RGB format (if needed)
def convert_to_rgb(image):
    """# Convert any image to RGB format (in case of greyscaled images)
    Args:
        var = "image" (the input image to be converted to RGB format)
    Returns:
        var = "image" (the now converted image - in RGB format)
    """
    return image.convert('RGB')

def transform_images():
    """ Define the image transformations for training and testing datasets:
    - Resize all images to a standard size (e.g., 128x128 pixels)
    - Convert images to RGB format (if needed)
    - Augment training images with random horizontal flips and rotations
    - Normalize pixel values to a standard range (e.g., 0 to 1 or -1 to 1)
    - Convert images to PyTorch tensors for CNN model input
    Args:
        var = None (this function does not take any arguments)
    Returns:
        var = "train_transform" (the image transformations to apply to the training dataset)
        var = "test_transform" (the image transformations to apply to the testing dataset)
    """
    # Now, let's standardize our image dimensions:
    image_size = 128  # Set to 128x128 pixels

    # Set any images in the training set to RGB, augment them, resize them, normalize them, and turn them into Tensors
    train_transform = transforms.Compose([
        transforms.Lambda(convert_to_rgb),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    print('Training images will be converted to RGB, resized to '
        + str(image_size)
        + 'x'
        + str(image_size)
        + 'px, normalized, and converted to tensors when loaded.')

    # Set any images in the testing set to RGB, resize them, normalize them, and turn them into Tensors
    test_transform = transforms.Compose([
        transforms.Lambda(convert_to_rgb),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    print('Testing images will be converted to RGB, resized to '
        + str(image_size) + 'x'
        + str(image_size) + 'px, normalized, and converted to tensors when loaded.')

    return train_transform, test_transform

# Call the function to define the image transformations for training and testing datasets
train_transform, test_transform = transform_images()

def load_and_transform_data(train_path, test_path, train_transform, test_transform):
    """Loads the training and testing datasets using PyTorch's ImageFolder class, applying the defined transformations:
        Args:
            var = "train_path" (the file path to the training dataset)
            var = "test_path" (the file path to the testing dataset)
            var = "train_transform" (the image transformations to apply to the training dataset)
            var = "test_transform" (the image transformations to apply to the testing dataset)
        Returns:
            var = "train_dataset" (the loaded and transformed training dataset)
            var = "test_dataset" (the loaded and transformed testing dataset)
    """
    #  Use the "try" flow control argument to "try" and import all necessary packages
    try:
        # Load the datasets
        train_dataset = ImageFolder(root=train_path, transform=train_transform)  # training
        print('Training dataset successfully loaded.')
        test_dataset = ImageFolder(root=test_path, transform=test_transform)     # testing
        print('Testing dataset successfully loaded.')
    #  Use an "except" clause to catch any unexpected errors
    except Exception as ex:
        print('Error loading datasets.')

    return train_dataset, test_dataset

# Call the function to load and transform the datasets
train_dataset, test_dataset = load_and_transform_data(train_path, test_path, train_transform, test_transform)

## -- -- ================== MORE EDITS NEEDED BELOW ================== -- -- ##
## ================== -- --  ================== -- -- ================ -- -- ##
# Check how PyTorch interpreted the image classes
print('Training classes: ' + str(train_dataset.classes))
print('Testing classes: ' + str(test_dataset.classes) + '\n')

# Check how PyTorch numerically encoded the classes
print('Training class mapping: ' + str(train_dataset.class_to_idx))
print('Testing class mapping: ' + str(test_dataset.class_to_idx) + '\n')

# Check the number of images loaded into each dataset
print('Training images loaded: ' + str(len(train_dataset)) + '\n')
print('Testing images loaded: ' + str(len(test_dataset)) + '\n')

# Pull one transformed image and label from the training dataset
sample_image, sample_label = train_dataset[0]

# Display the transformed image shape and label
print('Sample image tensor shape: ' + str(sample_image.shape))
print('Sample image label: ' + str(sample_label))

# Set the batch size for model training
batch_size = 32

# Create the training DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create the testing DataLoader
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Confirm the DataLoaders were created
print('Training DataLoader created with batch size: ' + str(batch_size))
print('Testing DataLoader created with batch size: ' + str(batch_size))

# Pull the first batch of images and labels from the training set (DataLoader object)
for images, labels in train_loader:
    print('Successfully loaded first batch!' + '\n')
    break

# Image batch shape format: [Batch Size, Color Channels, Image Height, Image Width]
print('Image batch shape: ' + str(images.shape) + '\n')
print('Label batch shape: ' + str(labels.shape))

# Now, let's define our CNN model architecture from scratch (using PyTorch's nn.Module class).:
class SimpleCNN(nn.Module):  # Create a basic CNN model from scratch

    # Define the structure/layers of the neural network
    def __init__(self):

        # Initialize the parent PyTorch neural network class
        super(SimpleCNN, self).__init__()

        # Convolutional layers apply filters to images.
        # Example: one filter may detect vertical edges, another may detect horizontal edges, another may detect curves, and another may detect dark/light transitions.
        # These learned filters create "feature maps" that help the model recognize cats vs dogs.

        # First convolutional layer: (input = 3 RGB channels; Output = 16 feature maps)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)

        # Second convolutional layer: (this layer can combine simpler features into more complex patterns like eyes, ears, fur, etc.)
        # Input = 16 feature maps, Output = 32 feature maps.
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        # Pooling layer reduces image dimensions by half each time it is applied (ex: 128x128 -> 64x64 -> 32x32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layer: (after two pooling steps, image size is 32x32 with 32 feature maps - *see above explanation*)
        # 32 feature maps * 32 height * 32 width = 32,768 input values.
        self.fc1 = nn.Linear(32 * 32 * 32, 128)

        # Output layer: (2 outputs because this is a binary classification problem - e.g., cats vs dogs.)
        self.fc2 = nn.Linear(128, 2)

    # Define how images move through the network from input to prediction:
    def forward(self, x):

        # First block: convolution -> ReLU activation -> pooling (shape changes from [batch size, 3, 128, 128] to [batch size, 16, 64, 64])
        x = self.pool(F.relu(self.conv1(x)))

        # Second block: convolution -> ReLU activation -> pooling (shape changes from [batch size, 16, 64, 64] to [batch size, 32, 32, 32])
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten the feature maps into one long vector before the dense layers (shape changes from [batch size, 32, 32, 32] to [batch size, 32768])
        x = x.view(x.size(0), -1)

        # First dense layer learns combinations of the extracted image features
        x = F.relu(self.fc1(x))

        # Final dense layer returns 2 raw prediction scores, one for each class
        x = self.fc2(x)

        return x

print('CNN class and feed-forward function successfully defined.')

# Set the device to GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create the CNN model and send it to the device
model = SimpleCNN().to(device)

# Confirm where the model is stored
print('Model created and sent to: ' + str(device))

# Display the model architecture
print(model)

# Define the loss function ('CrossEntropyLoss' is commonly used for classification - though, usually multi-class problems.)
criterion = nn.CrossEntropyLoss()

# Define the optimizer
# Adam updates the model weights during training.
learning_rate = 0.001

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)

print('Loss function defined: CrossEntropyLoss')
print('Optimizer defined: Adam')
print('Learning rate: ' + str(learning_rate))

# Set the number of full passes through the training dataset
num_epochs = 20

# Train the CNN model
for epoch in range(num_epochs):

    # Set model to training mode
    model.train()

    # Track total loss for the current epoch
    running_loss = 0.0

    # Loop through each batch of images and labels
    for images, labels in train_loader:

        # Move images and labels to GPU/CPU device
        images = images.to(device)
        labels = labels.to(device)

        # Clear old gradients from the previous batch
        optimizer.zero_grad()

        # Forward pass: generate predictions
        outputs = model(images)

        # Calculate prediction error
        loss = criterion(outputs, labels)

        # Backward pass: calculate gradients
        loss.backward()

        # Update model weights
        optimizer.step()

        # Add batch loss to running loss
        running_loss = running_loss + loss.item()

    # Calculate average loss for the epoch
    average_loss = running_loss / len(train_loader)

    print('Epoch ' + str(epoch + 1) + '/' + str(num_epochs) + ' - Loss: ' + str(round(average_loss, 4)))

    # Set model to evaluation mode
model.eval()

# Initialize counters for correct predictions and total observations
correct = 0
total = 0

# Turn off gradient calculations during testing
with torch.no_grad():

    # Loop through each batch in the testing DataLoader
    for images, labels in test_loader:

        # Move images and labels to GPU/CPU device
        images = images.to(device)
        labels = labels.to(device)

        # Generate model predictions
        outputs = model(images)

        # Select the class with the highest prediction score
        _, predicted = torch.max(outputs, 1)

        # Count total labels
        total = total + labels.size(0)

        # Count correct predictions
        correct = correct + (predicted == labels).sum().item()

# Calculate testing accuracy
test_accuracy = 100 * correct / total

print('Correct Predictions: ' + str(correct))
print('Total Predictions: ' + str(total))
print('Test Accuracy: ' + str(round(test_accuracy, 2)) + '%')

# Use the "try" flow control argument to "try" and generate our confusion matrix
try:
    # Set model to evaluation mode
    model.eval()

    # Initialize empty lists to store actual and predicted labels
    y_test = []
    y_pred = []

    # Turn off gradient calculations during testing
    with torch.no_grad():

        # Loop through each batch in the testing DataLoader
        for images, labels in test_loader:

            # Move images and labels to GPU/CPU device
            images = images.to(device)
            labels = labels.to(device)

            # Generate model prediction scores
            outputs = model(images)

            # Select the class with the highest score as the prediction
            _, predicted = torch.max(outputs, 1)

            # Move labels and predictions back to CPU and store them
            y_test.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Create a confusion matrix comparing the true labels and predicted labels
    cm = confusion_matrix(y_test, y_pred)

    # Create the confusion matrix display
    cm_display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=train_dataset.classes
    )

    # Plot the confusion matrix
    cm_display.plot()

    # Set the title of the confusion matrix
    plt.title('Confusion Matrix - CNN Model')

    # Print summary statistics again:
    print('Correct Predictions: ' + str(correct))
    print('Total Predictions: ' + str(total))
    print('Test Accuracy: ' + str(round(test_accuracy, 2)) + '%' + '\n')

    # Display the confusion matrix plot
    plt.show()

# Handle any errors that may occur during prediction or evaluation
except Exception as ex:
    # Print an error message if any occur
    print('Error occurred during model evaluation: ' + str(ex))

