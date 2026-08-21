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
    from torch.utils.data import DataLoader, Subset
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

# ===================== Directory Initialization =====================
def initialize_directory():
    """Initializes the dataset directory and creates the training/testing paths.

    This function:
    - Defines the main dataset directory
    - Confirms that the main directory exists
    - Lists the folders contained inside the directory
    - Creates explicit paths for the training and testing datasets
    - Confirms that the expected training and testing folders exist

    Args:
        var = None (this function does not take any arguments)

    Returns:
        var = "directory_path" (the main dataset directory)
        var = "train_path" (the path to the training dataset)
        var = "test_path" (the path to the testing dataset)
        var = "folder_list" (a list of folders contained in the main dataset directory)
    """
    # Initialize our directory path where the dataset is stored:
    directory_path = r'C:\Users\jackn\Desktop\Projects\Portfolio\ML\Image Recognition\data'

    # Attempt to locate the *directory* of the dataset (using os.path.isdir()) via the provided path (located above)
    if not os.path.isdir(directory_path):
        # Raise a FileNotFoundError for our user if the file path is NOT found
        raise FileNotFoundError('Error - the directory at: ' + directory_path + ' was not found!')
    # Otherwise, if our file is located:
    else:
        # Print our success message for user feedback
        print('Directory successfully located.')

    # Store the names of all folders contained in the dataset directory
    folder_list = []

    # Loop over each folder in the directory (using os.listdir())
    for folder in os.listdir(directory_path):
        # Build the full path to the current object
        folder_path = os.path.join(directory_path, folder)
        # Only add the object if it is actually a directory
        if os.path.isdir(folder_path):
            folder_list.append(folder)
    print('List of folders in the directory: ' + str(folder_list))

    # Create explicit paths for the training and testing directories
    train_path = os.path.join(directory_path, 'train')
    test_path = os.path.join(directory_path, 'test')

    # Confirm that the expected training folder exists
    if not os.path.isdir(train_path):
        raise FileNotFoundError('The training directory was not found at: ' + train_path)

    # Confirm that the expected testing folder exists
    if not os.path.isdir(test_path):
        raise FileNotFoundError('The testing directory was not found at: ' + test_path)

    # Print the paths of the folders within the directory
    print("Training directory: " + train_path + '\n' + "Testing directory: " + test_path)

    return directory_path, train_path, test_path, folder_list

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
    """ Define the image transformations for training, validation, and testing datasets:
    Training images:
    - Resize all images to a standard size (e.g., 128x128 pixels)
    - Convert images to RGB format
    - Augment training images with random horizontal flips and rotations
    - Normalize pixel values to a standard range (e.g., 0 to 1 or -1 to 1)
    - Convert images to PyTorch tensors for CNN model input

    Validation/testing images:
    - Same process but WITHOUT random augmentations (to ensure consistent evaluation)


    Args:
        var = None (this function does not take any arguments)
    Returns:
        var = "train_transform" (the image transformations to apply to the training dataset)
        var = "test_transform" (the image transformations to apply to the testing dataset)
    """
    # Now, let's standardize our image dimensions:
    training_image_size = 128  # Set to 128x128 pixels
    evaluation_image_size = 144  # Set to 144x144 pixels

    # Set any images in the training set to RGB, augment them, resize them, normalize them, and turn them into Tensors
    train_transform = transforms.Compose([
        transforms.Lambda(convert_to_rgb),
        # Randomly modify image brightness, contrast, and saturation to expose the CNN to slightly different versions of each image
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        # Randomly crop and resize portions of the image while maintaining aspect ratio
        transforms.RandomResizedCrop(size=training_image_size, scale=(0.75, 1.0), ratio=(0.85, 1.15)),
        # Randomly mirror approximately 50% of training images horizontally
        transforms.RandomHorizontalFlip(p=0.5),
        # Randomly rotate images between approximately -10 and +10 degrees
        transforms.RandomRotation(10),
        # Convert images into PyTorch tensors
        transforms.ToTensor(),
        # Normalize RGB pixel values to a range of -1 to 1 (mean=0.5, std=0.5 for each channel)
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    print('Training images will be converted to RGB, resized to '
        + str(training_image_size) + 'x'
        + str(training_image_size) + 'px, normalized, and converted to tensors when loaded.')

    # Set any images in the testing set to RGB, resize them, normalize them, and turn them into Tensors
    evaluation_transform = transforms.Compose([
        # Convert any non-RGB images into RGB format
        transforms.Lambda(convert_to_rgb),
        # Resize images slightly larger than the CNN input dimensions
        transforms.Resize((evaluation_image_size, evaluation_image_size)),
        # Crop the center 128x128 pixels
        transforms.CenterCrop(training_image_size),
        # Convert images into PyTorch tensors
        transforms.ToTensor(),
        # Same normalization used on training images (RGB pixel values from: -1 to 1 (mu=0.5, st.dev=0.5 for each channel)
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    print('Evaluation images will be converted to RGB, cropped, and resized to '
        + str(training_image_size) + 'x'
        + str(training_image_size) + 'px without random augmentation (rotation/flipping).')

    return train_transform, evaluation_transform

# ===================== Pre-Processing: Dataset Loading and Splitting =====================
def load_and_split_data(train_path, test_path, train_transform, evaluation_transform, validation_size=0.15, random_state=42):
    """Loads the image datasets and splits the training data into separate training and validation subsets.

    This function:
    - Reads image paths and class labels from the training directory
    - Splits the training data into training and validation indices
    - Uses stratification to preserve the cat/dog class distribution
    - Applies random augmentation only to the training images
    - Applies deterministic preprocessing to validation and testing images
    - Keeps the testing directory completely separate

    Args:
        var = "train_path" (the file path to the training dataset)
        var = "test_path" (the file path to the testing dataset)
        var = "train_transform" (random transformations applied to training images)
        var = "evaluation_transform" (non-random transformations applied to validation/testing images)
        var = "validation_size" (the percentage of training data assigned to validation)
        var = "random_state" (the random seed used to reproduce the same split)

    Returns:
        var = "train_dataset" (the final training subset)
        var = "validation_dataset" (the final validation subset)
        var = "test_dataset" (the separate testing dataset)
        var = "class_names" (the image class names)
        var = "class_to_idx" (the numerical mapping assigned to each class)
    """

    try:
        # Load the original training directory without transformations.
        #
        # This version is used only to obtain:
        # - The complete list of image observations
        # - The numerical class label assigned to each image
        base_train_dataset = ImageFolder(root=train_path)

        # Store the numerical class labels assigned by ImageFolder: cats = 0, dogs = 1
        targets = np.array(base_train_dataset.targets)

        # Create an array containing the index of every image in the original training directory.
        # Example: [0, 1, 2, 3, ..., 556]
        all_indices = np.arange(len(base_train_dataset))

        # Split the original training indices into separate training and validation groups.
        # test_size=validation_size: Sends 15% of the original training images to validation when validation_size is set to 0.15.
        # stratify=targets: Preserves approximately the same cat/dog percentage in both the training and validation subsets.
        # random_state: Produces the same split every time the program runs for development purposes.
        train_indices, validation_indices = train_test_split(
            all_indices, test_size=validation_size,
            random_state=random_state,stratify=targets
            )

        # Load the original training directory with random training augmentation enabled.
        train_dataset_full = ImageFolder(root=train_path, transform=train_transform)

        # Load the same original training directory again using deterministic evaluation transformations.
        # This prevents validation images from receiving random rotation, flipping, cropping, or color augmentation.
        validation_dataset_full = ImageFolder(root=train_path, transform=evaluation_transform)

        # Create the final training subset using only the indices assigned to training.
        train_dataset = Subset(train_dataset_full, train_indices)

        # Create the final validation subset using only the indices assigned to validation.
        validation_dataset = Subset(validation_dataset_full, validation_indices)

        # Load the completely separate testing directory using deterministic evaluation transformations.
        #
        # The test dataset is not involved in training, model selection, or early stopping.
        test_dataset = ImageFolder(root=test_path, transform=evaluation_transform)

        # Store the class names and numerical mappings before returning.
        # These are stored separately because PyTorch Subset objects do not directly provide .classes or .class_to_idx attributes.
        class_names = base_train_dataset.classes
        class_to_idx = base_train_dataset.class_to_idx

        print('Training, validation, and testing datasets successfully loaded.')

        return(train_dataset, validation_dataset, test_dataset, class_names, class_to_idx)
    # Raise the original error instead of only printing a message.
    except Exception as ex:
        # This prevents later sections from trying to use datasets that were never successfully created.
        raise RuntimeError('Error occurred while loading and splitting the datasets: ' + str(ex))

# Now, let's define our CNN model architecture from scratch (using PyTorch's nn.Module class):
class ImprovedCNN(nn.Module):
    """Creates an improved Convolutional Neural Network (CNN) for binary image classification.

    This CNN is designed to classify images as either cats or dogs.

    The model contains:
    - Four convolutional layers for learning increasingly complex image features
    - Batch normalization layers to stabilize and improve training
    - Max-pooling layers to reduce the spatial dimensions of the feature maps
    - Adaptive average pooling to summarize each final feature map into one value
    - A fully connected layer for combining the learned image features
    - Dropout to reduce overfitting
    - A final output layer containing one raw prediction score for each class

    Expected input shape:
        [batch_size, 3, 128, 128]

    Expected output shape:
        [batch_size, 2]
    """

    # Define the structure and layers of the neural network
    def __init__(self):

        # Initialize the parent PyTorch neural network class
        super(ImprovedCNN, self).__init__()

        # Convolutional layers apply learnable filters to images:
        #
        # Early convolutional layers may learn simple features such as:
        # - Vertical edges or horizontal edges, curves, and color transitions
        #
        # Deeper convolutional layers can combine those simpler features into more complex patterns such as:
        # - Fur textures, eyes, ears, whiskers, noses, and face/body shapes
        #
        # The learned outputs of convolutional layers are called feature maps.

        # First convolutional layer:
        # Input = 3 RGB color channels
        # Output = 32 learned feature maps
        #
        # Padding of 1 keeps the image dimensions at 128x128 after convolution.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)

        # First batch normalization layer (Normalizes the 32 feature maps produced by conv1):
        # Batch normalization can:
        # - Make training more stable, and faster, and reduce sensitivity to the initial model weights
        self.bn1 = nn.BatchNorm2d(32)

        # Second convolutional layer (input = 32 feature maps, output = 64 feature maps):
        # This layer can combine simple features learned by conv1
        # into more detailed patterns such as fur, curves, and small shapes.
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # Second batch normalization layer (normalizes the 64 feature maps produced by conv2):
        self.bn2 = nn.BatchNorm2d(64)

        # Third convolutional layer (input = 64 feature maps, output = 128 feature maps):
        # This deeper layer can combine earlier patterns into more meaningful
        # animal features such as eyes, ears, noses, and facial structures.
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # Third batch normalization layer (Normalizes the 128 feature maps produced by conv3):
        self.bn3 = nn.BatchNorm2d(128)

        # Fourth convolutional layer (input = 128 feature maps, output = 256 feature maps):
        # This layer can learn higher-level combinations of features that may
        # help distinguish the overall appearance of cats from dogs.
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)

        # Fourth batch normalization layer (normalizes the 256 feature maps produced by conv4):
        self.bn4 = nn.BatchNorm2d(256)

        # Max-pooling layer (reduces the height and width of each feature map by half):
        # For example: 128x128 -> 64x64, then 64x64 -> 32x32, then 32x32 -> 16x16, then 16x16 -> 8x8

        # Pooling reduces computation while retaining the strongest or most important detected features.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Adaptive average pooling layer (reduces every final 8x8 feature map into a single average value):
        # Before adaptive pooling: [batch_size, 256, 8, 8] after adaptive pooling: [batch_size, 256, 1, 1]
        # This prevents the model from needing a very large fully connected layer containing millions of parameters.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # First fully connected layer (receives one summarized value from each of the 256 feature maps):
        # Input = 256 learned image features, output = 128 combined features
        self.fc1 = nn.Linear(in_features=256,out_features=128)

        # Dropout layer (randomly disables 40% of the 128 values during each training pass):
        # This reduces the model's ability to memorize the training images and can improve performance on new, unseen images.
        # Dropout is active during model.train(), but is automatically disabled during model.eval().
        self.dropout = nn.Dropout(p=0.4)

        # Final output layer:
        # Input = 128 values from the previous fully connected layer, output = 2 raw prediction scores, called "logits"
        # One score represents the cat class, one score represents the dog class
        #
        # A Softmax layer is not required here because CrossEntropyLoss internally handles the necessary probability calculations.
        self.fc2 = nn.Linear(in_features=128, out_features=2)

    # Define how images move through the neural network from input to prediction
    def forward(self, x):
        """
        Performs the forward pass through the CNN.

        Args:
        x: A batch of image tensors with the shape: [batch_size, 3, 128, 128]

        Returns:
        x: A tensor containing two raw class prediction scores for each image, with the shape: [batch_size, 2]
        """

        # First convolutional block (convolution -> batch normalization -> ReLU activation -> pooling):
        # Input shape: [batch_size, 3, 128, 128]
        # After conv1: [batch_size, 32, 128, 128]
        # After pooling: [batch_size, 32, 64, 64]
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        # Second convolutional block (convolution -> batch normalization -> ReLU activation -> pooling):
        # Input shape: [batch_size, 32, 64, 64]
        # After conv2: [batch_size, 64, 64, 64]
        # After pooling: [batch_size, 64, 32, 32]
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Third convolutional block (convolution -> batch normalization -> ReLU activation -> pooling):
        # Input shape: [batch_size, 64, 32, 32]
        # After conv3: [batch_size, 128, 32, 32]
        # After pooling: [batch_size, 128, 16, 16]
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)

        # Fourth convolutional block (convolution -> batch normalization -> ReLU activation -> pooling):
        # Input shape: [batch_size, 128, 16, 16]
        # After conv4: [batch_size, 256, 16, 16]
        # After pooling: [batch_size, 256, 8, 8]
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool(x)

        # Global average pooling (e.g., calculate one average value for each of the 256 feature maps):
        # Shape changes from: [batch_size, 256, 8, 8] to [batch_size, 256, 1, 1]
        x = self.global_pool(x)

        # Flatten the pooled feature maps into one vector per image:
        # Shape changes from: [batch_size, 256, 1, 1] to [batch_size, 256]
        # torch.flatten(x, 1) preserves dimension 0, which is the batch size, and flattens every dimension after it.
        x = torch.flatten(x, start_dim=1)

        # Fully connected layer (learns combinations of the 256 summarized image features):
        # Shape changes from: [batch_size, 256] to [batch_size, 128]
        x = self.fc1(x)

        # Apply the ReLU activation function:
        # Converts negative values to zero while keeping positive values.
        # This introduces nonlinearity so the network can learn more complex
        # relationships than a sequence of purely linear transformations.
        x = F.relu(x)

        # Apply dropout (randomly disables 40% of the values during training to help reduce overfitting.):
        # The shape remains: [batch_size, 128]
        x = self.dropout(x)

        # Final output layer (returns two raw prediction scores for each image):
        # Shape changes from: [batch_size, 128] to [batch_size, 2]
        # The class with the larger score becomes the predicted class!
        x = self.fc2(x)

        return x

# Call the function to configure pandas display settings
configure_pd_display()

# Call the function to initialize our directory and store the returned variables
directory_path, train_path, test_path, folder_list = initialize_directory()

# ===================== EDA: Checking for Class Imbalances =====================
# Check how many images exist in each set of our data (class imbalances)
print("Train Cats: " + str(len(os.listdir(train_path + '\\cats'))))  # Check the training set
print("Train Dogs: " + str(len(os.listdir(train_path + '\\dogs'))))
print("Test Cats: " + str(len(os.listdir(test_path + '\\cats'))))    # Check the testing set
print("Test Dogs: " + str(len(os.listdir(test_path + '\\dogs'))))

# Perform the image quality check with the function and store the returned metadata
image_quality_check(train_path)

# Define the image transformations with the function for training and testing datasets
train_transform, evaluation_transform = transform_images()

# Load the datasets and split the original training data
train_dataset, validation_dataset, test_dataset, class_names, class_to_idx = load_and_split_data(
    train_path=train_path, test_path=test_path, train_transform=train_transform,
    evaluation_transform=evaluation_transform, validation_size=0.15, random_state=42)

# Set the number of images processed in each batch
batch_size = 32

# Create the training DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create the validation DataLoader
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

# Create the testing DataLoader
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Confirm that the DataLoaders were created
print('Training DataLoader created with batch size: ' + str(batch_size))
print('Validation DataLoader created with batch size: ' + str(batch_size))
print('Testing DataLoader created with batch size: ' + str(batch_size))

# Display the dataset information
print('Image classes: ' + str(class_names))
print('Class mapping: ' + str(class_to_idx))
print('Training images loaded: ' + str(len(train_dataset)))
print('Validation images loaded: ' + str(len(validation_dataset)))
print('Testing images loaded: ' + str(len(test_dataset)))

# Set the device to GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create the CNN model and send it to the device
model = ImprovedCNN().to(device)

# Confirm where the model is stored
print('Model created and sent to: ' + str(device))

# Display the model architecture
print(model)

# Define the loss function ('CrossEntropyLoss' is commonly used for classification - though, usually multi-class problems.)
criterion = nn.CrossEntropyLoss()

# Define the optimizer (Adam updates the model weights during training.)
learning_rate = 0.001
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)

print('Loss function defined: CrossEntropyLoss')
print('Optimizer defined: AdamW')
print('Learning rate: ' + str(learning_rate))
print('Weight decay: ' + str(0.0001))

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
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

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