# Improved Alzheimer's Disease Classification Model
# Fixing overfitting issues

# import system libs
import os
import time
import shutil
import pathlib
import itertools

# Set matplotlib to non-interactive mode upfront
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# import data handling tools
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm  # Import tqdm for progress bars

# import Deep learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

print('modules loaded')

# Custom callback implementation with progress bar
class MyCallback(keras.callbacks.Callback):
    def __init__(self, model, patience, stop_patience, threshold, factor, batches, epochs, ask_epoch):
        super(MyCallback, self).__init__()
        self.model = model
        self.patience = patience # specifies how many epochs without improvement before learning rate is adjusted
        self.stop_patience = stop_patience # specifies how many times to adjust lr without improvement to stop training
        self.threshold = threshold # specifies training accuracy threshold when lr will be adjusted based on validation loss
        self.factor = factor # factor by which to reduce the learning rate
        self.batches = batches # number of training batch to run per epoch
        self.epochs = epochs
        self.ask_epoch = ask_epoch
        self.ask_epoch_initial = ask_epoch # save this value to restore if restarting training

        # callback variables
        self.count = 0 # how many times lr has been reduced without improvement
        self.stop_count = 0
        self.best_epoch = 1   # epoch with the lowest loss
        self.initial_lr = float(tf.keras.backend.get_value(model.optimizer.lr)) # get the initial learning rate and save it
        self.highest_tracc = 0.0 # set highest training accuracy to 0 initially
        self.lowest_vloss = np.inf # set lowest validation loss to infinity initially
        self.best_weights = self.model.get_weights() # set best weights to model's initial weights
        self.initial_weights = self.model.get_weights()   # save initial weights if they have to get restored
        self.pbar = None  # Progress bar for batch processing

    # Define a function that will run when train begins
    def on_train_begin(self, logs= None):
        msg = 'Do you want model asks you to halt the training [y/n] ?'
        print(msg)
        ans = input('')
        if ans in ['Y', 'y']:
            self.ask_permission = 1
        elif ans in ['N', 'n']:
            self.ask_permission = 0

        print("\n" + "="*80)
        print(f"{'Epoch':^8s}{'Loss':^10s}{'Accuracy':^9s}{'V_loss':^9s}{'V_acc':^9s}{'LR':^9s}{'Next LR':^9s}{'Monitor':^10s}{'% Improv':10s}{'Duration':^8s}")
        print("="*80)
        self.start_time = time.time()

    def on_train_end(self, logs= None):
        stop_time = time.time()
        tr_duration = stop_time - self.start_time
        hours = tr_duration // 3600
        minutes = (tr_duration - (hours * 3600)) // 60
        seconds = tr_duration - ((hours * 3600) + (minutes * 60))

        msg = f'Training complete in {str(int(hours))} hours, {int(minutes)} minutes, {seconds:.2f} seconds'
        print("\n" + "="*len(msg))
        print(msg)
        print("="*len(msg))

        # set the weights of the model to the best weights
        self.model.set_weights(self.best_weights)
        
        # Close progress bar if it exists
        if self.pbar is not None:
            self.pbar.close()

    def on_epoch_begin(self, epoch, logs= None):
        self.ep_start = time.time()
        # Initialize progress bar for batches
        print(f"\nStarting Epoch {epoch+1}/{self.epochs}...")
        try:
            self.pbar = tqdm(total=self.batches, desc=f"Epoch {epoch+1}/{self.epochs}", 
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        except Exception as e:
            print(f"Warning: Could not initialize batch progress bar: {e}")
            self.pbar = None
            # Use simple counter instead
            self.batch_counter = 0
            self.last_log_time = time.time()

    def on_train_batch_end(self, batch, logs= None):
        # get batch accuracy and loss
        acc = logs.get('accuracy') * 100
        loss = logs.get('loss')
        
        # Update the progress bar
        if self.pbar is not None:
            try:
                self.pbar.update(1)
                self.pbar.set_postfix(loss=f"{loss:.4f}", acc=f"{acc:.2f}%")
            except Exception as e:
                # If tqdm fails, fallback to simple counter
                self.pbar = None
                print(f"Warning: Progress bar failed: {e}")
                self.batch_counter = batch
                self.last_log_time = time.time()
        else:
            # Simple fallback counter - show update every 5 batches
            self.batch_counter = batch
            current_time = time.time()
            if batch % 5 == 0 or batch == self.batches-1:
                print(f"  Batch {batch+1}/{self.batches} - loss: {loss:.4f}, accuracy: {acc:.2f}%", 
                      end='\r', flush=True)

    # Define method runs on the end of each epoch
    def on_epoch_end(self, epoch, logs= None):
        # Close batch progress bar
        if self.pbar is not None:
            try:
                self.pbar.close()
            except:
                pass
            self.pbar = None
        else:
            # Clear the batch progress line
            print("", end='\r')
        
        # Print a newline to separate from batch output
        print("")
        
        ep_end = time.time()
        duration = ep_end - self.ep_start

        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr)) # get the current learning rate
        current_lr = lr
        acc = logs.get('accuracy')  # get training accuracy
        v_acc = logs.get('val_accuracy')  # get validation accuracy
        loss = logs.get('loss')  # get training loss for this epoch
        v_loss = logs.get('val_loss')  # get the validation loss for this epoch

        if acc < self.threshold: # if training accuracy is below threshold adjust lr based on training accuracy
            monitor = 'accuracy'
            if epoch == 0:
                pimprov = 0.0
            else:
                pimprov = (acc - self.highest_tracc ) * 100 / self.highest_tracc # define improvement of model progres

            if acc > self.highest_tracc: # training accuracy improved in the epoch
                self.highest_tracc = acc # set new highest training accuracy
                self.best_weights = self.model.get_weights() # training accuracy improved so save the weights
                self.count = 0 # set count to 0 since training accuracy improved
                self.stop_count = 0 # set stop counter to 0
                if v_loss < self.lowest_vloss:
                    self.lowest_vloss = v_loss
                self.best_epoch = epoch + 1  # set the value of best epoch for this epoch

            else:
                # training accuracy did not improve check if this has happened for patience number of epochs
                # if so adjust learning rate
                if self.count >= self.patience - 1: # lr should be adjusted
                    lr = lr * self.factor # adjust the learning by factor
                    tf.keras.backend.set_value(self.model.optimizer.lr, lr) # set the learning rate in the optimizer
                    self.count = 0 # reset the count to 0
                    self.stop_count = self.stop_count + 1 # count the number of consecutive lr adjustments
                    self.count = 0 # reset counter
                    if v_loss < self.lowest_vloss:
                        self.lowest_vloss = v_loss
                else:
                    self.count = self.count + 1 # increment patience counter

        else: # training accuracy is above threshold so adjust learning rate based on validation loss
            monitor = 'val_loss'
            if epoch == 0:
                pimprov = 0.0

            else:
                pimprov = (self.lowest_vloss - v_loss ) * 100 / self.lowest_vloss

            if v_loss < self.lowest_vloss: # check if the validation loss improved
                self.lowest_vloss = v_loss # replace lowest validation loss with new validation loss
                self.best_weights = self.model.get_weights() # validation loss improved so save the weights
                self.count = 0 # reset count since validation loss improved
                self.stop_count = 0
                self.best_epoch = epoch + 1 # set the value of the best epoch to this epoch

            else: # validation loss did not improve
                if self.count >= self.patience - 1: # need to adjust lr
                    lr = lr * self.factor # adjust the learning rate
                    self.stop_count = self.stop_count + 1 # increment stop counter because lr was adjusted
                    self.count = 0 # reset counter
                    tf.keras.backend.set_value(self.model.optimizer.lr, lr) # set the learning rate in the optimizer

                else:
                    self.count = self.count + 1 # increment the patience counter

                if acc > self.highest_tracc:
                    self.highest_tracc = acc

        # Print epoch results
        print("-" * 80)
        print(f"EPOCH {epoch+1}/{self.epochs} SUMMARY:")
        print(f"Training:   Loss: {loss:.4f} - Accuracy: {acc*100:.2f}%")
        print(f"Validation: Loss: {v_loss:.4f} - Accuracy: {v_acc*100:.2f}%")
        print(f"Learning Rate: Current: {current_lr:.6f} - Next: {lr:.6f}")
        print(f"Monitor: {monitor} - Improvement: {pimprov:.2f}% - Duration: {duration:.2f}s")
        print("-" * 80)

        if self.stop_count > self.stop_patience - 1: # check if learning rate has been adjusted stop_count times with no improvement
            msg = f' training has been halted at epoch {epoch + 1} after {self.stop_patience} adjustments of learning rate with no improvement'
            print(msg)
            self.model.stop_training = True # stop training

        else:
            if self.ask_epoch != None and self.ask_permission != 0:
                if epoch + 1 >= self.ask_epoch:
                    msg = 'enter H to halt training or an integer for number of epochs to run then ask again'
                    print(msg)

                    ans = input('')
                    if ans == 'H' or ans == 'h':
                        msg = f'training has been halted at epoch {epoch + 1} due to user input'
                        print(msg)
                        self.model.stop_training = True # stop training

                    else:
                        try:
                            ans = int(ans)
                            self.ask_epoch += ans
                            msg = f' training will continue until epoch {str(self.ask_epoch)}'
                            print(msg)
                            print("\n" + "="*80)
                            print("CONTINUING TRAINING")
                            print("="*80)

                        except Exception:
                            print('Invalid input. Training will continue.')

# Custom training progress callback
class TqdmProgressCallback(keras.callbacks.Callback):
    def __init__(self, epochs, verbose=1):
        super(TqdmProgressCallback, self).__init__()
        self.epochs = epochs
        self.verbose = verbose
        self.epoch_pbar = None
        
    def on_train_begin(self, logs=None):
        print("\n" + "*"*80)
        print("TRAINING PROGRESS".center(80))
        print("*"*80)
        print("Starting training process. This may take some time...")
        print("If progress bars don't display properly, you'll still see epoch updates below.")
        try:
            self.epoch_pbar = tqdm(total=self.epochs, desc="Total Progress", 
                                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        except Exception as e:
            print(f"Warning: Could not initialize progress bar: {e}")
            self.epoch_pbar = None
    
    def on_epoch_end(self, epoch, logs=None):
        # Always print epoch status even if progress bar fails
        val_loss = logs.get('val_loss', 0)
        val_acc = logs.get('val_accuracy', 0) * 100
        tr_loss = logs.get('loss', 0)
        tr_acc = logs.get('accuracy', 0) * 100
        
        print(f"Epoch {epoch+1}/{self.epochs} completed - "
              f"loss: {tr_loss:.4f}, acc: {tr_acc:.2f}%, "
              f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.2f}%")
        
        if self.epoch_pbar is not None:
            try:
                self.epoch_pbar.update(1)
                if self.verbose > 0:
                    self.epoch_pbar.set_postfix(val_loss=f"{val_loss:.4f}", val_acc=f"{val_acc:.2f}%")
            except Exception as e:
                print(f"Warning: Progress bar update failed: {e}")
    
    def on_train_end(self, logs=None):
        if self.epoch_pbar is not None:
            try:
                self.epoch_pbar.close()
            except:
                pass
        print("\n" + "*"*80)
        print("TRAINING COMPLETE".center(80))
        print("*"*80)

# Generate data paths with labels
def define_paths(data_dir):
    filepaths = []
    labels = []

    folds = os.listdir(data_dir)
    for fold in folds:
        foldpath = os.path.join(data_dir, fold)
        filelist = os.listdir(foldpath)
        for file in filelist:
            fpath = os.path.join(foldpath, file)
            filepaths.append(fpath)
            labels.append(fold)

    return filepaths, labels

# Concatenate data paths with labels into one dataframe
def define_df(files, classes):
    Fseries = pd.Series(files, name= 'filepaths')
    Lseries = pd.Series(classes, name='labels')
    return pd.concat([Fseries, Lseries], axis= 1)

# Split dataframe to train, valid, and test
def split_data(data_dir):
    # train dataframe
    files, classes = define_paths(data_dir)
    df = define_df(files, classes)
    strat = df['labels']
    train_df, dummy_df = train_test_split(df, train_size=0.7, shuffle=True, random_state=42, stratify=strat)  # MODIFIED: reduced train size

    # valid and test dataframe
    strat = dummy_df['labels']
    valid_df, test_df = train_test_split(dummy_df, train_size=0.5, shuffle=True, random_state=42, stratify=strat)

    return train_df, valid_df, test_df

def create_gens(train_df, valid_df, test_df, batch_size):
    '''
    Enhanced image data generator with more aggressive data augmentation
    to help combat overfitting
    '''
    img_size = (224, 224)
    channels = 3
    color = 'rgb'
    img_shape = (img_size[0], img_size[1], channels)

    # Determine test batch size
    ts_length = len(test_df)
    test_batch_size = max(sorted([ts_length // n for n in range(1, ts_length + 1) if ts_length%n == 0 and ts_length/n <= 80]))
    test_steps = ts_length // test_batch_size

    # Enhanced preprocessing function that normalizes pixel values
    def preprocess_input(img):
        img = img / 255.0  # Normalize to [0,1]
        return img

    # More aggressive data augmentation for training
    tr_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,           # Rotate images by up to 20 degrees
        width_shift_range=0.2,       # Shift horizontally by up to 20%
        height_shift_range=0.2,      # Shift vertically by up to 20%
        shear_range=0.15,            # Apply shearing transformations
        zoom_range=0.15,             # Zoom in or out by up to 15%
        horizontal_flip=True,        # Flip images horizontally
        fill_mode='nearest',         # Fill in newly created pixels
        brightness_range=[0.8, 1.2]  # Adjust brightness
    )
    
    # Only preprocessing for validation and test data
    ts_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = tr_gen.flow_from_dataframe(
        train_df, 
        x_col='filepaths', 
        y_col='labels', 
        target_size=img_size, 
        class_mode='categorical',
        color_mode=color, 
        shuffle=True, 
        batch_size=batch_size
    )

    valid_gen = ts_gen.flow_from_dataframe(
        valid_df, 
        x_col='filepaths', 
        y_col='labels', 
        target_size=img_size, 
        class_mode='categorical',
        color_mode=color, 
        shuffle=True, 
        batch_size=batch_size
    )

    test_gen = ts_gen.flow_from_dataframe(
        test_df, 
        x_col='filepaths', 
        y_col='labels', 
        target_size=img_size, 
        class_mode='categorical',
        color_mode=color, 
        shuffle=False, 
        batch_size=test_batch_size
    )

    return train_gen, valid_gen, test_gen

def show_images(gen, save_to_file=True):
    '''
    Show sample images from a generator or save them to a file
    '''
    g_dict = gen.class_indices        # defines dictionary {'class': index}
    classes = list(g_dict.keys())     # defines list of dictionary's keys (classes)
    images, labels = next(gen)        # get a batch of samples from the generator

    # calculate number of displayed samples
    length = len(labels)              # length of batch
    sample = min(length, 25)          # limit to max 25 images

    plt.figure(figsize=(20, 20))

    for i in range(sample):
        plt.subplot(5, 5, i + 1)
        image = images[i]             # Data is already normalized in generator
        plt.imshow(image)
        index = np.argmax(labels[i])  # get image class index
        class_name = classes[index]   # get class name
        plt.title(class_name, color='blue', fontsize=12)
        plt.axis('off')
    
    if save_to_file:
        plt.savefig('sample_images.png')
        plt.close()
        print(f"Sample images saved to 'sample_images.png'")
    else:
        plt.show()
    
    return classes

def plot_training(hist, save_to_file=True):
    '''
    Plot training history showing accuracy and loss curves
    or save them to a file
    '''
    # Define needed variables
    tr_acc = hist.history['accuracy']
    tr_loss = hist.history['loss']
    val_acc = hist.history['val_accuracy']
    val_loss = hist.history['val_loss']
    index_loss = np.argmin(val_loss)
    val_lowest = val_loss[index_loss]
    index_acc = np.argmax(val_acc)
    acc_highest = val_acc[index_acc]
    Epochs = [i+1 for i in range(len(tr_acc))]
    loss_label = f'best epoch= {str(index_loss + 1)}'
    acc_label = f'best epoch= {str(index_acc + 1)}'

    # Plot training history
    plt.figure(figsize= (20, 8))
    plt.style.use('fivethirtyeight')

    plt.subplot(1, 2, 1)
    plt.plot(Epochs, tr_loss, 'r', label= 'Training loss')
    plt.plot(Epochs, val_loss, 'g', label= 'Validation loss')
    plt.scatter(index_loss + 1, val_lowest, s= 150, c= 'blue', label= loss_label)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(Epochs, tr_acc, 'r', label= 'Training Accuracy')
    plt.plot(Epochs, val_acc, 'g', label= 'Validation Accuracy')
    plt.scatter(index_acc + 1 , acc_highest, s= 150, c= 'blue', label= acc_label)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    
    if save_to_file:
        plt.savefig('training_history.png')
        plt.close()
        print(f"Training history plot saved to 'training_history.png'")
    else:
        plt.show()

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues, save_to_file=True):
    '''
    Plot confusion matrix or save it to a file
    '''
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized Confusion Matrix')
    else:
        print('Confusion Matrix, Without Normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_to_file:
        plt.savefig('confusion_matrix.png')
        plt.close()
        print(f"Confusion matrix saved to 'confusion_matrix.png'")
    else:
        plt.show()


# # Install Kaggle and configure
# !pip install -q kaggle
# from google.colab import files
# # Upload the kaggle.json file
# uploaded = files.upload()
# !mkdir -p ~/.kaggle
# !cp kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json

# # Download Alzheimer's dataset
# !kaggle datasets download -d tourist55/alzheimers-dataset-4-class-of-images
# !unzip -q /content/alzheimers-dataset-4-class-of-images.zip

# Define the data directory - CHANGED to use a local Windows path
# Replace this with the actual path to your dataset
data_dir = os.path.join(os.getcwd(), "Alzheimer_s Dataset", "train")

# Check if the directory exists and contains class folders
required_classes = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
missing_classes = []

if not os.path.exists(data_dir):
    print(f"Error: Dataset directory not found at {data_dir}")
    missing_classes = required_classes
else:
    # Check if the required class folders exist and contain images
    for class_name in required_classes:
        class_path = os.path.join(data_dir, class_name)
        if not os.path.exists(class_path):
            missing_classes.append(class_name)
        else:
            # Check if directory is empty
            if len(os.listdir(class_path)) == 0:
                missing_classes.append(class_name)

if missing_classes:
    print("\nMissing or empty class folders:")
    for missing in missing_classes:
        print(f"- {missing}")
    
    print("\nDataset Instructions:")
    print("1. Download the Alzheimer's dataset from Kaggle:")
    print("   https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images")
    print("2. Extract the zip file")
    print("3. Place the dataset in the following structure:")
    print(f"   {os.path.join(os.getcwd(), 'Alzheimer_s Dataset', 'train')}")
    print("   with subfolders for each class (MildDemented, ModerateDemented, NonDemented, VeryMildDemented)")
    
    # Ask user for dataset path
    print("\nWould you like to specify a different dataset location? (y/n)")
    user_input = input()
    if user_input.lower() == 'y':
        print("Please enter the full path to the dataset folder containing the class folders:")
        user_path = input()
        if not os.path.exists(user_path):
            print(f"Error: Path {user_path} does not exist. Exiting.")
            exit(1)
        data_dir = user_path
        # Check if the new path contains the required folders
        missing_classes = []
        for class_name in required_classes:
            class_path = os.path.join(data_dir, class_name)
            if not os.path.exists(class_path) or len(os.listdir(class_path)) == 0:
                missing_classes.append(class_name)
        
        if missing_classes:
            print("\nThe specified directory is still missing required class folders:")
            for missing in missing_classes:
                print(f"- {missing}")
            print("Exiting program. Please set up the dataset correctly.")
            exit(1)
    else:
        print("Exiting program. Please set up the dataset first.")
        exit(1)

try:
    # Get splitted data with adjusted train/val/test ratio
    train_df, valid_df, test_df = split_data(data_dir)
    
    # Display the distribution of classes in each split
    print("Training data distribution:")
    print(train_df['labels'].value_counts())
    print("\nValidation data distribution:")
    print(valid_df['labels'].value_counts())
    print("\nTest data distribution:")
    print(test_df['labels'].value_counts())
    
    # Get Generators with enhanced data augmentation
    batch_size = 32  # Reduced batch size
    print("\nInitializing data generators with augmentation...")
    train_gen, valid_gen, test_gen = create_gens(train_df, valid_df, test_df, batch_size)
    print("Data generators created successfully!")
    
    # Continue with the rest of the program
    # Show sample images from the training generator
    print("\nPreparing to show sample images from training data...")
    try:
        # Try to save images to a file instead of displaying them
        try:
            import matplotlib
            # Use a non-interactive backend that doesn't require a display
            matplotlib.use('Agg') 
            
            # Get sample images from the generator
            g_dict = train_gen.class_indices        # defines dictionary {'class': index}
            classes = list(g_dict.keys())     # defines list of dictionary's keys (classes)
            images, labels = next(train_gen)        # get a batch of samples from the generator
            
            # Calculate number of displayed samples
            length = len(labels)              # length of batch
            sample = min(length, 16)          # limit to max 16 images
            
            plt.figure(figsize=(12, 12))
            
            for i in range(sample):
                plt.subplot(4, 4, i + 1)
                image = images[i]             # Data is already normalized in generator
                plt.imshow(image)
                index = np.argmax(labels[i])  # get image class index
                class_name = classes[index]   # get class name
                plt.title(class_name, color='blue', fontsize=10)
                plt.axis('off')
            
            # Save the figure instead of displaying it
            plt.tight_layout()
            plt.savefig('sample_training_images.png')
            plt.close()
            print("Sample images saved to 'sample_training_images.png'")
            
        except Exception as e:
            print(f"Could not save sample images: {e}")
            print("Continuing with training...")
        
    except Exception as e:
        print(f"Error in image display: {e}")
        print("Continuing with training process...")
    
    # Print class information instead of showing images
    print("\nClass information:")
    for idx, class_name in enumerate(train_gen.class_indices):
        print(f"  Class {idx}: {class_name}")
    
    # Create Model Structure with modifications to reduce overfitting
    print("\nBuilding model architecture...")
    img_size = (224, 224)
    channels = 3
    img_shape = (img_size[0], img_size[1], channels)
    class_count = len(list(train_gen.class_indices.keys()))
    print(f"Number of classes: {class_count}")
    
    # Create a smaller, more regularized model
    # Option 1: Modified EfficientNetB0 (smaller than B3)
    print("\nLoading EfficientNetB0 base model from TensorFlow...")
    try:
        base_model = tf.keras.applications.efficientnet.EfficientNetB0(
            include_top=False, 
            weights="imagenet",
            input_shape=img_shape,
            pooling='max'
        )
        print("Base model loaded successfully!")
    except Exception as e:
        print(f"Error loading base model: {e}")
        raise

    # Freeze the base model to prevent overfitting
    print("Freezing base model layers...")
    for layer in base_model.layers:
        layer.trainable = False
    print("Base model layers frozen.")

    # Build the model with stronger regularization
    print("\nBuilding full model architecture with regularization...")
    model = Sequential([
        base_model,
        BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
        # More dropout and stronger regularization
        Dropout(rate=0.5, seed=42),  # Increased dropout
        Dense(128, kernel_regularizer=regularizers.l2(l=0.02),  # Smaller dense layer with stronger regularization
              activity_regularizer=regularizers.l1(0.01),
              bias_regularizer=regularizers.l1(0.01),
              activation='relu'),
        Dropout(rate=0.5, seed=123),  # Additional dropout
        Dense(class_count, activation='softmax')
    ])
    print("Model architecture built successfully!")

    # Use Adam optimizer with reduced learning rate
    print("\nCompiling model with Adam optimizer...")
    model.compile(
        Adam(learning_rate=0.0005),  # Lower learning rate
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print("Model compilation complete!")

    print("\nModel summary:")
    model.summary()

    # Training parameters with early stopping
    print("\nConfiguring training parameters...")
    batch_size = 32
    epochs = 50  # Increase max epochs but expect early stopping
    patience = 3  # Increased patience
    stop_patience = 5  # Increased stop patience
    threshold = 0.85  # Reduced accuracy threshold
    factor = 0.5
    ask_epoch = 5
    batches = int(np.ceil(len(train_gen.labels) / batch_size))
    print(f"Training batches per epoch: {batches}")

    # Add standard callbacks for early stopping and learning rate reduction
    print("\nSetting up training callbacks...")
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=0.00001,
        verbose=1
    )

    # Custom callback for the training process
    custom_callback = MyCallback(
        model=model,
        patience=patience,
        stop_patience=stop_patience,
        threshold=threshold,
        factor=factor,
        batches=batches,
        epochs=epochs,
        ask_epoch=ask_epoch
    )

    callbacks = [custom_callback, early_stopping, reduce_lr]
    print("Callbacks configured successfully!")

    # Train the model with progress bar
    print("\n" + "="*80)
    print("STARTING MODEL TRAINING")
    print("="*80)
    print("\nPreparing to start training process...")
    
    history = model.fit(
        x=train_gen,
        epochs=epochs,
        verbose=0,  # Set to 0 as we're using custom progress bars
        callbacks=callbacks + [TqdmProgressCallback(epochs=epochs)],  # Add our progress callback
        validation_data=valid_gen,
        validation_steps=None,
        shuffle=False
    )

    # Plot training history
    plot_training(history, save_to_file=True)

    # Fine-tune the model by unfreezing some layers
    print("Fine-tuning the model by unfreezing the top layers of the base model...")

    # Unfreeze the top layers of the base model
    for layer in base_model.layers[-20:]:  # Unfreeze the last 20 layers
        layer.trainable = True

    # Recompile with a lower learning rate for fine-tuning
    model.compile(
        Adam(learning_rate=0.00001),  # Very low learning rate for fine-tuning
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Update the callbacks for fine-tuning
    fine_tune_callback = MyCallback(
        model=model,
        patience=patience,
        stop_patience=stop_patience,
        threshold=threshold,
        factor=factor,
        batches=batches,
        epochs=20,  # Fewer epochs for fine-tuning
        ask_epoch=5
    )

    fine_tune_callbacks = [fine_tune_callback, early_stopping, reduce_lr]

    # Fine-tune the model with progress bar
    fine_tune_history = model.fit(
        x=train_gen,
        epochs=20,  # Fewer epochs for fine-tuning
        verbose=0,  # Set to 0 as we're using custom progress bars
        callbacks=fine_tune_callbacks + [TqdmProgressCallback(epochs=20)],  # Add our progress callback
        validation_data=valid_gen,
        validation_steps=None,
        shuffle=False
    )

    # Plot fine-tuning history
    plot_training(fine_tune_history, save_to_file=True)

    # Evaluate the model on all datasets
    ts_length = len(test_df)
    test_batch_size = max(sorted([ts_length // n for n in range(1, ts_length + 1) if ts_length%n == 0 and ts_length/n <= 80]))
    test_steps = ts_length // test_batch_size

    print("Evaluating the model...")
    train_score = model.evaluate(train_gen, steps=test_steps, verbose=1)
    valid_score = model.evaluate(valid_gen, steps=test_steps, verbose=1)
    test_score = model.evaluate(test_gen, steps=test_steps, verbose=1)

    print("\nModel Evaluation Results:")
    print("-" * 40)
    print("Train Loss: {:.4f}".format(train_score[0]))
    print("Train Accuracy: {:.4f}".format(train_score[1]))
    print('-' * 40)
    print("Validation Loss: {:.4f}".format(valid_score[0]))
    print("Validation Accuracy: {:.4f}".format(valid_score[1]))
    print('-' * 40)
    print("Test Loss: {:.4f}".format(test_score[0]))
    print("Test Accuracy: {:.4f}".format(test_score[1]))

    # Make predictions on the test set
    print("\nGenerating predictions for the test set...")
    preds = model.predict(test_gen)
    y_pred = np.argmax(preds, axis=1)

    # Get class names
    g_dict = test_gen.class_indices
    classes = list(g_dict.keys())

    # Confusion matrix
    cm = confusion_matrix(test_gen.classes, y_pred)
    plot_confusion_matrix(cm=cm, classes=classes, title='Confusion Matrix', save_to_file=True)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(test_gen.classes, y_pred, target_names=classes))

    # Save the model
    model.save("alzheimers_model_improved.h5")
    print("Model saved as 'alzheimers_model_improved.h5'")

    # Additional analysis - visualize feature maps
    def visualize_feature_maps(model, img_path, layer_name, save_to_file=True):
        """Visualize feature maps from a specific layer for a given image"""
        # Load and preprocess the image
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize to [0,1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Create a model that outputs the feature maps from the specified layer
        layer_outputs = [layer.output for layer in model.layers if layer.name == layer_name]
        if not layer_outputs:
            print(f"Layer '{layer_name}' not found in model")
            return
    
        activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
        
        # Get activations
        activations = activation_model.predict(img_array)
        
        # Plot the feature maps
        plt.figure(figsize=(15, 8))
        plt.suptitle(f"Feature Maps from Layer: {layer_name}", fontsize=16)
        
        # Plot the original image
        plt.subplot(2, 8, 1)
        plt.imshow(img)
        plt.title("Original Image")
        plt.axis('off')
        
        # Plot the feature maps
        feature_maps = activations[0]
        if len(feature_maps.shape) == 4:  # For convolutional layers
            features = feature_maps[0]
            for i in range(min(15, features.shape[-1])):  # Show up to 15 channels
                plt.subplot(2, 8, i + 2)
                plt.imshow(features[:, :, i], cmap='viridis')
                plt.title(f"Channel {i}")
                plt.axis('off')
        else:  # For dense layers
            plt.subplot(2, 8, 2)
            plt.bar(range(min(15, feature_maps.shape[1])), feature_maps[0, :15])
            plt.title("Neuron Activations")
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        if save_to_file:
            plt.savefig(f'feature_maps_{layer_name}.png')
            plt.close()
            print(f"Feature maps saved to 'feature_maps_{layer_name}.png'")
        else:
            plt.show()

    # Sample a few images from the test set to visualize feature maps
    sample_images = test_df['filepaths'].sample(3).tolist()
    for img_path in sample_images:
        # Get actual class from filename
        class_name = img_path.split('/')[-2]
        print(f"\nVisualizing feature maps for image from class: {class_name}")
        
        # Visualize feature maps from an intermediate layer
        visualize_feature_maps(model, img_path, 'efficientnetb0', save_to_file=True)

    # Sample prediction function to test on individual images
    def predict_single_image(model, img_path, img_size=(224, 224), save_to_file=True):
        """Make a prediction for a single image"""
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize to [0,1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Make prediction
        prediction = model.predict(img_array)
        pred_class = np.argmax(prediction, axis=1)[0]
        
        # Get class names
        g_dict = test_gen.class_indices
        classes = list(g_dict.keys())
        
        # Get actual class from filepath
        actual_class = img_path.split('/')[-2]
        
        # Display results
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        
        predicted_class = classes[pred_class]
        confidence = prediction[0][pred_class] * 100
        
        title_color = 'green' if predicted_class == actual_class else 'red'
        plt.title(f"Actual: {actual_class}\nPredicted: {predicted_class} ({confidence:.2f}%)", 
                  color=title_color, fontsize=14)
        plt.axis('off')
        
        if save_to_file:
            plt.savefig(f'single_image_prediction_{actual_class}_{predicted_class}.png')
            plt.close()
            print(f"Prediction saved to 'single_image_prediction_{actual_class}_{predicted_class}.png'")
        else:
            plt.show()
        
        # Print prediction details
        print(f"Image: {img_path}")
        print(f"Actual class: {actual_class}")
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.2f}%")
        print("Class probabilities:")
        for i, (cls, prob) in enumerate(zip(classes, prediction[0])):
            print(f"  {cls}: {prob*100:.2f}%")
        
        return predicted_class, confidence

    # Test the function on a few random images
    print("\nTesting model on individual images:")
    sample_test_images = test_df['filepaths'].sample(5).tolist()
    for img_path in sample_test_images:
        predict_single_image(model, img_path, save_to_file=True)

    # Analysis of model performance across different classes
    def analyze_class_performance(y_true, y_pred, classes, save_to_file=True):
        """Analyze model performance for each class"""
        report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
        
        # Extract metrics for each class
        metrics = []
        for cls in classes:
            metrics.append({
                'Class': cls,
                'Precision': report[cls]['precision'],
                'Recall': report[cls]['recall'],
                'F1-Score': report[cls]['f1-score'],
                'Support': report[cls]['support']
            })
        
        # Convert to DataFrame for better visualization
        df = pd.DataFrame(metrics)
        
        # Plot metrics
        plt.figure(figsize=(12, 8))
        
        # Precision, Recall, F1-Score
        plt.subplot(1, 2, 1)
        df_plot = df.set_index('Class')
        df_plot[['Precision', 'Recall', 'F1-Score']].plot(kind='bar', ax=plt.gca())
        plt.title('Performance Metrics by Class')
        plt.ylim(0, 1.0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Support (class distribution)
        plt.subplot(1, 2, 2)
        plt.pie(df['Support'], labels=df['Class'], autopct='%1.1f%%')
        plt.title('Test Data Distribution')
        
        plt.tight_layout()
        
        if save_to_file:
            plt.savefig('class_performance.png')
            plt.close()
            print(f"Class performance analysis saved to 'class_performance.png'")
        else:
            plt.show()
        
        return df

    print("\nAnalyzing performance across different classes:")
    performance_df = analyze_class_performance(test_gen.classes, y_pred, classes, save_to_file=True)
    print(performance_df)

    # Identify misclassified images for error analysis
    def identify_misclassifications(test_gen, y_pred, test_df, model, classes, num_samples=5, save_to_file=True):
        """Identify and display misclassified images"""
        test_filepaths = test_df['filepaths'].values
        y_true = test_gen.classes
        
        # Find indices of misclassified images
        misclassified_indices = np.where(y_true != y_pred)[0]
        
        if len(misclassified_indices) == 0:
            print("No misclassifications found!")
            return
        
        # Select a sample of misclassified images
        sample_indices = np.random.choice(
            misclassified_indices, 
            size=min(num_samples, len(misclassified_indices)), 
            replace=False
        )
        
        # Display the misclassified images
        plt.figure(figsize=(15, 12))
        for i, idx in enumerate(sample_indices):
            # Get the image
            img_path = test_filepaths[idx]
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            
            # Get true and predicted classes
            true_class = classes[y_true[idx]]
            pred_class = classes[y_pred[idx]]
            
            # Get prediction probabilities
            img_batch = np.expand_dims(img_array, axis=0)
            probs = model.predict(img_batch)[0]
            pred_prob = probs[y_pred[idx]] * 100
            true_prob = probs[y_true[idx]] * 100
            
            # Display image with information
            plt.subplot(3, 2, i+1)
            plt.imshow(img)
            plt.title(f"True: {true_class} ({true_prob:.1f}%)\nPred: {pred_class} ({pred_prob:.1f}%)", 
                      color='red', fontsize=12)
            plt.axis('off')
        
        plt.tight_layout()
        plt.suptitle("Misclassified Images Analysis", fontsize=16, y=0.99)
        plt.subplots_adjust(top=0.9)
        
        if save_to_file:
            plt.savefig('misclassified_images.png')
            plt.close()
            print(f"Misclassified images analysis saved to 'misclassified_images.png'")
        else:
            plt.show()

    # Analyze misclassified images
    print("\nAnalyzing misclassified images:")
    identify_misclassifications(test_gen, y_pred, test_df, model, classes, num_samples=6, save_to_file=True)

    # Model interpretability using Grad-CAM
    def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
        """Generate Grad-CAM heatmap for visualization"""
        # First, create a model that maps the input image to the activations
        # of the last conv layer and the output predictions
        grad_model = tf.keras.models.Model(
            [model.inputs], 
            [model.get_layer(last_conv_layer_name).output, model.output]
        )

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # This is the gradient of the output neuron with respect to
        # the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we normalize the heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

    def display_gradcam(img_path, heatmap, alpha=0.4, save_to_file=True):
        """Display Grad-CAM heatmap overlaid on the original image"""
        # Load the original image
        img = tf.keras.preprocessing.image.load_img(img_path)
        img = tf.keras.preprocessing.image.img_to_array(img)
        
        # Resize heatmap to match the original image size
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        # Use jet colormap to colorize the heatmap
        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        
        # Create an image with RGB colorized heatmap
        jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
        
        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * alpha + img
        superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
        
        # Display the original image and the heatmap
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(img / 255.0)  # Convert to 0-1 range for display
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(heatmap, cmap='jet')
        plt.title("Grad-CAM Heatmap")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(superimposed_img)
        plt.title("Grad-CAM Overlay")
        plt.axis('off')
        
        plt.tight_layout()
        
        if save_to_file:
            # Extract class name from path if possible
            try:
                class_name = img_path.split('/')[-2]
            except:
                class_name = "unknown"
            plt.savefig(f'gradcam_{class_name}.png')
            plt.close()
            print(f"Grad-CAM visualization saved to 'gradcam_{class_name}.png'")
        else:
            plt.show()

    # Apply Grad-CAM visualization to sample images
    print("\nGenerating Grad-CAM visualizations for model interpretability:")
    for img_class in classes:
        # Get a sample image from each class
        class_samples = test_df[test_df['labels'] == img_class]['filepaths'].values
        if len(class_samples) > 0:
            sample_img_path = class_samples[0]
            
            # Preprocess the image
            img = tf.keras.preprocessing.image.load_img(sample_img_path, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Generate predictions
            preds = model.predict(img_array)
            pred_class = np.argmax(preds[0])
            
            # Find the last convolutional layer
            # For EfficientNetB0, we need to find the last convolutional layer
            last_conv_layer = None
            for layer in reversed(model.layers[0].layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv_layer = layer.name
                    break
            
            if last_conv_layer:
                # Generate heatmap
                heatmap = make_gradcam_heatmap(
                    img_array, 
                    model.layers[0],  # Base model is the first layer in Sequential
                    last_conv_layer
                )
                
                # Display the heatmap
                print(f"\nGrad-CAM for class: {img_class}")
                print(f"Predicted as: {classes[pred_class]} with confidence {preds[0][pred_class]*100:.2f}%")
                display_gradcam(sample_img_path, heatmap, save_to_file=True)
            else:
                print(f"Could not find convolutional layer for Grad-CAM visualization")

    # Recommendations for further improvement
    print("""
    # Recommendations for Further Improvement of Alzheimer's Classification Model

    1. **Collect More Diverse Data**:
       - More data generally helps deep learning models generalize better
       - Focus on collecting samples that are currently misclassified

    2. **Try Different Model Architectures**:
       - Test other CNN architectures like ResNet, Inception, or VGG
       - Consider using ensembles of multiple models

    3. **Advanced Data Augmentation**:
       - Implement more sophisticated augmentation techniques specific to medical imaging
       - Consider using GANs to generate synthetic training examples

    4. **Model Explainability**:
       - Further develop model interpretability with techniques like LIME or SHAP
       - This helps build trust in the model's predictions

    5. **Domain Expert Validation**:
       - Have medical experts review misclassified images
       - Incorporate their feedback into model refinement

    6. **Hyperparameter Optimization**:
       - Perform systematic hyperparameter tuning using techniques like Bayesian optimization
       - Optimize regularization parameters to find the right balance

    7. **Consider Transfer Learning from Other Medical Domains**:
       - Pre-train on larger related medical imaging datasets before fine-tuning

    8. **Class Imbalance Techniques**:
       - Apply oversampling/undersampling if classes are imbalanced
       - Use weighted loss functions to give more importance to underrepresented classes
    """)

    # Save the final model with metadata
    model.save("alzheimers_model_final.h5")
    print("\nFinal model saved as 'alzheimers_model_final.h5'")

    # Export the model summary to a text file
    with open('model_summary.txt', 'w') as f:
        # Redirect stdout to the file
        import sys
        stdout_original = sys.stdout
        sys.stdout = f
        model.summary()
        sys.stdout = stdout_original

    print("Model summary saved to 'model_summary.txt'")

    # Save the training history
    history_dict = {
        'train_loss': history.history['loss'] + fine_tune_history.history['loss'],
        'train_accuracy': history.history['accuracy'] + fine_tune_history.history['accuracy'],
        'val_loss': history.history['val_loss'] + fine_tune_history.history['val_loss'],
        'val_accuracy': history.history['val_accuracy'] + fine_tune_history.history['val_accuracy']
    }

    np.save('training_history.npy', history_dict)
    print("Training history saved to 'training_history.npy'")

    print("\nThank you for using this Alzheimer's classification model!")

except Exception as e:
    print(f'Error: {e}')
    print("Program execution stopped due to above error.")
    exit(1)
