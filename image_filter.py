"""
Mark Lundine
Good/bad coastal image classification
adapted from https://keras.io/examples/vision/image_classification_from_scratch/
"""
import os
import numpy as np
import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import sys

def get_script_path():
    return os.path.dirname(os.path.abspath(__file__))

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def scheduler(epoch, lr):
    return lr * np.exp(-0.1) 

def load_dataset_gray(training_data_directory,
                      image_size,
                      batch_size):
    """
    loads in training data in training and validation sets
    inputs:
    training_data_directory (str): path to the training data
    returns:
    train_ds, val_ds
    """
    train_ds, val_ds = tf.keras.preprocessing.image_dataset_from_directory(training_data_directory,
                                                                           validation_split=0.5,
                                                                           color_mode="grayscale",
                                                                           subset="both",
                                                                           seed=1337,
                                                                           image_size=image_size,
                                                                           batch_size=batch_size
                                                                           )
    return train_ds, val_ds

def load_dataset_rgb(training_data_directory,
                     image_size,
                     batch_size):
    """
    loads in training data in training and validation sets
    inputs:
    training_data_directory (str): path to the training data
    returns:
    train_ds, val_ds
    """
    train_ds, val_ds = tf.keras.preprocessing.image_dataset_from_directory(training_data_directory,
                                                                           validation_split=0.5,
                                                                           color_mode="rgb",
                                                                           subset="both",
                                                                           seed=1337,
                                                                           image_size=image_size,
                                                                           batch_size=batch_size
                                                                           )
    return train_ds, val_ds

def data_augmentation(images):
    """
    applies data augmentation to images
    """
    data_augmentation_layers = [
        layers.RandomFlip("horizontal"),
        layers.RandomFlip("vertical"),
        layers.RandomRotation(0.25),
        layers.RandomZoom(0.5, 0.5),
        
    ]
    
    for layer in data_augmentation_layers:
        images = layer(images)
        
    return images

def define_model(input_shape, mode, num_classes=2):
    """
    Defines the classification model
    inputs:
    input_shape (tuple (xdim, ydim)): shape of images for model
    num_classes (int, optional): number of classes for the model
    """
    inputs = keras.Input(shape=input_shape)

    # Entry block
    if mode == 'train':
        x = data_augmentation(inputs)
    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        units = 1
    else:
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=None)(x)

    return keras.Model(inputs, outputs)

def train_model_rgb(model,
                    image_size,
                    train_ds,
                    val_ds,
                    model_folder,
                    epochs=100):
    """
    Trains the good/bad classification model
    inputs:
    model (keras model): this is defined in define_model()
    image_size (tuple, (xdim, ydim): dimensions of images for model
    train_ds: training dataset obtained from load_dataset
    val_ds: validation dataset obtained from load_dataset
    model_folder (str): path to save the model to
    epochs (int, optional): number of epochs to train for
    """
    model = define_model(input_shape=image_size + (3,), mode='train', num_classes=2)

    ##this makes a plot of the model, you need pydot installed, 
    keras.utils.plot_model(model, to_file=os.path.join(model_folder, 'model_graph.png'), show_shapes=True)
    
    ckpt_file = os.path.join(model_folder, "model_{epoch}.h5")
    epochs = 25
    lr_callback = keras.callbacks.LearningRateScheduler(scheduler)
    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='auto', restore_best_weights=True)
    callbacks = [keras.callbacks.ModelCheckpoint(ckpt_file),
                 early_stopping_callback,
                 lr_callback
                 ]
    model.compile(optimizer=keras.optimizers.Adam(lr=1e-6),
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[keras.metrics.BinaryAccuracy(name="acc"),
                           keras.metrics.Precision(name='precision'),
                           keras.metrics.Recall(name='recall')
                           ]
                  )
    history = model.fit(train_ds,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=val_ds
                        )
    return model, history, ckpt_file

def train_model_gray(model,
                     image_size,
                     train_ds,
                     val_ds,
                     model_folder,
                     epochs=100):
    """
    Trains the good/bad classification model
    inputs:
    model (keras model): this is defined in define_model()
    image_size (tuple, (xdim, ydim): dimensions of images for model
    train_ds: training dataset obtained from load_dataset
    val_ds: validation dataset obtained from load_dataset
    model_folder (str): path to save the model to
    epochs (int, optional): number of epochs to train for
    """
    model = define_model(input_shape=image_size + (1,), mode='train', num_classes=2)

    ##this makes a plot of the model, you need pydot installed, 
    keras.utils.plot_model(model, to_file=os.path.join(model_folder, 'model_graph.png'), show_shapes=True)
    
    ckpt_file = os.path.join(model_folder, "model_{epoch}.h5")
    epochs = 25
    lr_callback = keras.callbacks.LearningRateScheduler(scheduler)
    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='auto', restore_best_weights=True)
    callbacks = [keras.callbacks.ModelCheckpoint(ckpt_file),
                 early_stopping_callback,
                 lr_callback
                 ]
    model.compile(optimizer=keras.optimizers.Adam(3e-5),
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[keras.metrics.BinaryAccuracy(name="acc"),
                           keras.metrics.Precision(name='precision'),
                           keras.metrics.Recall(name='recall')
                           ]
                  )
    history = model.fit(train_ds,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=val_ds
                        )
    return model, history, ckpt_file

def sort_images(inference_df_path,
                output_folder,
                threshold=0.40):
    """
    Using model results to sort the images the model was run on into good and bad folders
    inputs:
    inference_df_path (str): path to the csv containing model results
    output_folder (str): path to the directory containing the inference images
    """
    good_nir = os.path.join(os.path.dirname(output_folder), 'NIR')
    bad_nir = os.path.join(good_nir, 'bad')
    good_swir = os.path.join(os.path.dirname(output_folder), 'SWIR')
    bad_swir = os.path.join(good_swir, 'bad')
    bad_dir = os.path.join(output_folder, 'bad')
    dirs = [output_folder, bad_dir, bad_nir, bad_swir]
    for d in dirs:
        try:
            os.mkdir(d)
        except:
            pass
    inference_df = pd.read_csv(inference_df_path)
    for i in range(len(inference_df)):
        input_image_path = inference_df['im_paths'].iloc[i]
        im_name = os.path.basename(input_image_path)
        im_name_no_ext = os.path.splitext(im_name)[0]
        sat = im_name_no_ext[-2:]
        idx = im_name_no_ext.find('RGB')
        im_name_nir = im_name_no_ext[0:idx]+'NIR_'+sat+'.jpg'
        im_name_swir = im_name_no_ext[0:idx]+'SWIR_'+sat+'.jpg'
        input_image_path_nir = os.path.join(good_nir, im_name_nir)
        input_image_path_swir = os.path.join(good_swir, im_name_swir)
        if inference_df['model_scores'].iloc[i] < threshold:
            output_image_path = os.path.join(bad_dir, im_name)
            output_image_path_nir = os.path.join(bad_nir, im_name_nir)
            output_image_path_swir = os.path.join(bad_swir, im_name_swir)
            try:
                shutil.move(input_image_path, output_image_path)
            except:
                pass
            try:
                shutil.move(input_image_path_nir, output_image_path_nir)
            except:
                pass
            try:
                shutil.move(input_image_path_swir, output_image_path_swir)
            except:
                pass
        
def run_inference_gray(path_to_model_ckpt,
                       path_to_inference_imgs,
                       output_folder,
                       result_path,
                       threshold=0.446,
                       sort=True):
    """
    Runs the trained model on images, classifying them either as good or bad
    Saves the results to a csv (image_path, class (good or bad), score (0 to 1)
    Sorts the images into good or bad folders
    Images should be '.jpg'
    inputs:
    path_to_model_ckpt (str): path to the saved keras model
    path_to_inference_imgs (str): path to the folder containing images to run the model on
    output_folder (str): path to save outputs to
    result_path (str): csv path to save results to
    threshold (float): threshold on sigmoid of model output (ex: 0.6 means mark images as good if model output is >= 0.6, or 60% sure it's a good image)
    sort (bool): True to sort images, False to not sort (this is mainly for testing)
    returns:
    result_path (str): csv path of saved results
    """
    try:
        os.mkdir(output_folder)
    except:
        pass
    image_size = (128, 128)
    model = define_model(input_shape=image_size + (1,), mode='inference', num_classes=2)
    model.load_weights(path_to_model_ckpt)
    types = ('*.jpg', '*.jpeg', '*.png') 
    im_paths = []
    for files in types:
        im_paths.extend(glob.glob(os.path.join(path_to_inference_imgs, files)))
    model_scores = [None]*len(im_paths)
    im_classes = [None]*len(im_paths)
    i=0
    for im_path in im_paths:
        img = keras.utils.load_img(im_path, color_mode='grayscale',target_size=image_size)
        img_array = keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = float(keras.activations.sigmoid(predictions[0][0]))
        model_scores[i] = score
        i=i+1
    ##save results to a csv
    df = pd.DataFrame({'im_paths':im_paths,
                       'model_scores':model_scores
                       }
                      )
    df.to_csv(result_path)
    if sort == True:
        sort_images(result_path,
                    output_folder,
                    threshold=threshold)
    return result_path

def run_inference_rgb(path_to_model_ckpt,
                      path_to_inference_imgs,
                      output_folder,
                      result_path,
                      threshold=0.335,
                      sort=True):
    """
    Runs the trained model on images, classifying them either as good or bad
    Saves the results to a csv (image_path, class (good or bad), score (0 to 1)
    Sorts the images into good or bad folders
    Images should be '.jpg'
    inputs:
    path_to_model_ckpt (str): path to the saved keras model
    path_to_inference_imgs (str): path to the folder containing images to run the model on
    output_folder (str): path to save outputs to
    result_path (str): csv path to save results to
    threshold (float): threshold on sigmoid of model output (ex: 0.6 means mark images as good if model output is >= 0.6, or 60% sure it's a good image)
    sort (bool): True to sort images, False to not sort (this is mainly for testing)
    returns:
    result_path (str): csv path of saved results
    """
    try:
        os.mkdir(output_folder)
    except:
        pass
    image_size = (128, 128)
    model = define_model(input_shape=image_size + (3,), mode='inference', num_classes=2)
    model.load_weights(path_to_model_ckpt)
    types = ('*.jpg', '*.jpeg', '*.png') 
    im_paths = []
    for files in types:
        im_paths.extend(glob.glob(os.path.join(path_to_inference_imgs, files)))
    model_scores = [None]*len(im_paths)
    im_classes = [None]*len(im_paths)
    i=0
    for im_path in im_paths:
        print(im_path)
        img = keras.utils.load_img(im_path, color_mode='rgb',target_size=image_size)
        img_array = keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = float(keras.activations.sigmoid(predictions[0][0]))
        model_scores[i] = score
        i=i+1
    ##save results to a csv
    df = pd.DataFrame({'im_paths':im_paths,
                       'model_scores':model_scores
                       }
                      )
    print(result_path)

    df.to_csv(result_path)
    if sort == True:
        sort_images(result_path,
                    output_folder,
                    threshold=threshold)
    return result_path

def plot_history(history,
                 history_save_path):
    """
    This makes a plot of the loss curve
    inputs:
    history: history object from model.fit_generator
    history_save_path (str): path to save the plot to
    """
    plt.subplot(4,1,1)
    plt.plot(history.history['loss'], color='b')
    plt.plot(history.history['val_loss'], color='r')
    plt.minorticks_on()
    plt.ylabel('Loss (BCE)')
    plt.xlabel('Epoch')
    plt.legend(['Training Data', 'Validation Data'],loc='upper right')
    
    plt.subplot(4,1,2)
    plt.plot(history.history['acc'], color='b')
    plt.plot(history.history['val_acc'], color='r')
    plt.minorticks_on()
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training Data', 'Validation Data'],loc='upper right')
    
    plt.subplot(4,1,3)
    plt.plot(history.history['precision'], color='b')
    plt.plot(history.history['val_precision'], color='r')
    plt.minorticks_on()
    plt.ylabel('Precision')
    plt.xlabel('Epoch')
    plt.legend(['Training Data', 'Validation Data'],loc='upper right')

    plt.subplot(4,1,4)
    plt.plot(history.history['recall'], color='b')
    plt.plot(history.history['val_recall'], color='r')
    plt.minorticks_on()
    plt.ylabel('Recall')
    plt.xlabel('Epoch')
    plt.legend(['Training Data', 'Validation Data'],loc='upper right')
    plt.tight_layout()
    plt.savefig(history_save_path, dpi=300)
    plt.close('all')
    
def training(path_to_training_data,
             output_folder,
             epochs=25,
             mode='gray'):
    """
    Training the good/bad classification model
    inputs:
    path_to_training_data (str): path to the directory containing good and bad subdirectories
    output_folder (str): path to save outputs to
    epoch (int, optional): number of epochs to train for
    mode (str): 'gray' or 'rgb', default is 'gray'
    """
    ##Making new directories
    try:
        os.mkdir(output_folder)
    except:
        pass
    
    model_folder = os.path.join(output_folder, 'models')
    history_save_path = os.path.join(model_folder, 'history.png')
    try:
        os.mkdir(model_folder)
    except:
        pass

    ##Setting image size for model and loading datasets
    image_size = (128, 128)


    ##Defining and then training model
    if mode=='gray':
        train_ds, val_ds = load_dataset_gray(path_to_training_data,
                                             image_size,
                                             32)
        model = define_model(input_shape=image_size + (1,), mode='train', num_classes=2)
        model, history, ckpt_file = train_model_gray(model,
                                                     image_size,
                                                     train_ds,
                                                     val_ds,
                                                     model_folder,
                                                     epochs=epochs)
    else:
        train_ds, val_ds = load_dataset_rgb(path_to_training_data,
                                            image_size,
                                            32)
        model = define_model(input_shape=image_size + (3,), mode='train', num_classes=2)
        
        model, history, ckpt_file = train_model_rgb(model,
                                                    image_size,
                                                    train_ds,
                                                    val_ds,
                                                    model_folder,
                                                    epochs=epochs)
    
    best_ckpt_file = os.path.join(model_folder, 'best.h5')
    model.save(best_ckpt_file)

    ##Plotting and saving loss curve
    plot_history(history, history_save_path)
    hist_df = pd.DataFrame(history.history) 
    # save to csv: 
    hist_csv_file = os.path.join(os.path.join(model_folder, 'history.csv'))
    hist_df.to_csv(hist_csv_file, index=False)    

    return best_ckpt_file

def inference_multiple_sessions(home, threshold, model='rgb', sort=True):
    """
    Runs filter on multiple CoastSeg data sessions, will skip a site if there is already a good_bad.csv
    inputs:
    home (str): path to where each data folder is
    threshold (float): threshold on sigmoid of model output (ex: 0.6 means mark images as good if model output is >= 0.6, or 60% sure it's a good image)
    model (str): 'rgb' or 'gray'
    sort (bool): True to sort images, False to not sort (this is mainly for testing)
    """
    sites = get_immediate_subdirectories(home)
    for site in sites:
        site = os.path.join(home, site)
        csv_path =  os.path.join(site, 'good_bad.csv')
        if os.path.isfile(csv_path):
            print('skip ' + site)
            continue
        else:
            print('doing ' + site)
            if model=='rgb':
                run_inference_rgb(os.path.join(get_script_path(), 'models', 'image_rgb', 'best.h5'),
                                  os.path.join(site, 'jpg_files', 'preprocessed', 'RGB'),
                                  os.path.join(site, 'jpg_files', 'preprocessed', 'RGB'),
                                  os.path.join(site, 'good_bad.csv'),
                                  threshold,
                                  sort=sort
                                  )
            else:
                run_inference_gray(os.path.join(get_script_path(), 'models', 'image_rgb', 'best.h5'),
                                   os.path.join(site, 'jpg_files', 'preprocessed', 'RGB'),
                                   os.path.join(site, 'jpg_files', 'preprocessed', 'RGB'),
                                   os.path.join(site, 'good_bad.csv'),
                                   threshold,
                                   sort=sort
                                   )

def sort_multiple_sessions(home, threshold, model='rgb'):
    """
    Runs filter on multiple CoastSeg data sessions, will skip a site if there is already a good_bad.csv
    inputs:
    home (str): path to where each data folder is
    """
    sites = get_immediate_subdirectories(home)
    for site in sites:
        site = os.path.join(home, site)
        csv_path =  os.path.join(site, 'good_bad.csv')
        print('doing ' + site)
        sort_images(csv_path,
                    os.path.join(site, 'jpg_files', 'preprocessed', 'RGB'),
                    threshold=threshold)

def train_and_test(dataset):
    """
    trains, tests, outputs metric figures
    inputs:
    dataset (str): path to the dataset
                   dataset
                          train
                               good
                               bad
                          test
                               good
                               bad
    """
    try:
        os.mkdir(os.path.join(get_script_path(), 'test_results'))
    except:
        pass
    ##train grayscale model
    training(os.path.join(dataset,'train'),
            get_script_path(),
            epochs=50,
            mode='gray')
    ##train rgb model
    training(os.path.join(dataset,'train'),
            get_script_path(),
            epochs=50,
            mode='rgb')
    ##test gray model
    test_dir = os.path.join(dataset, 'test')
    test_dir_good = os.path.join(dataset, 'test', 'bad')
    run_inference_gray(os.path.join(get_script_path(), 'models', 'gray', 'best.h5'),
                    test_dir_good,
                    test_dir_good,
                    os.path.join(get_script_path(), 'test_results', 'result_test_bad_gray.csv'),
                    threshold=0.20,
                    sort=False)
    test_dir = os.path.join(dataset, 'test')
    test_dir_bad = os.path.join(dataset, 'test', 'good')
    run_inference_gray(os.path.join(get_script_path(), 'models', 'gray', 'best.h5'),
                    test_dir_bad,
                    test_dir_bad,
                    os.path.join(get_script_path(), 'test_results', 'result_test_good_gray.csv'),
                    threshold=0.20,
                    sort=False)

    ##test rgb model
    test_dir = os.path.join(dataset, 'test')
    test_dir_good = os.path.join(dataset, 'test', 'bad')
    run_inference_rgb(os.path.join(get_script_path(), 'models', 'rgb', 'best.h5'),
                    test_dir_good,
                    test_dir_good,
                    os.path.join(get_script_path(), 'test_results', 'result_test_bad_rgb.csv'),
                    threshold=0.20,
                    sort=False)
    test_dir = os.path.join(dataset, 'test')
    test_dir_bad = os.path.join(dataset, 'test', 'good')
    run_inference_rgb(os.path.join(get_script_path(), 'models', 'rgb', 'best.h5'),
                    test_dir_bad,
                    test_dir_bad,
                    os.path.join(get_script_path(), 'test_results', 'result_test_good_rgb.csv'),
                    threshold=0.20,
                    sort=False)
    ##make figures
    os.system('python metrics_image.py')

