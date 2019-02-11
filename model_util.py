import os
import numpy as np

from datetime import datetime
from shutil import copyfile

from tensorflow.keras.models import model_from_json

#
# A generic unction to save the model and the weights as follows:
# - Model saved in the file 'filename.json' as a json file
# - The weights in the file 'filename.h5'
# filename being ghe full filename without the extension. Example: dir1/dir2/model
# 
def SaveModel(model, filename):
    if filename!= None:
        folder = os.path.dirname(filename)
        
        # Check if the directory to save the file at exists. If not create it
        if not os.path.exists(folder):
            try:
                os.makedirs(folder)
            except IOError as err:
                print("Error while creating folder (" + str(folder) + "). Error[" + str(err.errno) + "]: " + str(err.strerror))
        
        # Save the model in json file
        file = filename + ".json"
        if os.path.exists(file):
            print("File %s already exists. Backing it up to %s" % (file, filename + datetime.today().strftime('%Y%m%d') + ".json"))
            copyfile(file, filename + datetime.today().strftime('%Y%m%d') + ".json")
        with open(file, "w") as json_file:
            json_file.write(model.to_json())

        # Save weights in h5 file
        file = filename + ".h5"
        if os.path.exists(file):
            print("File %s already exists. Backing it up to %s" % (file, filename + datetime.today().strftime('%Y%m%d') + ".h5"))
            copyfile(file, filename + datetime.today().strftime('%Y%m%d') + ".h5")
        model.save_weights(file)


#
# A generic unction to load a model and its weights as follows:
# - Model read from json file 'filename.json'
# - Weights read from file 'filename.h5'
# filename being ghe full filename without the extension. Example: dir1/dir2/model
#
def LoadModel(filename, custom_objects):
    model = None
    
    if filename != None:
        # load and create model from json file
        file = filename + ".json"
        if os.path.exists(file):
            with open(file, "r") as json_file:
                model = model_from_json(json_file.read(), custom_objects=custom_objects)
        
            # load weights from h5 file into new model
            file = filename + ".h5"
            if os.path.exists(file):
                model.load_weights(file)
            else:
                print("File %s does not exist" % (file))
        else:
            print("File %s does not exist" % (file))
    
    return model

#
# Function to save training, validation and testing results into a file.
# The saved results will be as follows:
# 1- Training info: information about the training parameters used:
#    - Loss function
#    - Optimisation method
#    - Learning rate
#    - Batch size
#    - Number of epochs
# 2- Training results: The last value reached while training for each of the metrics that are
#    available in history which is a Keras callback dictionary containing all training and
#    validation (if available) metrics
#    The metrics to be printed are the ones passed in 'metrics' (an array of metrics)
# 3- Validation results: Same as Training results (if available)
# 4- Testing results: The same metrics values for testing (if available), passed via the dictionary
#    'test_result'
# 5- Model Summary
#
def SaveResults(model, init, history, test_result, metrics):
    if init != None and init.save != None:
        file = init.save + ".txt"
        if os.path.exists(file):
            print("File %s already exists. Backing it up to %s" % (file, init.save + datetime.today().strftime('%Y%m%d') + ".txt"))
            copyfile(file, init.save + datetime.today().strftime('%Y%m%d') + ".txt")
        
        # Save information about Training and Validation parameters, hyper-parameters and results
        with open(init.save + ".txt", "w") as f:
            f.write("Training Info:\n")
            f.write("\tLoss Function: " + str(init.loss) + "\n")
            f.write("\tOptimisation Method: " + str(init.optimiser) + "\n")
            f.write("\tLearning Rate: " + str(init.lr) + "\n")
            f.write("\tBatch Size: " + str(init.batchsize) + "\n")
            f.write("\tNumber of Epochs: " + str(init.epochs) + "\n")

            f.write("\nTraining Results:\n")
            for m in metrics:
                key = m
                f.write("\t" + str(m.title()) + ": " + str(history[key][-1]) + "\n")

            if init.validate == True:
                f.write("\nValidation Results:\n")
                for m in metrics:
                    key = "val_" + m
                    f.write("\t" + str(m.title()) + ": " + str(history[key][-1]) + "\n")

            if init.evaltest == True:
                f.write("\nTesting Results:\n")
                for m in metrics:
                    f.write("\t" + str(m.title()) + ": " + str(test_result[m]) + "\n")

            f.write("\nModel Summary:\n")
            model.summary(print_fn=lambda x: f.write('\t' + x + '\n'))


#
# Save all history values in order to be plotted later if needed
#
# The format for saving those is as follows:
#    First line will contain the metrics names separated by commas
#    The next epochs-lines (one record for each epoch) will contain the
#    values for each metric, also comma separated
# So as a summary, the file will contain a column for each metric and as 
# many records as the epochs values.
# File dimensions: Epochs + 1 lines (+1 for the name of metrics)
#                  Number of Metrics columns
#
def SaveHistory(filename, history):    
    if filename != None:
        key_list = []
        for key in history.keys():
            key_list.append(key)
        header = ','.join(key_list)
        h_array = np.concatenate(
                      (np.array(list(history.keys())).reshape(-1, 1).T,
                        np.array(list(history.values())).T),
                        axis=0)

        file = filename + "_history.csv"
        if os.path.exists(file):
            print("File %s already exists. Backing it up to %s" % (file, filename + datetime.today().strftime('%Y%m%d') + "_history.csv"))
            copyfile(file, filename + datetime.today().strftime('%Y%m%d') + "_history.csv")

        np.savetxt(file, np.array(list(history.values())).T, header=header, delimiter=',')
