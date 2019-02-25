# Neural Network Utilities

This is a repository that contains generic utilities used by other deep learning projects.

The following are currently available:
## model_util
A collection of generic functions that can be used to handle Tensorflow Keras models. It contains the following functions:
- __SaveModel:__ Saves a model and its weights into a json file and a .h5 file respectively
- __LoadModel:__ Reads a model from a json file and weights from a .h5 file to be used for further training or prediction
- __SaveResults:__ Saves training, validation and testing results as well as model summary into a text file
- __SaveHistory:__ Saves the training history data into a csv file

## Msglog
A module that optimises the logging class in order to generate logs with the following format:<br>
`YYYY-mm-dd HH:MM:SS.msec (process_id) (levelname) module.function -> message`<br>
Example:<br>
```2019-02-21 18:44:37.548 (668456) (INFO) main.<module> -> Tensorflow version: 1.11.0```<br>
### Functionality
This module contains one function:<br>
`LogInit(name, filename, debuglevel = logging.INFO, log = True)`
- __name:__ Name of the logger that is passed from the maon python file and which must be used by all other files
- __filename:__ Full path of the file where logs are to be written. This is a TimedRotatingFile that rotates at midnight.
- __debuglevel:__ The minimum log level that will be written into the file. All messages with level less than debuglevel will not be written into the file handler. Please check the logging class documentation for more info.
- __log:__ Whether to generate logs or not.

This module will do the following:
- All messages with level Error or CRITICAL will be written on the terminal where the main python file is being run regardless of the argument `log` that is passed. All error and critical messages are written on the terminal
- If `log == True`, a logfile will be created with all messages whose level is greater than debuglevel.
### Usage
In the main python file, do the following:
- from Msglog import LogInit
- log = LogInit('logger-name', logfilename, debuglevel)

In all other python files, add the followuing
- import logging
- log = logging.getLogger('logger-name') where logger-name is the same one used in the first parameter when LogInit was called.

You can then write to logfile using the log class.<br>
Example: `log.info("Tensorflow version: %s", tf.__version__)`
