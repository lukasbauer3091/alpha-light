from pylsl import StreamInlet, resolve_byprop
import numpy as np


def BCI(inlet, classifier, transformer=None, action=print, buffer_length=1024, n_channels=5):
    '''
    Implements a generic Brain-Computer Interface.
    
    Classification is continuously performed on the signal given in `inlet`, and at
    each classification, an `action` is performed.


    Arguments:
        inlet: a pylsl `StreamInlet` of the brain signal.
        classifier: a function which performs classification on the most recent data (transformed as needed). returns class.
        transformer: function which takes in the most recent data (`buffer`) and returns the transformed input the classifer expects.
        action: a function which takes in the classification, and performs some action.
        buffer_length(int): the length of the `buffer`; specifies the number of samples of the signal to keep for classification.
        n_channels(int): the number of channels in the signal.
    '''

    inlet.open_stream()

    buffer = np.empty((0, n_channels)) # TODO: n_channels can be found from `inlet`. perhaps remove need for it to be passed in as a param.

    running = True # currently constantly running. TODO: implement ending condition?
    while running:
        chunk = inlet.pull_chunk()[0]
        if np.size(chunk) != 0: # Check if new data available
            buffer = np.append(buffer, np.array(chunk), axis=0)
            
            if buffer.shape[0] > buffer_length: 
                buffer = buffer[-buffer_length:] # clip to buffer_length

                # transform buffer for classification
                if transformer is not None:
                    clf_input = transformer(buffer) 
                else:
                    clf_input = buffer
                
                classification = classifier(clf_input) # perform classification

                action(classification) # run action based on classification
            
    inlet.close_stream()