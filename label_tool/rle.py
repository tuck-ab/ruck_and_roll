import numpy as np

from .labels import Label, mapper

SEPARATOR_CHAR = ":"

class LabelTracker:
    """Object to deal with labels. Supports methods to add and remove
    labels to be used while labelling a video. Run length encoding
    to compress the labels into a smaller more user understandable
    format. Conversion between different data structures eg numpy
    `ndarray`
    """
    def __init__(self):
        self._runs = []
        
        self._previous_label = Label.NOTHING
        self._current_run_len = 0
        
    def add_label(self, label):
        """Add the label for the next frame

        Args:
            label (Label): The label of the frame
        """
        ## If the label is in the same run
        if self._previous_label == label:
            ## Increase run length
            self._current_run_len += 1
        else:
            ## If its different then add the completed run to the list
            self._runs.append((self._previous_label, self._current_run_len))
            
            ## Start a new run
            self._previous_label = label
            self._current_run_len = 1
            
    def undo_label(self):
        """Undo the label assigned to the most recent frame
        """
        ## If its the first label in a run
        if self._current_run_len <= 1:
            if len(self._runs) > 0:
                ## Reload the previous run
                self._previous_label, self._current_run_len = self._runs.pop()
            else: ## If its the start of the video then restart all the runs
                self._previous_label = Label.NOTHING
                self._current_run_len = 0
        else: ## Otherwise decrease size of the run
            self._current_run_len -= 1
        
    def load_from_file(self, path: str):
        """Loads the labels from a lbl file.

        Args:
            path (str): Path of the file to load the labels from

        Returns:
            LabelTracker: Returns an instance of itself
        """
        
        with open(path, "rt") as f:
            raw_data = f.read()
            
        runs = [row.split(SEPARATOR_CHAR) for row in raw_data.split("\n")]
        
        ## Remove any hanging lines
        while len(runs[-1]) != 2:
            runs.pop()
            
        ## Parse the length to be an integer and label to enum
        runs = [(mapper[label], int(length)) for label, length in runs]
        
        self._previous_label, self._current_run_len = runs.pop()
        
        return self
    
    def write_to_file(self, path: str):
        """Writes the labels stored to a lbl file

        Args:
            path (str): Path of the file to write the label to
        """
        self._runs.append((self._previous_label, self._current_run_len))
        
        with open(path, "wt") as f:
            f.write("\n".join([f"{x[0]}{SEPARATOR_CHAR}{x[1]}" for x in self._runs]))

        self._runs.pop()

    def as_numpy_array(self):
        """Returns the decompressed labels as a numpy `ndarray`

        Returns:
            ndarray: Numpy array of decompressed labels
        """
        temp = self._runs.copy()
        temp.append((self._previous_label, self._current_run_len))
        
        return np.array(temp)