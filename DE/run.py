
def run_DE(train, test, perf_measure, learner):
    """
     This function would take a train dataset and would find parameter using DE by performing CV with fold == 5.
    :param train: file path of train
    :param test: file path of test
    :param learner: 
    :param perf_measure: Accuracy, recall etc.
    :return: performance measure
    """