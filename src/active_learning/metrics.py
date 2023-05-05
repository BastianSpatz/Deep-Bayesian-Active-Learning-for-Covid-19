from baal.utils.metrics import Metrics
import numpy as np
from sklearn.metrics import confusion_matrix


class ClassificationReport(Metrics):
    """
    Compute a classification report as a metric.
    Args:
        num_classes (int): the number of classes.
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        super().__init__(average=False)

    def reset(self):
        self.class_data = np.zeros([self.num_classes, self.num_classes])

    def update(self, output=None, target=None):
        """
        Update the confusion matrice according to output and target.
        Args:
            output (tensor): predictions of model
            target (tensor): labels
        """
        output = output.detach().cpu().numpy()
        target = target.detach().cpu().numpy()

        if output.ndim > target.ndim:
            # Argmax not done
            output = output.argmax(1)  # 1 is always our class axis.

        self.class_data += confusion_matrix(
            target.reshape([-1]).astype(int),
            output.reshape([-1]).astype(int),
            labels=np.arange(self.class_data.shape[0]),
        )

    @property
    def value(self):
        # print("\n" + str(self.class_data))
        fp = self.class_data.sum(axis=0) - np.diag(self.class_data)
        fn = self.class_data.sum(axis=1) - np.diag(self.class_data)
        tp = np.diag(self.class_data)
        tn = self.class_data.sum() - (fp + fn + tp)
        acc = (tp + tn) / np.maximum(1, tp + fp + fn + tn)
        precision = tp / np.maximum(1, tp + fp)
        recall = tp / np.maximum(1, tp + fn)
        return {"accuracy": acc, "precision": precision, "recall": recall}
