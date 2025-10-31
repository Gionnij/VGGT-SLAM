import numpy as np
from sklearn.metrics import confusion_matrix

"""https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/master/metrics/stream_metrics.py"""
class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()


class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())

    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k != "Class IoU":
                string += "%s: %f\n" % (k, v)

        # string+='Class IoU:\n'
        # for k, v in results['Class IoU'].items():
        #    string += "\tclass %d: %f\n"%(k, v)
        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self,ignored_classes=None):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        if ignored_classes is None:
            ignored_classes = []
        hist = self.confusion_matrix
        acc = _safe_divide(np.diag(hist).sum(), hist.sum())
        acc_cls = _safe_divide(np.diag(hist), hist.sum(axis=1))
        acc_cls = np.nanmean(acc_cls)
        #iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        num =np.diag(hist)
        denom=(hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        iu=_safe_divide(num,denom)
        num_micro = num.sum()
        denom_micro = denom.sum()
        mean_iu_micro = np.nanmean(_safe_divide(num_micro,denom_micro))
        mean_iu = np.nanmean(iu)
        freq = _safe_divide(hist.sum(axis=1), hist.sum())
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))
        mean_cls_iu=np.nanmean(np.array([v for k, v in cls_iu.items() if k not in ignored_classes]))
        return {
            "Overall Acc": acc,
            "Mean Acc": acc_cls,
            "FreqW Acc": fwavacc,
            "Mean IoU": mean_iu,
            "Micro Mean IoU": mean_iu_micro,
            "Class IoU": cls_iu,
            "Mean Class IoU": mean_cls_iu,
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


def _safe_divide(num: np.ndarray, denom: np.ndarray) -> np.ndarray:
    """Safe division, by preventing division by zero.

    Additionally casts to float if input is not already to secure backwards compatibility.
    """
    if len(denom.shape)!=0:
        denom[denom == 0.0] = 1.0
    return num / denom

class AverageMeter(object):
    """Computes average values"""

    def __init__(self):
        self.book = dict()

    def reset_all(self):
        self.book.clear()

    def reset(self, id):
        item = self.book.get(id, None)
        if item is not None:
            item[0] = 0
            item[1] = 0

    def update(self, id, val):
        record = self.book.get(id, None)
        if record is None:
            self.book[id] = [val, 1]
        else:
            record[0] += val
            record[1] += 1

    def get_results(self, id):
        record = self.book.get(id, None)
        assert record is not None
        return record[0] / record[1]