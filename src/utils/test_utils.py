import sys
import torch
from tqdm import tqdm as tqdm
import numpy as np
from numpy import mean

global_lists = {}


class Meter(object):
    """Meters provide a way to keep track of important statistics in an online manner.
    This class is abstract, but provides a standard interface for all meters to follow.
    """

    def reset(self):
        """Reset the meter to default settings."""
        pass

    def add(self, value):
        """Log a new value to the meter
        Args:
            value: Next result to include.
        """
        pass

    def value(self):
        """Get the value of the meter in the current state."""
        pass

class AverageValueMeter(Meter):
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val = 0

    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean = 0.0 + self.sum  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan


class Epoch:
    def __init__(self, model, loss, metrics, stage_name, device="cpu", verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s

    def batch_update(self, x1, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):
        global global_lists  # Declare the variable as global

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(
            dataloader,
            desc=self.stage_name,
            file=sys.stdout,
            disable=not (self.verbose),
        ) as iterator:
            for x1, y in iterator:
                x1, y = x1.to(self.device), y.to(self.device)
                # print("x1 shape - ", x1.shape)
                # print("y shape - ", y.shape)
                loss, y_pred = self.batch_update(x1, y)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    print('Computing :', metric_fn.__name__)
                    metrics_meters = {metric.__name__: [AverageValueMeter() for _ in range(5)] for metric in self.metrics}

                    for metric_fn in self.metrics:
                        metric_values = metric_fn(y_pred, y.int()).cpu().detach().numpy()  # Assume metric_values is a list or array of length 5
                        for i, metric_value in enumerate(metric_values):
                            metrics_meters[metric_fn.__name__][i].add(metric_value)

                metrics_logs = {}
                dynamic_lists={}
                # list0, list1, list2, list3, list4 = [],[],[],[],[]
                for metric_name, meters in metrics_meters.items():
                    for i, meter in enumerate(meters):
                        class_specific_metric_name = f"{metric_name}_class{i}"
                        metrics_logs[class_specific_metric_name] = meter.mean
                        for i, meter in enumerate(meters):
                            key = f"list{i}"
                            if key not in dynamic_lists:
                                dynamic_lists[key] = []
                            dynamic_lists[key].append(meter)

                logs.update(metrics_logs)
                #Print each list
                for key, value in dynamic_lists.items():
                    float_values = [v.mean for v in value]
                    if key not in global_lists:
                        global_lists[key] = []
                    global_lists[key].extend(float_values)
                    
                    # print(f"Local {key}: {float_values}")
                    # print(f"Global {key}: {global_lists[key]}")



                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpoch(Epoch):
    def __init__(self, model, loss, metrics, optimizer, device ='cuda', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="train",
            device=device,
            verbose=verbose
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x1, y):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x1)
        # print("\nprediction shape - ", prediction.shape)
        # print("y shape - ", y.shape)
        loss = self.loss(prediction, y)
        loss.backward()
        self.optimizer.step()
        return loss, prediction


class ValidEpoch(Epoch):
    def __init__(self, model, loss, metrics, device='cuda', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="valid",
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x1, y):
        with torch.no_grad():
            prediction = self.model.forward(x1)
            loss = self.loss(prediction, y)
        return loss, prediction