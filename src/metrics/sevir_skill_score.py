from typing import Sequence

import torch
from torchmetrics import Metric

from ..dataset.preprocessing import denormalize_data_dict


def _threshold(target, pred, T):
    """
    Returns binary tensors t,p the same shape as target & pred.  t = 1 wherever
    target > t.  p =1 wherever pred > t.  p and t are set to 0 wherever EITHER
    t or p are nan.
    This is useful for counts that don't involve correct rejections.

    Parameters
    ----------
    target
        torch.Tensor
    pred
        torch.Tensor
    T
        numeric_type:   threshold
    Returns
    -------
    t
    p
    """
    t = (target >= T).float()
    p = (pred >= T).float()
    is_nan = torch.logical_or(torch.isnan(target), torch.isnan(pred))
    t[is_nan] = 0
    p[is_nan] = 0
    return t, p


class SEVIRSkillScore(Metric):
    r"""
    The calculation of skill scores in SEVIR challenge is slightly different:
        `mCSI = sum(mCSI_t) / T`
    See https://github.com/MIT-AI-Accelerator/sevir_challenges/blob/dev/radar_nowcasting/RadarNowcastBenchmarks.ipynb for more details.
    """
    full_state_update: bool = True

    def __init__(
        self,
        denormalize_method: str = "sevir",
        threshold_list: Sequence[int] = (16, 74, 133, 160, 181, 219),
        metrics_list: Sequence[str] = ("csi", "bias", "sucr", "pod"),
        eps: float = 1e-4,
        dist_sync_on_step: bool = False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.denormalize_method = denormalize_method
        self.threshold_list = threshold_list
        self.metrics_list = metrics_list
        self.eps = eps

        state_shape = (len(self.threshold_list),)

        self.add_state("hits", default=torch.zeros(state_shape), dist_reduce_fx="sum")
        self.add_state("misses", default=torch.zeros(state_shape), dist_reduce_fx="sum")
        self.add_state("fas", default=torch.zeros(state_shape), dist_reduce_fx="sum")

    @staticmethod
    def pod(hits, misses, fas, eps):
        return hits / (hits + misses + eps)

    @staticmethod
    def sucr(hits, misses, fas, eps):
        return hits / (hits + fas + eps)

    @staticmethod
    def csi(hits, misses, fas, eps):
        return hits / (hits + misses + fas + eps)

    @staticmethod
    def bias(hits, misses, fas, eps):
        bias = (hits + fas) / (hits + misses + eps)
        logbias = torch.pow(bias / torch.log(torch.tensor(2.0)), 2.0)
        return logbias

    def calc_hits_misses_fas(self, pred, target, threshold):
        with torch.no_grad():
            t, p = _threshold(target, pred, threshold)
            hits = torch.sum(t * p).int()
            misses = torch.sum(t * (1 - p)).int()
            fas = torch.sum((1 - t) * p).int()
        return hits, misses, fas

    def preprocess(self, pred, target):
        if self.denormalize_method == "sevir":
            pred = denormalize_data_dict(data_dict={"vil": pred.detach().float()}, method="sevir")["vil"]
            target = denormalize_data_dict(data_dict={"vil": target.detach().float()}, method="sevir")["vil"]
        else:
            raise NotImplementedError
        return pred, target

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        # preprocessing was moved to validation and test steps in the PL modules instead
        # pred, target = self.preprocess(pred, target)

        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas = self.calc_hits_misses_fas(pred, target, threshold)
            self.hits[i] += hits
            self.misses[i] += misses
            self.fas[i] += fas

    def compute(self):
        metrics_dict = {"pod": self.pod, "csi": self.csi, "sucr": self.sucr, "bias": self.bias}
        ret = {}
        for threshold in self.threshold_list:
            ret[threshold] = {}
        ret["avg"] = {}
        for metrics in self.metrics_list:
            score_avg = 0
            # shape = (len(threshold_list),)
            scores = metrics_dict[metrics](self.hits, self.misses, self.fas, self.eps)
            scores = scores.detach().cpu().numpy()
            for i, threshold in enumerate(self.threshold_list):
                score = scores[i].item()  # shape = (1, )
                ret[threshold][metrics] = score
                score_avg += score
            score_avg /= len(self.threshold_list)
            ret["avg"][metrics] = score_avg
        return ret
