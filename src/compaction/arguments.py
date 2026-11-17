from argparse import ArgumentParser
from enum import Enum

from utils.param_group import ExtendedParamGroup


class CompactionMethod(Enum):
    PUP = "pup"


class CompactionParams(ExtendedParamGroup):
    def __init__(self, parser: ArgumentParser, fill_none: bool = False):
        self.compaction_method = CompactionMethod.PUP.value
        self.prune_ratios = [0.9]
        self.refine = False
        self.scene_file_name = "point_cloud.pickle"
        self.volume_file_name = "vol_pred_compressed.npy"

        super().__init__(parser, "Compaction parameters", fill_none)
