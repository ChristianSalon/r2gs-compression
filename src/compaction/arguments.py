from argparse import ArgumentParser
from enum import StrEnum

from r2_gaussian.utils.argument_utils import ParamGroup


class CompactionMethod(StrEnum):
    PUP = "pup"


class CompactionParams(ParamGroup):
    def __init__(self, parser: ArgumentParser, fill_none: bool = False):
        self.compaction_method = CompactionMethod.PUP
        self.prune_ratio = 0.9

        super().__init__(parser, "Compaction parameters", fill_none)
