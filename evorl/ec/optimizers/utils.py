from evorl.types import PyTreeData


class ExponentialScheduleSpec(PyTreeData):
    init: float
    final: float
    decay: float
