from evorl.types import PyTreeData


class ExponetialScheduleSpec(PyTreeData):
    init: float
    final: float
    decay: float
