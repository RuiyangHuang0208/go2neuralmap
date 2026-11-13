from .state_recorder import StateRecorder, StateRecorderConfig, NullStateRecorder

StateRecorderRegistry = {
    "state_recorder": (StateRecorder, StateRecorderConfig()),
    "null_recorder": (NullStateRecorder, StateRecorderConfig()),
}
