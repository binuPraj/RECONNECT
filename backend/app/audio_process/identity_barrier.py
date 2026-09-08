"""Order only the identity-resolution stage of concurrently processed recordings."""

from dataclasses import dataclass, field
from threading import Condition


@dataclass
class _BarrierState:
    next_recording_id: int
    completed: set[int] = field(default_factory=set)


class SessionIdentityResolutionBarrier:
    """Allow one session's recognition commits to proceed in recording order.

    Pre-recognition work remains concurrent. Call ``register`` when a recording
    is submitted, ``wait_for_turn`` immediately before recognition mutates
    session identity state, and ``complete`` once that state is committed.
    """

    def __init__(self):
        self._condition = Condition()
        self._states: dict[str, _BarrierState] = {}

    def register(self, session_id: str, recording_id: int) -> None:
        with self._condition:
            if session_id not in self._states:
                self._states[session_id] = _BarrierState(recording_id)

    def wait_for_turn(self, session_id: str, recording_id: int) -> None:
        with self._condition:
            state = self._states.setdefault(session_id, _BarrierState(recording_id))
            while recording_id > state.next_recording_id:
                self._condition.wait()

    def complete(self, session_id: str, recording_id: int) -> None:
        with self._condition:
            state = self._states.setdefault(session_id, _BarrierState(recording_id))
            if recording_id < state.next_recording_id:
                return
            state.completed.add(recording_id)
            while state.next_recording_id in state.completed:
                state.completed.remove(state.next_recording_id)
                state.next_recording_id += 1
            self._condition.notify_all()
