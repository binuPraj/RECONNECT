import threading
import time

from app.audio_process.identity_barrier import SessionIdentityResolutionBarrier


def test_barrier_orders_one_session_without_blocking_another():
    barrier = SessionIdentityResolutionBarrier()
    barrier.register("session-a", 1)
    barrier.register("session-a", 2)
    barrier.register("session-b", 1)

    second_entered = threading.Event()
    other_entered = threading.Event()

    def wait_for_second() -> None:
        barrier.wait_for_turn("session-a", 2)
        second_entered.set()

    def wait_for_other_session() -> None:
        barrier.wait_for_turn("session-b", 1)
        other_entered.set()

    second = threading.Thread(target=wait_for_second)
    other = threading.Thread(target=wait_for_other_session)
    second.start()
    other.start()

    assert other_entered.wait(timeout=1)
    assert not second_entered.wait(timeout=0.05)

    barrier.complete("session-a", 1)
    assert second_entered.wait(timeout=1)

    barrier.complete("session-a", 2)
    barrier.complete("session-b", 1)
    second.join(timeout=1)
    other.join(timeout=1)
    assert not second.is_alive()
    assert not other.is_alive()


def test_barrier_completion_is_idempotent():
    barrier = SessionIdentityResolutionBarrier()
    barrier.register("session-a", 1)
    barrier.complete("session-a", 1)
    barrier.complete("session-a", 1)

    barrier.register("session-a", 2)
    start = time.monotonic()
    barrier.wait_for_turn("session-a", 2)
    assert time.monotonic() - start < 0.1
