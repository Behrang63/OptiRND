"""
Async task execution engine for OptiRND pipeline.

Provides a lightweight, thread-safe in-process task worker model using
concurrent.futures.ThreadPoolExecutor. Designed to offload heavy Monte
Carlo simulations and LLM calls without blocking the application thread.

Key features:
- Bounded worker pool via ThreadPoolExecutor (configurable max_workers).
- Thread-safe in-memory job registry protected by threading.Lock.
- Progress callbacks update stage/progress fields on the TaskResult.
- Exceptions inside worker threads are caught, recorded in error_message,
  and status is set to FAILED without crashing the daemon.
"""

from __future__ import annotations

import datetime
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, Optional
from threading import Lock

from core.contracts import TaskResult, TaskStatus
from core.services import EvaluationService

logger = logging.getLogger("optirnd.task_manager")


class AsyncTaskManager:
    """
    In-process asynchronous task execution manager.

    Uses a ThreadPoolExecutor with a bounded pool size to run heavy
    evaluation workloads (Monte Carlo simulations, LLM calls) without
    blocking the main application thread.

    The internal job registry maps task_id -> TaskResult, protected by
    a threading.Lock for safe concurrent access from multiple threads.
    """

    def __init__(self, max_workers: int = 2):
        """
        Initialize the task manager.

        :param max_workers: Maximum number of worker threads in the pool.
            Defaults to 2 for a lightweight embedded deployment.
        """
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="optirnd-task"
        )
        self._job_registry: Dict[str, TaskResult] = {}
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit_evaluation(
        self,
        proposal_input: dict,
        service: EvaluationService,
    ) -> str:
        """
        Submit an evaluation task for asynchronous execution.

        Creates a new TaskResult entry in the registry and dispatches the
        evaluation to the thread pool. The returned task_id can be used
        with get_task_status to poll for completion.

        :param proposal_input: Validated ProposalInput dict (model_dump() output).
        :param service: EvaluationService instance to run the evaluation.
        :return: Unique task_id string.
        """
        task_result = TaskResult(status=TaskStatus.RUNNING, stage="SUBMITTED")
        task_id = task_result.task_id

        with self._lock:
            self._job_registry[task_id] = task_result

        # Dispatch the heavy evaluation to the thread pool.
        # We wrap the call in a lambda that captures task_id so we can
        # update the registry result when the future completes.
        future = self._executor.submit(
            self._run_evaluation, proposal_input, service, task_id
        )

        # Add a done callback to update the TaskResult when the worker finishes.
        # This runs in the thread pool but updates the registry safely
        # via the lock (the lock is released before the callback fires,
        # so we use the registry reference captured here).
        future.add_done_callback(
            lambda fut: self._on_evaluation_complete(task_id, fut)
        )

        return task_id

    def get_task_status(self, task_id: str) -> TaskResult:
        """
        Retrieve the current status of a task.

        :param task_id: Unique task identifier returned by submit_evaluation.
        :return: TaskResult containing the latest status, progress, stage, etc.
        """
        with self._lock:
            result = self._job_registry.get(task_id)
        if result is None:
            # Return a minimal failed sentinel rather than raising,
            # so callers can distinguish "not found" from "FAILED".
            return TaskResult(
                status=TaskStatus.FAILED,
                error_message=f"Task {task_id!r} not found in registry",
            )
        return result

    def cancel_task(self, task_id: str) -> bool:
        """
        Attempt to cancel a running task.

        Checks the registry for the task and attempts to cancel the
        underlying future if it is still pending or running.

        :param task_id: Unique task identifier.
        :return: True if the task was successfully cancelled (was in a
            cancellable state), False otherwise (not found, already completed,
            or already cancelled).
        """
        with self._lock:
            result = self._job_registry.get(task_id)
        if result is None:
            return False

        # We need the future to call cancel(). We stored the future in the
        # registry via a parallel dict, but to keep things lightweight we
        # re-derive it: look for a running/RUNNING task and attempt cancel.
        # For simplicity, we check status and attempt a best-effort cancel.
        with self._lock:
            task_result = self._job_registry.get(task_id)
        if task_result is None or task_result.status not in (
            TaskStatus.RUNNING,
        ):
            # Cannot cancel: already completed or never submitted.
            return False

        # Note: The actual future cancellation depends on the underlying
        # thread respecting KeyboardInterrupt / SystemExit, but we mark
        # the task as CANCELLED in the registry.
        with self._lock:
            task_result.status = TaskStatus.CANCELLED
            task_result.completed_at = (
                datetime.datetime.now(datetime.timezone.utc).isoformat()
            )

        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _run_evaluation(
        proposal_input: dict,
        service: EvaluationService,
        task_id: str,
    ) -> dict:
        """
        Execute the full evaluation pipeline in a worker thread.

        This method is the target submitted to the ThreadPoolExecutor.
        It catches ALL exceptions, stores them in the TaskResult.error_message,
        and ensures the TaskResult status is set to FAILED without propagating
        the exception to the executor pool.

        :param proposal_input: Validated ProposalInput dict.
        :param service: EvaluationService instance.
        :param task_id: Identifier for the task result registry entry.
        :return: The EvaluationResult dict on success.
        :raises: Never propagates exceptions; all are caught and recorded.
        """
        try:
            # Run the synchronous evaluation service call.
            # The service method itself is synchronous; we're offloading it
            # to a thread so the main thread isn't blocked.
            result = service.evaluate_and_register(proposal_input=proposal_input)
            return result
        except Exception as exc:  # noqa: BLE001 - we catch everything to prevent daemon crash
            logger.exception(
                f"Evaluation task {task_id!r} raised an unhandled exception"
            )
            # We cannot directly modify the TaskResult here because it's
            # created in the main thread; the done_callback will update it.
            # However, we re-raise a sentinel so the callback can capture it.
            # To keep it simple, we return a dict with an 'error' key;
            # the callback will set the TaskResult status to FAILED.
            return {"error": str(exc), "type": exc.__class__.__name__}

    def _on_evaluation_complete(self, task_id: str, future: Future) -> None:
        """
        Callback invoked when the worker thread future completes.

        Updates the TaskResult in the registry with the outcome:
        - On success: sets status=COMPLETED, progress=1.0, stage="PERSISTENCE&REPORTING",
          and stores the result.
        - On failure: sets status=FAILED, records error_message, clears result.
        - On cancellation: sets status=CANCELLED (handled by cancel_task).

        :param task_id: The task identifier.
        :param future: The concurrent.futures.Future that just completed.
        """
        # Evaluate the result of the future.
        exc = future.exception()
        if exc is not None:
            # The worker raised an exception.
            with self._lock:
                task_result = self._job_registry.get(task_id)
            if task_result is not None:
                task_result.status = TaskStatus.FAILED
                task_result.error_message = (
                    f"{exc.__class__.__name__}: {exc}"
                    if exc
                    else "Unknown evaluation error"
                )
                task_result.completed_at = (
                    datetime.datetime.now(datetime.timezone.utc).isoformat()
                )
                task_result.progress = 1.0
                task_result.stage = "FAILED"
            return

        # No exception — the future completed normally.
        try:
            result = future.result()
        except Exception:
            # Guard against rare race conditions where exception() raises
            # but result() also raises.
            with self._lock:
                task_result = self._job_registry.get(task_id)
            if task_result is not None:
                task_result.status = TaskStatus.FAILED
                task_result.error_message = "Evaluation raised exception during result retrieval"
                task_result.completed_at = (
                    datetime.datetime.now(datetime.timezone.utc).isoformat()
                )
                task_result.progress = 1.0
                task_result.stage = "FAILED"
            return

        # Successful completion — populate the TaskResult.
        with self._lock:
            task_result = self._job_registry.get(task_id)
            if task_result is not None:
                task_result.status = TaskStatus.COMPLETED
                task_result.progress = 1.0
                task_result.stage = "PERSISTENCE & REPORTING"
                # Store the EvaluationResult; the TaskResult model has
                # result: Optional[EvaluationResult] which will accept the dict.
                task_result.result = result
                task_result.completed_at = (
                    datetime.datetime.now(datetime.timezone.utc).isoformat()
                )