"""Async utilities for concurrent LLM calls."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)


class AsyncExecutor:
    """Manages async execution of LLM calls with concurrency limits."""

    def __init__(self, max_concurrent: int = 8, use_threads: bool = False):
        """
        Initialize async executor.

        Args:
            max_concurrent: Maximum concurrent operations
            use_threads: If True, use ThreadPoolExecutor; else use asyncio
        """
        self.max_concurrent = max_concurrent
        self.use_threads = use_threads
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self._thread_executor: Optional[ThreadPoolExecutor] = None

    async def run_sync(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """
        Run a synchronous function with concurrency control.

        Args:
            func: Synchronous function to run
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Function result
        """
        if self.use_threads:
            # Run in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._get_thread_executor(), lambda: func(*args, **kwargs)
            )
        else:
            # In asyncio mode, run sync function in a separate thread to avoid blocking
            if not hasattr(self, "_sync_executor"):
                self._sync_executor = ThreadPoolExecutor(max_workers=self.max_concurrent)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._sync_executor, lambda: func(*args, **kwargs))

    async def map_async(
        self, func: Callable[..., Any], items: List[Any], *args, **kwargs
    ) -> List[Any]:
        """
        Apply function to all items concurrently with limits.

        Args:
            func: Function to apply (can be sync or async)
            items: List of items to process
            *args: Additional positional args for func
            **kwargs: Additional keyword args for func

        Returns:
            List of results in order corresponding to items
        """
        if self.use_threads:
            # For sync functions, use ThreadPoolExecutor directly
            return await self._map_threadpool(func, items, *args, **kwargs)
        else:
            # For async functions, use asyncio.gather with semaphore
            return await self._map_asyncio(func, items, *args, **kwargs)

    async def _map_asyncio(
        self, func: Callable[..., Any], items: List[Any], *args, **kwargs
    ) -> List[Any]:
        """Map using asyncio.gather with semaphore protection."""
        tasks = [self._run_with_semaphore(func, item, *args, **kwargs) for item in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # Handle exceptions
        processed = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {i} failed: {result}")
                processed.append(None)
            else:
                processed.append(result)
        return processed

    async def _run_with_semaphore(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """Run function with semaphore protection."""
        async with self.semaphore:
            return await func(*args, **kwargs)

    async def _map_threadpool(
        self, func: Callable[..., Any], items: List[Any], *args, **kwargs
    ) -> List[Any]:
        """Map using ThreadPoolExecutor with limited workers."""
        import asyncio

        if not self._thread_executor:
            self._thread_executor = ThreadPoolExecutor(max_workers=self.max_concurrent)

        loop = asyncio.get_event_loop()
        tasks = []
        for item in items:
            # Check if func is a coroutine function (async)
            if asyncio.iscoroutinefunction(func):
                # Wrap to run async function in its own event loop within the thread
                def wrapper(item=item):
                    return asyncio.run(func(item, *args, **kwargs))

                task = loop.run_in_executor(self._thread_executor, wrapper)
            else:
                task = loop.run_in_executor(
                    self._thread_executor, lambda: func(item, *args, **kwargs)
                )
            tasks.append(task)
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        processed = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Thread task failed: {result}")
                processed.append(None)
            else:
                processed.append(result)
        return processed

    def _get_thread_executor(self) -> ThreadPoolExecutor:
        """Get or create thread executor."""
        if self._thread_executor is None:
            self._thread_executor = ThreadPoolExecutor(max_workers=self.max_concurrent)
        return self._thread_executor

    def shutdown(self) -> None:
        """Clean up resources."""
        if self._thread_executor:
            self._thread_executor.shutdown(wait=True)
            self._thread_executor = None
        if hasattr(self, "_sync_executor") and self._sync_executor:
            self._sync_executor.shutdown(wait=True)
            self._sync_executor = None


async def gather_with_concurrency(n: int, tasks: List[Callable[[], Any]]) -> List[Any]:
    """
    Run coroutines with limited concurrency.

    Args:
        n: Maximum number of concurrent tasks
        tasks: List of no-arg async callables

    Returns:
        List of results in order
    """
    semaphore = asyncio.Semaphore(n)

    async def limited_task(task: Callable[[], Any]) -> Any:
        async with semaphore:
            return await task()

    return await asyncio.gather(*[limited_task(task) for task in tasks], return_exceptions=True)


def run_parallel(
    func: Callable[..., Any], items: List[Any], max_workers: int = 8, use_threads: bool = True
) -> List[Any]:
    """
    Simple synchronous parallel execution.

    Args:
        func: Function to apply to each item
        items: List of items to process
        max_workers: Number of worker threads
        use_threads: Use threads if True, else multiprocessing

    Returns:
        List of results
    """
    if use_threads:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(func, items))
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(func, items))
