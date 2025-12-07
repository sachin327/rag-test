from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class ThreadPoolManager:
    """
    A simple wrapper around ThreadPoolExecutor to manage parallel execution of tasks.
    """

    def __init__(self, max_workers: int = 4):
        """
        Initialize the ThreadPoolManager.

        Args:
            max_workers: The maximum number of threads to use. Defaults to 4.
        """
        self.max_workers = max_workers

    def execute(self, func: Callable[[T], R], items: List[T]) -> List[R]:
        """
        Execute a function in parallel for each item in the list.

        Args:
            func: The function to execute for each item.
            items: The list of items to process.

        Returns:
            A list of results in the same order as the input items.
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(func, items))
        return results
