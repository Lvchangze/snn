from abc import ABC, abstractmethod
from typing import List, Dict
from textattack.attack_results import SuccessfulAttackResult, FailedAttackResult, SkippedAttackResult
from textattack.attack_results.attack_result import AttackResult

class Metric(ABC):
    def __init__(self, compare_key='-loss'):
        compare_key = compare_key.lower()
        if not compare_key.startswith('-') and compare_key[0].isalnum():
            compare_key = "+{}".format(compare_key)
        self.compare_key = compare_key

    def __str__(self):
        return ', '.join(['{}: {:.4f}'.format(key, value) for (key, value) in self.get_metric().items()])

    @abstractmethod
    def __call__(self, ) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_metric(self, reset: bool = False) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    def __gt__(self, other: "Metric"):
        is_large = self.compare_key.startswith('+')
        key = self.compare_key[1:]
        assert key in self.get_metric()

        if is_large:
            return self.get_metric()[key] > other.get_metric()[key]
        else:
            return self.get_metric()[key] < other.get_metric()[key]

    def __ge__(self, other: "Metric"):
        is_large = self.compare_key.startswith('+')
        key = self.compare_key[1:]
        assert key in self.get_metric()

        if is_large:
            return self.get_metric()[key] >= other.get_metric()[key]
        else:
            return self.get_metric()[key] <= other.get_metric()[key]

class SimplifidResult:
    def __init__(self, ):
        self._succeed = 0
        self._fail = 0
        self._skipped = 0

    def __str__(self):
        return ', '.join(['{}: {:.2f}%'.format(key, value) for (key, value) in self.get_metric().items()])

    def __call__(self, result: AttackResult) -> None:
        assert isinstance(result, AttackResult)
        if isinstance(result, SuccessfulAttackResult):
            self._succeed += 1
        elif isinstance(result, FailedAttackResult):
            self._fail += 1
        elif isinstance(result, SkippedAttackResult):
            self._skipped += 1

    def get_metric(self, reset: bool = False) -> Dict:
        all_numbers = self._succeed + self._fail + self._skipped
        correct_numbers = self._succeed + self._fail

        if correct_numbers == 0:
            success_rate = 0.0
        else:
            success_rate = self._succeed / correct_numbers

        if all_numbers == 0:
            clean_accuracy = 0.0
            robust_accuracy = 0.0
        else:
            clean_accuracy = correct_numbers / all_numbers
            robust_accuracy = self._fail / all_numbers

        if reset:
            self.reset()

        return {"Accu(cln)": clean_accuracy * 100,
                "Accu(rob)": robust_accuracy * 100,
                "Succ": success_rate * 100}

    def reset(self) -> None:
        self._succeed = 0
        self._fail = 0
        self._skipped = 0