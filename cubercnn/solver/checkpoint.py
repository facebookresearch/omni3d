# Copyright (c) Meta Platforms, Inc. and affiliates
from detectron2.checkpoint import PeriodicCheckpointer
from typing import Any

class PeriodicCheckpointerOnlyOne(PeriodicCheckpointer):
    def step(self, iteration: int, **kwargs: Any) -> None:
        """
        Perform the appropriate action at the given iteration.

        Args:
            iteration (int): the current iteration, ranged in [0, max_iter-1].
            kwargs (Any): extra data to save, same as in
                :meth:`Checkpointer.save`.
        """
        iteration = int(iteration)
        additional_state = {"iteration": iteration}
        additional_state.update(kwargs)

        if (iteration + 1) % self.period == 0:
            
            # simply save a single recent model
            self.checkpointer.save(
                "{}_recent".format(self.file_prefix), **additional_state
            )

        if self.max_iter is not None:
            if iteration >= self.max_iter - 1:
                self.checkpointer.save(f"{self.file_prefix}_final", **additional_state)