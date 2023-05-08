# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from hydra_zen import make_config, MISSING, instantiate, just, launch
import pytorch_lightning as pl
from typing import Callable, List
from omegaconf import DictConfig
import submitit
import time

SlurmConfig = make_config(
    executor_dir=MISSING,
    slurm_partition=MISSING,
    slurm_gres=MISSING,
    slurm_constraint=MISSING,
    timeout_min=5,
    multirun=False,
)


class PowerSweep:
    def __init__(
        self,
        task_cfg: DictConfig,
        task_fn: Callable[[DictConfig], pl.LightningModule],
        slurm_cfg: DictConfig,
        gpower: List[int] = None,
        multirun: bool = False,
    ):

        self.task_cfg = task_cfg
        self.task_fn = task_fn
        self.slurm_cfg = slurm_cfg
        self.multirun = multirun

        self.gpower = None
        if gpower is not None:
            self.gpower = "+gpower=" + ",".join([str(g) for g in gpower])

    def wrap_task(self, cfg: DictConfig):
        executor = submitit.AutoExecutor(folder=self.slurm_cfg.executor_dir)

        if self.gpower is not None:
            executor.update_parameters(
                timeout_min=self.slurm_cfg.timeout_min,
                slurm_partition=self.slurm_cfg.slurm_partition,
                slurm_gres=self.slurm_cfg.slurm_gres,
                slurm_constraint=self.slurm_cfg.slurm_constraint,
                slurm_additional_parameters=dict(gpupower=cfg.gpower),
            )
        else:
            executor.update_parameters(
                timeout_min=self.slurm_cfg.timeout_min,
                slurm_partition=self.slurm_cfg.slurm_partition,
                slurm_gres=self.slurm_cfg.slurm_gres,
                slurm_constraint=self.slurm_cfg.slurm_constraint,
            )

        job = executor.submit(self.task_fn, cfg)

        return job

    def launch(self, overrides: List[str] = None):
        if self.gpower is not None:
            if overrides is not None:
                overrides.append(self.gpower)
            else:
                overrides = [
                    self.gpower,
                ]

        job = launch(
            self.task_cfg,
            task_function=self.wrap_task,
            multirun=self.multirun,
            to_dictconfig=True,
            overrides=overrides,
        )

        if self.multirun:
            while True:
                states = [j.return_value.state for j in job[0]]
                print(f"Job states: {states}")

                complete = [s == "COMPLETED" for s in states]
                failed = [s == "FAILED" for s in states]

                if all(complete):
                    print("All jobs completed!")
                    break
                if any(failed):
                    print("At least one job failed.")
                    break

                time.sleep(5)

        else:
            while True:
                state = job.return_value.state
                print(f"Job state: {state}")

                complete = state == "COMPLETED"
                failed = state == "FAILED"

                if complete:
                    print("Job complete!")
                    break
                if failed:
                    print("Job failed.")
                    break

                time.sleep(5)

        return job
