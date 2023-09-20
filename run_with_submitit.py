# Copyright 2023 by Ismail Khalfaoui-Hassani, ANITI Toulouse.
#
# All rights reserved.
#
# This file is part of the Dcls-Audio package, and
# is released under the "MIT License Agreement".
# Please see the LICENSE file that should have been included as part
# of this package.


import argparse
import os
import uuid
from pathlib import Path

import submitit

import main as classification


def parse_args():
    classification_parser = classification.get_args_parser()
    parser = argparse.ArgumentParser(
        "Submitit for DCLS-Audio", parents=[classification_parser]
    )
    parser.add_argument(
        "--ngpus", default=8, type=int, help="Number of gpus to request on each node"
    )
    parser.add_argument(
        "--nodes", default=2, type=int, help="Number of nodes to request"
    )
    parser.add_argument(
        "--timeout", default=72, type=int, help="Duration of the job, in hours"
    )
    parser.add_argument("--job_name", default="DCLS-Audio", type=str, help="Job name")
    parser.add_argument(
        "--job_dir", default="", type=str, help="Job directory; leave empty for default"
    )
    parser.add_argument(
        "--partition", default="gpu_p2", type=str, help="Partition where to submit"
    )
    parser.add_argument(
        "--use_volta32", action="store_true", default=False, help="Big models? Use this"
    )
    parser.add_argument("--account", default="owj@v100", type=str, help="Account name")
    parser.add_argument(
        "--constraint", default="v100", type=str, help="constraint v100 or a100"
    )
    parser.add_argument(
        "--comment",
        default="",
        type=str,
        help="Comment to pass to scheduler, e.g. priority message",
    )
    return parser.parse_args()


def get_shared_folder() -> Path:
    work = os.getenv("WORK")
    if Path(f"{work}/Dcls-Audio/checkpoint").is_dir():
        p = Path(f"{work}/Dcls-Audio/checkpoint")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import main as classification

        self._setup_gpu_args()
        classification.main(self.args)

    def checkpoint(self):
        import os

        import submitit

        self.args.dist_url = get_init_file().as_uri()
        self.args.auto_resume = True
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        from pathlib import Path

        import submitit

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(self.args.job_dir)
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main():
    args = parse_args()

    if args.job_dir == "":
        args.job_dir = get_shared_folder() / "%j"

    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout * 60

    partition = args.partition
    kwargs = {}
    if args.use_volta32:
        kwargs["slurm_constraint"] = "v100-32g"
    if args.comment:
        kwargs["slurm_comment"] = args.comment

    kwargs["slurm_constraint"] = args.constraint

    executor.update_parameters(
        # mem_gb=40 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=args.num_workers,
        slurm_account=args.account,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        **kwargs,
    )

    executor.update_parameters(name=args.job_name)

    args.dist_url = get_init_file().as_uri()
    args.output_dir = args.job_dir

    trainer = Trainer(args)
    job = executor.submit(trainer)

    print("Submitted job_id:", job.job_id)


if __name__ == "__main__":
    main()
