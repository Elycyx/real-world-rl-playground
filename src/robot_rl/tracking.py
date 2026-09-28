"""Send Actor and Learner measurements to one W&B training run."""

import threading
from numbers import Real


def wandb_values(event: dict) -> dict[str, int | float]:
    if event["role"] == "learner":
        prefix, axis = "learner", "learner_updates"
        values = {"learner/environment_step": event["step"]}
    else:
        prefix, axis = ("timing" if event.get("event") == "timing" else "episode"), "environment_step"
        values = {}
    values[axis] = event["updates" if prefix == "learner" else "step"]
    for key, value in event.items():
        if key not in ("role", "pid", "session_id", "step", "updates", "event") and isinstance(value, Real):
            values[f"{prefix}/{key}"] = float(value)
    return values


class WandbTracker:
    def __init__(self, args, config: dict, queue):
        import wandb

        self.queue = queue
        self.error = None
        self.wandb_error = getattr(wandb, "Error", RuntimeError)
        self.run = wandb.init(project=args.wandb_project, mode=args.wandb_mode,
                              id=args.session_id, name=f"{args.task}-{args.algorithm}-{args.session_id[:8]}",
                              group=f"{args.task}-{args.algorithm}", job_type="train",
                              dir=str(args.output.resolve()), config=config)
        for prefix, axis in (("episode", "environment_step"), ("timing", "environment_step"),
                             ("learner", "learner_updates")):
            self.run.define_metric(f"{prefix}/*", step_metric=axis)
        self.thread = threading.Thread(target=self._consume, name="wandb-metrics", daemon=True)
        self.thread.start()

    def _consume(self) -> None:
        while True:
            event = self.queue.get()
            if event is None:
                break
            if self.error is None:
                try:
                    self.run.log(wandb_values(event))
                except (RuntimeError, OSError, ValueError, self.wandb_error) as error:
                    self.error = error

    def finish(self, success: bool) -> None:
        self.queue.put(None)
        self.thread.join()
        try:
            self.run.finish(exit_code=0 if success and self.error is None else 1)
        finally:
            if self.error is not None:
                raise RuntimeError("W&B failed to record training metrics") from self.error
