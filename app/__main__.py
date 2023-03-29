from app.run.trainer import Trainer
from app.utils import tools


def main():
    args = tools.arg_parse()

    runner = None
    if args.mode == "train" and args.checkpoint is None:
        runner = Trainer(args.dataset_root, args.output_dir)
    elif args.mode == "train":
        runner = Trainer.load(
            args.checkpoint, args.dataset_root, args.output_dir)

    assert runner is not None, f"Sorry, mode:[{args.mode}] is under development"

    runner.run()


if __name__ == "__main__":
    main()
