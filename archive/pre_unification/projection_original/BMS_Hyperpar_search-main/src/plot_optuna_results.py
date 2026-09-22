from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import optuna


def load_trials(storage_url: str, study_name: str):
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    completed_trials = [trial for trial in study.trials if trial.value is not None]
    completed_trials.sort(key=lambda trial: trial.number)
    return study, completed_trials


def plot_trials(trials, output_path: Path, study_name: str):
    if not trials:
        raise RuntimeError(f"No completed trials were found for study '{study_name}'.")

    trial_numbers = [trial.number for trial in trials]
    trial_values = [trial.value for trial in trials]

    best_so_far = []
    current_best = None
    for value in trial_values:
        current_best = value if current_best is None else max(current_best, value)
        best_so_far.append(current_best)

    plt.figure(figsize=(10, 6))
    plt.plot(trial_numbers, trial_values, marker="o", linewidth=1.5, label="Trial value")
    plt.plot(trial_numbers, best_so_far, marker="s", linewidth=2.0, label="Best so far")
    plt.title(f"Optuna Study: {study_name}")
    plt.xlabel("Trial number")
    plt.ylabel("Objective value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.show()


def main():
    parser = ArgumentParser()
    parser.add_argument("--study_name", type=str, default="projection_reward")
    parser.add_argument("--db_path", type=Path, default=Path("optuna.db"))
    parser.add_argument("--output", type=Path, default=Path("plots") / "projection_reward_optuna.png")
    args = parser.parse_args()

    storage_url = f"sqlite:///{args.db_path.resolve()}"
    study, completed_trials = load_trials(storage_url, args.study_name)
    plot_trials(completed_trials, args.output, study.study_name)

    print(f"Saved plot to: {args.output.resolve()}")
    print(f"Completed trials plotted: {len(completed_trials)}")


if __name__ == "__main__":
    main()