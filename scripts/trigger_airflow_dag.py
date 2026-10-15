"""Trigger an Airflow DAG with a JSON config file without shell quoting issues."""

import argparse
import json
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description="Trigger an Airflow DAG from JSON config.")
    parser.add_argument("dag_id")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, encoding="utf-8") as config_file:
        config = json.load(config_file)

    command = ["airflow", "dags", "trigger", args.dag_id, "--conf", json.dumps(config)]
    if args.run_id:
        command.extend(["--run-id", args.run_id])
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
