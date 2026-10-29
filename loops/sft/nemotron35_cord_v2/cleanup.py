import argparse
import json
import re
from pathlib import Path

from common import api_request, load_env


def cleanup(output, sampler=False):
    filename, key, resource = (
        ("sampler.json", "sampler_id", "samplers")
        if sampler
        else ("loops_run.json", "run_id", "runs")
    )
    record = json.loads((output / filename).read_text())
    identifier = record[key]
    if not isinstance(identifier, str) or not re.fullmatch(
        r"[A-Za-z0-9_-]+", identifier
    ):
        raise ValueError("Refusing an invalid resource identifier.")
    if sampler and record.get("owned_by_this_evaluation") is not True:
        raise ValueError("Sampler record does not assert ownership by this evaluation.")
    path = f"/v1/loops/{resource}/{identifier}"
    api_request(path + "/deactivate", method="POST")
    metadata = api_request(path)
    result = {key: identifier, "requested": True, "status": metadata.get("status")}
    (output / "deactivation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Deactivate only the resource recorded in a recipe output directory."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--training-output", type=Path)
    source.add_argument("--evaluation-output", type=Path)
    parser.add_argument("--env-file")
    parser.add_argument("--yes", action="store_true")
    arguments = parser.parse_args()
    if not arguments.yes:
        parser.error(
            "Read the output's run/sampler record, then pass --yes to confirm deactivation."
        )
    load_env(arguments.env_file)
    output = arguments.training_output or arguments.evaluation_output
    print(
        json.dumps(
            cleanup(output, sampler=arguments.evaluation_output is not None), indent=2
        )
    )


if __name__ == "__main__":
    main()
