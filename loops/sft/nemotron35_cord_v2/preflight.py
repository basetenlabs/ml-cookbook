import argparse
import json

from common import load_env, prepare, settings, validate_catalog


def main():
    parser = argparse.ArgumentParser(
        description="Audit all CORD-v2 inputs without provisioning a trainer."
    )
    parser.add_argument("--env-file")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--max-seq-len", type=int)
    arguments = parser.parse_args()
    load_env(arguments.env_file)
    config = settings()
    if arguments.max_seq_len is not None:
        if arguments.max_seq_len < 1:
            parser.error("--max-seq-len must be positive.")
        config["max_seq_len"] = arguments.max_seq_len
    capability = None
    if not arguments.offline:
        capability = validate_catalog(config)
    _, audit = prepare(config)
    print(
        json.dumps(
            {
                "audit": audit,
                "capability": capability,
                "managed_settings": config["managed_settings"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
