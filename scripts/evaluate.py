"""Script to evaluate a fine-tuned CoT checkpoint."""
import argparse
import yaml


def main(config_path: str, checkpoint: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    # TODO: load checkpoint and run evaluation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    main(args.config, args.checkpoint)
