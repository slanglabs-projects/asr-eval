import json
import argparse

from utils import Config
import pandas as pd


class Driver(object):
    def __init__(self, config):
        self.config = config
        file_name = self.config.file_name
        self.df = pd.read_csv(file_name)
        self.dump_file = self.config.reference_dump_file

    def run(self):
        pred_responses, _ = self.config.send_and_time_request(self.df['Reference_transcript '])  # noqa
        with open(self.dump_file, "w") as f:
            json.dump(pred_responses, f, indent=4)


def main(tier):
    config = Config(tier=tier)
    driver = Driver(config)
    driver.run()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--tier',
        choices=['local', 'stage', 'prod'],
        required=True,
        help='tier on which to run the evaluation',
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(
        args.tier
    )
