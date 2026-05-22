import logging
import sys
import argparse

from lerobot.common.utils.utils import has_method, init_logging, log_say



def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Log and optionally speak text from command line')
    parser.add_argument('text', nargs='*', help='Text to log and say')
    args = parser.parse_args()

    # Handle case where text is provided as multiple arguments
    full_text = ' '.join(args.text)

    # Call the log_say function with command line arguments
    log_say(full_text, True, False)
    input(full_text + " confirm, press enter.")


if __name__ == "__main__":
    main()