import argparse
import chatbot

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Fitness Assistant LLM Selector")
    parser.add_argument('--llm', type=str, required=False, choices=['davidgoggins', 'anime', 'ali', 'gpt4'], default='davidgoggins', help="Specify which LLM to use")
    args = parser.parse_args()

    chatbot.main(args.llm)