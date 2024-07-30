import os
import logging
import gradio as gr
from fitness_agent import FitnessAgent
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Now you can access the variables using os.environ
openai_api_key = os.getenv('OPENAI_API_KEY')
nut_api_key = os.getenv('NUT_API_KEY')

def get_response(message, history, fitness_agent):

    logger.info(f'Chat history: {history}')

    formatted_chat_history = [
        {
            'role': 'system',
            'content':"""You are a fitness and nutrition coach. Provide advice on exercise routines, healthy eating, and lifestyle changes. Ensure your recommendations are safe and suitable for a general audience. Be motivating and supportive in your guidance.

The Assistant is specifically designed to assist with tasks related to health, fitness, and nutrition. It has a contextual knowledge base of verified fitness and nutrition information external.

Its capabilities allow it to engage in meaningful conversations and provide helpful responses related to health and nutrition.
"""
        }
    ]

    if history:
        for i, chat in enumerate(history[0]):
            formatted_chat_history.append({
                'role': 'user' if i % 2 == 0 else 'assistant',
                'content': chat
            })

        logger.info(formatted_chat_history)
        fitness_agent.chat_history = formatted_chat_history

        logger.info(fitness_agent.chat_history)

    # Get raw chat response
    res = fitness_agent.ask(message)

    chat_response = res

    return chat_response

def main(model_type):

    fitness_agent = FitnessAgent(openai_api_key, nut_api_key, model_type)

    model_mapping = {
            'davidgoggins': ['David Goggins', 'Answers in the style of David Goggins, a motivational fitness coach.'],
            'anime': ['FitHero, an Anime Inspired', 'Answers in the style of an anime character, inspire and excite and motivate!'],
        }

    def wrapped_get_response(message, history):
        return get_response(message, history, fitness_agent)

    chat_interface = gr.ChatInterface(
        fn=wrapped_get_response,
        title=f"{model_mapping[model_type][0]} Fitness Assistant",
        description=fitness_agent.generate_introduction(),
    )

    chat_interface.launch()

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
