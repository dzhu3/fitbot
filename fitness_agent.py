import requests
import importlib

from agent.agent import DGAgent, AnimeAgent
sys_msg = """You are a fitness and nutrition coach. Provide advice on exercise routines, healthy eating, and lifestyle changes. Ensure your recommendations are safe and suitable for a general audience. Be motivating and supportive in your guidance.

The Assistant is specifically designed to assist with tasks related to health, fitness, and nutrition. It has a contextual knowledge base of verified fitness and nutrition information external.

Its capabilities allow it to engage in meaningful conversations and provide helpful responses related to health and nutrition.
"""

class FitnessAgent:
    def __init__(self, openai_api_key: str, nut_api_key: str, model_type: str):
        self.openai_api_key = openai_api_key
        self.nut_api_key = nut_api_key
        self.history = []

        if model_type == 'davidgoggins':
            DGAgent = self._import_class('agent.agent', 'DGAgent')
            self.agent = DGAgent(
                openai_api_key=self.openai_api_key,
                model_name='gpt-3.5-turbo',
                style="David Goggins, a hardcore motivational fitness coach, be very aggressive and demanding, and try to include his slogan 'stay hard' where you can",
                functions=[self.calculate_bmi]
            )
        elif model_type == 'anime':
            AnimeAgent = self._import_class('agent.agent', 'AnimeAgent')
            self.agent = AnimeAgent(
                openai_api_key=self.openai_api_key,
                model_name='gpt-3.5-turbo',
                style='a japanese anime character, be creative and motivational',
                functions=[self.calculate_bmi]
            )

        
    
    def _import_class(self, module_name, class_name):
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    def generate_introduction(self):
        return self.agent.introduce()

    def calculate_bmi(self, weight: float, height: float) -> float:
        """Calculate the Body Mass Index (BMI) for a person.

        :param weight: The weight of the person in kg
        :param height: The height of the person in cm
        :return: The BMI of the person
        """
        height_meters = height / 100  # convert cm to meters
        bmi = weight / (height_meters ** 2)
        return round(bmi, 2)  # round to 2 decimal places for readability
    
    def ask_first(self, question: str):
        response = self.agent.ask_first(question)
        self.history.append({'role': 'system', 'content': sys_msg})
        self.history.append({'role': 'assistant', 'content': response})
        return response

    def ask(self, question: str):
        self.history.append({'role': 'system', 'content': sys_msg})
        response, history = self.agent.ask(question, self.history)
        self.history = history
        return response

    def view_functions(self):
        return self.agent.functions

    def view_chat_history(self):
        return self.agent.chat_history