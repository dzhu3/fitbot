import logging
from typing import Optional
from agent.parser import func_to_json
import random

from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys_msg = """You are a fitness and nutrition coach. Provide advice on exercise routines, healthy eating, and lifestyle changes. Ensure your recommendations are safe and suitable for a general audience. Be motivating and supportive in your guidance.

The Assistant is specifically designed to assist with tasks related to health, fitness, and nutrition. It has a contextual knowledge base of verified fitness and nutrition information external.

Its capabilities allow it to engage in meaningful conversations and provide helpful responses related to health and nutrition.
"""

urls = [
    'https://www.webmd.com/fitness-exercise/ss/slideshow-fit-overweight',
    'https://www.mayoclinic.org/healthy-lifestyle/weight-loss/in-depth/exercise/art-20050999',
    'https://www.healthline.com/health/fitness/explosive-workouts#tips-and-considerations',
    'https://www.healthline.com/nutrition/workout-routine-for-men#home-exercises',
    'https://www.muscleandfitness.com/routine/workouts/workout-routines/build-brute-strength-workout/',
    'https://www.helpguide.org/articles/healthy-living/what-are-the-best-exercises-for-me.htm',
    'https://dr-muscle.com/bodybuilding-meal-plan/',
    'https://www.healthline.com/nutrition/bodybuilding-meal-plan#supplements'    
]

loaders = UnstructuredURLLoader(urls=urls)
data = loaders.load()

text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1500, chunk_overlap=200)
docs = text_splitter.split_documents(data)

questions = "What are your short term goals to conquer?, How often are you putting in the work?, What fuel are you putting into your body?, Are you hitting it hard in the morning, or are you more of an evening grinder?, Any specific areas you're ready to attack?, How many hours are you clocking in for rest each night?, How much water are you pushing through your system daily?, What keeps you locked in and pushing forward?, Which exercises get you fired up, and which ones are you avoiding?, How are you keeping track of your grind?, What challenges are you battling right now?, Any injuries or setbacks I need to know about?, Are you currently working with a doctor?, How do you handle stress and recovery after the battle?"

class DGAgent:
    def __init__(
        self,
        openai_api_key: str,
        model_name: str,
        style: str,
        functions: Optional[list] = None,
    ):
        self.urls=urls.copy()
        self.urls.append('https://www.pinkvilla.com/health/fitness/david-goggins-workout-routine-1237721') #append custom links pertaining to style


        loaders = UnstructuredURLLoader(urls=self.urls)
        data = loaders.load()

        text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1500, chunk_overlap=200)
        docs = text_splitter.split_documents(data)

        self.openai_api_key = openai_api_key
        self.model_name = model_name
        self.functions = self._parse_functions(functions)
        self.func_mapping = self._create_func_mapping(functions)
        self.chat_history = [{'role': 'system', 'content': sys_msg}]  #change the content to adjust to user
        self.style = style
        
        embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        self.vectorstore = FAISS.from_documents(docs, embeddings)
        self.retriever = self.vectorstore.as_retriever()
        
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name=self.model_name,
            openai_api_key=self.openai_api_key
        )
    
    def _convert_chat_history_to_string(self) -> str:
        return "\n".join([f"{entry['role']}: {entry['content']}" for entry in self.chat_history])
    
    def ask_first(self, query: str) -> str:

        template = f"""You are a fitness assistant for question-answering tasks.
            Answer in the style of {self.style}. Introduce yourself. Make sure to use line spaces to break up the response.
            Ask for the user's height, age, weight, and gender to help curate a personalized fitness routine.
            """
        prompt = ChatPromptTemplate.from_template(template)
        rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        response = rag_chain.invoke(query)
        return response



    def ask(self, query: str, history) -> str:
        history.append({'role': 'assistant', 'content': query})

        def _convert_chat_history_to_string(history2) -> str:
            return "\n".join([f"{entry['role']}: {entry['content']}" for entry in history2])
        
        chat_history_str = _convert_chat_history_to_string(history)


        template = f"""You are a fitness assistant for question-answering tasks. Do not introduce yourself again.
            Use the following pieces of retrieved context to answer the question. Be detailed in your response.
            Answer in the style of {self.style}. Follow the conversation flow using the chat history: {chat_history_str}, make sure to use line spaces to break up response. Do not repeat responses.
            If the response does not require a table, provide a textual response in the specified style.
            If the response requires a table (e.g., weekly exercise routine), include it in HTML format, and be detailed in the response. For example:
            <table>
                <tr>
                    <th>Day</th>
                    <th>Exercise</th>
                    <th>Duration</th>
                </tr>
                <tr>
                    <td>Monday</td>
                    <td>Sample exercise</td>
                    <td>30 mins</td>
                </tr>
                <tr>
                    <td>Tuesday</td>
                    <td>Sample exercise</td>
                    <td>45 mins</td>
                </tr>
                <tr>
                    <td>Wednesday</td>
                    <td>Sample exercise</td>
                    <td>60 mins</td>
                </tr>
                <tr>
                    <td>Thursday</td>
                    <td>Sample exercise</td>
                    <td>30 mins</td>
                </tr>
                <tr>
                    <td>Friday</td>
                    <td>Sample exercise</td>
                    <td>40 mins</td>
                </tr>
                <tr>
                    <td>Saturday</td>
                    <td>Sample exercise</td>
                    <td>50 mins</td>
                </tr>
                <tr>
                    <td>Sunday</td>
                    <td>Sample exercise</td>
                    <td>50 mins</td>
                </tr>
            </table>
            Make sure you only include one question to end with. 
            Make sure you only have 1 question at the end. Make sure to end your response with a question that encourages the user to share more about their preferences, experiences, or thoughts. Some questions you can ask are: What are your short term goals to conquer?, How often are you putting in the work?, What fuel are you putting into your body?, Are you hitting it hard in the morning, or are you more of an evening grinder?, Any specific areas you're ready to attack?, How many hours are you clocking in for rest each night?, How much water are you pushing through your system daily?, What keeps you locked in and pushing forward?, Which exercises get you fired up, and which ones are you avoiding?, How are you keeping track of your grind?, What challenges are you battling right now?, Any injuries or setbacks I need to know about?, How do you handle stress and recovery after the battle?. Do not repeat a question that has already been asked. If the response or question is too conclusive, end with "Is there anything else I can help with?" make sure only one question is asked.
            Question: {{question}}
            Context: {{context}}
            Answer:
            """
        prompt = ChatPromptTemplate.from_template(template)
        rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        response = rag_chain.invoke(query)
        history.append({'role': 'assistant', 'content': response})
        return response, history
    
    def _parse_functions(self, functions: Optional[list]) -> Optional[list]:
        if functions is None:
            return None
        return [func_to_json(func) for func in functions]

    def _create_func_mapping(self, functions: Optional[list]) -> dict:
        if functions is None:
            return {}
        return {func.__name__: func for func in functions}
    
    def introduce(self):

        responses =["I’m your personal fitness assistant for answering fitness and nutrition-based questions, here to help you get that win every single time. I believe in the training economy and density. It’s time to put in the work and get your results, and get out. How can I help you achieve victory?",
                    "I am a motivational fitness assistant capable of answering any and all questions concerning fitness and nutrition. I don't stop when I'm tired, I stop when I'm done. How can I serve you?",
                    "I'm David Goggins, your no-excuses, high-intensity fitness guide, here to push you beyond your limits and crush your goals. Ready to get after it? Tell me what you need!",
                    "I'm David Goggins, your no-excuses, hard-hitting personal fitness assistant here to push you beyond your limits and crush your goals. What do you need to get after today?",
                    ]
        
        self.chat_history.append({'role': 'assistant', 'content': responses})
        
        return random.choice(responses)
    

class AnimeAgent:
    def __init__(
        self,
        openai_api_key: str,
        model_name: str,
        style,
        functions: Optional[list] = None,
    ):        
        self.urls=urls.copy()
        self.urls.append('https://jimmyjrichard.com/anime-workouts') #append custom links pertaining to style

        self.openai_api_key = openai_api_key
        self.model_name = model_name
        self.functions = self._parse_functions(functions)
        self.func_mapping = self._create_func_mapping(functions)
        self.chat_history = [{'role': 'system', 'content': sys_msg}]  #change the content to adjust to user
        self.style = style
        
        embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        self.vectorstore = FAISS.from_documents(docs, embeddings)
        self.retriever = self.vectorstore.as_retriever()

        
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name=self.model_name,
            openai_api_key=self.openai_api_key
        )
    
    def _convert_chat_history_to_string(self) -> str:
        return "\n".join([f"{entry['role']}: {entry['content']}" for entry in self.chat_history])


    def ask(self, query: str) -> str:
        self.chat_history.append({'role': 'user', 'content': query})

        chat_history_str = self._convert_chat_history_to_string()


        template = f"""You are a fitness assistant for question-answering tasks.
                Use the following pieces of retrieved context to answer the question.
                Answer in the style of {self.style}. Be detailed in your response. Follow the conversation flow using the chat history: {chat_history_str}, and if it is not known, ask for the user's weight, height and gender.
                Question: {{question}}
                Context: {{context}}
                Answer:
                """
        prompt = ChatPromptTemplate.from_template(template)
        rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        response = rag_chain.invoke(query)
        self.chat_history.append({'role': 'assistant', 'content': response})
        return response
    
    def _parse_functions(self, functions: Optional[list]) -> Optional[list]:
        if functions is None:
            return None
        return [func_to_json(func) for func in functions]

    def _create_func_mapping(self, functions: Optional[list]) -> dict:
        if functions is None:
            return {}
        return {func.__name__: func for func in functions}
    
    def introduce(self):

        responses =["I'm FitHero, your anime-inspired fitness companion, ready to power you up and help you conquer your fitness goals with the spirit of a true hero! What challenge are we tackling today?",
                    "Welcome to FitHero! I'm your anime-inspired fitness guide, here to boost your workouts and help you smash your goals with epic energy. Ready to level up?",
                    "Welcome to FitHero, your ultimate anime fitness partner! Let's power up your workouts and conquer your goals with unstoppable spirit. What can we achieve today?",
                    "Hey there! I'm FitHero, your vibrant anime fitness guide, here to inspire and push you towards your ultimate potential. Ready to become a fitness legend?",
                    ]
        
        self.chat_history.append({'role': 'assistant', 'content': responses})
        
        return random.choice(responses)