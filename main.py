import runpy
import os

file_path = 'C:/Users/Owner/PycharmProjects/LLMProject9/model-01.pkl'

if os.path.isfile(file_path):
    print("Model already exists, skipping training.")
else:
    runpy.run_path('Training.py')

runpy.run_path('Chat_Bot.py')
