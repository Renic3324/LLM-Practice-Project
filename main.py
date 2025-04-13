import runpy
import os

#runpy.run_path('Parquet_Reader.py')

file_path = 'C:/Users/Owner/PycharmProjects/LLMProject9/model-01.pkl'
file_path2 = 'C:/Users/Owner/PycharmProjects/LLMProject9/model_updated.pkl'

if os.path.isfile(file_path):
    print("Model already exists, skipping training.")
else:
    runpy.run_path('Training.py')

if os.path.isfile(file_path):
    print("Updated model already exists, skipping extra training.")
else:
    runpy.run_path('Training2.py')

runpy.run_path('Chat_Bot.py')
