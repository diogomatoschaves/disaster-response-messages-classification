import os

DATABASE_NAME = 'DisasterResponse.db'
DATABASE_DIRECTORY = 'data'
DATABASE_PATH = os.path.join(DATABASE_DIRECTORY, DATABASE_NAME)

TABLE_NAME = 'messages'

MODEL_NAME = 'trained_model.pkl'
MODEL_DIRECTORY = 'models'
MODEL_PATH = os.path.join(MODEL_DIRECTORY, MODEL_NAME)


