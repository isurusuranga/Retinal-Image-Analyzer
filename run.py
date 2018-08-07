import os
from flask import Flask
from application import create_app
from sqlalchemy import create_engine
import pymysql

# To run the project in commandline please set the below environment variables
#set FLASK_CONFIG=development
#set FLASK_APP=run.py
#Then run the application as below
#flask run

config_name = os.getenv('FLASK_CONFIG')
application = create_app(config_name)

if __name__ == '__main__':
    application.run()
