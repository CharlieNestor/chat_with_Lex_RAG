import os
from typing import Dict, Any
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

### Basic functions to connect to MongoDB

def get_mongodb_uri() -> str:
    """
    Construct MongoDB URI from environment variables or use a predefined URI
    """
    # If MONGO_URI is set in the environment, use it directly
    uri = os.getenv('MONGO_URI')
    if uri:
        return uri
    
    # Otherwise, construct the URI from individual components
    host = os.getenv('MONGO_HOST', 'localhost')
    port = os.getenv('MONGO_PORT', '27017')
    username = os.getenv('MONGO_USERNAME', 'admin')
    password = os.getenv('MONGO_PASSWORD', 'password')
    
    return f"mongodb://{username}:{password}@{host}:{port}/"


def connect_to_mongodb() -> MongoClient:
    """
    Connect to MongoDB using the URI
    """
    try:
        uri = get_mongodb_uri()
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ismaster')
        return client
    except ConnectionFailure as e:
        print(f"Failed to connect to MongoDB: {e}")
        return None
    

def check_connection(client: MongoClient, verbose: bool = False):
    """
    Check if the connection is still alive
    """
    try:
        client.admin.command('ismaster')
        if verbose:
            print("Connection to MongoDB is running")
        return True
    except ConnectionFailure as e:
        if verbose:
            print(f"Connection to MongoDB lost: {e}")
        return False

def close_connection(client: MongoClient):
    """
    Close the connection to MongoDB
    """
    client.close()

def get_database(client: MongoClient, database_name: str):
    """
    Get the database
    """
    return client[database_name]

def get_collection(database, collection_name: str):
    """
    Get the collection
    """
    return database[collection_name]


### Functions to create database and collections


def createDB_from_data(
    client: MongoClient,
    database_name: str,
    collection_name: str,
    initial_document: Dict[str, Any]
) -> MongoClient:
    """
    Create a database by inserting an initial document into a collection.
    :param client: MongoDB client
    :param database_name: Name of the database to create
    :param collection_name: Name of the collection to create
    :param initial_document: The first document to insert
    :return: The created database
    """
    # Check if the database already exists
    if database_name in client.list_database_names():
        print(f"Database '{database_name}' already exists.")
        db = client[database_name]
        
        # Check if the collection already exists
        if collection_name in db.list_collection_names():
            print(f"Collection '{collection_name}' already exists in database '{database_name}'.")
            return None
    else:
        db = client[database_name]

    # Insert the initial document into the specified collection. This actually creates the collection.
    result = db[collection_name].insert_one(initial_document)

    if result.acknowledged:
        print(f"Database '{database_name}' and collection '{collection_name}' created successfully.")
        print(f"Inserted document with ID: {result.inserted_id}")
    else:
        raise Exception("Failed to insert document and create database/collection.")

    return db

