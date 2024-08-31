import os
from typing import Dict, Any, List, Union
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
    

def check_connection(client: MongoClient, verbose: bool = False) -> bool:
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


def get_documents(collection, limit: int = None, exclude_id: bool = True, output_format: str = 'dict') -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Get all documents from a collection
    :param collection: MongoDB collection object
    :param limit: Optional limit on the number of documents to return
    :param exclude_id: Optional flag to exclude the '_id' field from the documents
    :param output_format: Optional format of the output documents. Can be 'dict' or 'list'. Default is 'dict'.
    :return: A dictionary or list of documents from the collection
    """
    # Obtain the documents with or without the _id field as specified
    if exclude_id:
        documents = collection.find({}, {'_id': 0})
    else:
        documents = collection.find({})
    # Limit the number of documents returned
    if limit:
        documents = documents.limit(limit)
        
    # Return the documents in the specified format
    if output_format == 'dict':
        if 'video_id' in documents[0]:
            return {doc['video_id']: doc for doc in documents}
        else:
            return {doc for doc in documents}
    elif output_format == 'list':
        return [doc for doc in documents]
    else:
        raise ValueError(f"Invalid output format: {output_format}. Please specify 'dict' or 'list'.")


### Functions to create database and collections

def createDB_from_data(
    client: MongoClient,
    database_name: str,
    collection_name: str,
    initial_document: Dict[str, Any],
    custom_id: Any = None
) -> MongoClient:
    """
    Create a database by inserting an initial document into a collection.
    :param client: MongoDB client
    :param database_name: Name of the database to create
    :param collection_name: Name of the collection to create
    :param initial_document: The first document to insert
    :param custom_id: Optional custom ID to use for the document
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

    # Prepare the document with custom ID if provided
    if custom_id is not None:
        initial_document['_id'] = custom_id

    # Insert the initial document into the specified collection. This actually creates the collection.
    result = db[collection_name].insert_one(initial_document)

    if result.acknowledged:
        print(f"Database '{database_name}' and collection '{collection_name}' created successfully.")
        print(f"Inserted document with ID: {result.inserted_id}")
    else:
        raise Exception("Failed to insert document and create database/collection.")

    return db


def insert_document(
    collection,
    document: Dict[str, Any],
    key: Any = None
) -> str:
    """
    Insert a single document into the specified collection with a custom key.
    :param collection: MongoDB collection object
    :param document: Document to insert
    :param key: Optional unique key value for the document
    :return: Inserted document ID if successful, None otherwise
    """
    try:
        if key is not None:
            # Check if a document with the same key already exists
            if collection.find_one({"_id": key}):
                print(f"Document with key '{key}' already exists in the collection.")
                return None
            # Merge the key and document
            document_to_insert = {"_id": key, **document}
        else:
            document_to_insert = document

        # Insert the document
        result = collection.insert_one(document_to_insert)
        
        if result.acknowledged:
            return str(result.inserted_id)
        else:
            print(f"Failed to insert document with key: {key}")
            return None
    except Exception as e:
        print(f"Error inserting document with key: {key}. Error: {str(e)}")
        return None

