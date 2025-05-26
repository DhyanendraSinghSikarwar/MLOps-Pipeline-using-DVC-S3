import os
import logging
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Create logs directory
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.DEBUG)

# Create handlers
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(os.path.join(log_dir, 'data_preprocessing.log'))

# Set level for handlers
console_handler.setLevel(logging.DEBUG)
file_handler.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Ensure NLTK data is available
def download_nltk_resources():
    try:
        # Download required NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    except Exception as e:
        logger.error(f"Error downloading NLTK resources: {e}")
        raise

# Call the download function before text processing
download_nltk_resources()

# Text normalization function
def normalize_text(text):
    try:
        if pd.isnull(text):
            return ""

        # Convert to lowercase and split on whitespace
        text = str(text).lower()
        # Split on whitespace and filter out stopwords and punctuation
        stop_words = set(stopwords.words("english"))
        tokens = [word for word in text.split() 
                 if word not in stop_words 
                 and word not in string.punctuation]
        return " ".join(tokens)

    except Exception as e:
        logger.error(f"Error during text normalization: {e}")
        return ""

# Main preprocessing function
def preprocess_df(df, text_column="text", target_column="label"):
    try:
        logger.debug("Starting preprocessing for DataFrame")

        # Encode target column if exists
        if target_column in df.columns:
            df[target_column] = df[target_column].astype("category").cat.codes
            logger.debug("Target column encoded")

        # Normalize text column
        if text_column in df.columns:
            df[text_column] = df[text_column].apply(normalize_text)
            logger.debug("Text column normalized")

        # Remove duplicates
        df.drop_duplicates(inplace=True)
        logger.debug("Duplicates removed")

        return df

    except Exception as e:
        logger.error(f"Failed to complete the data transformation process: {e}")
        raise



def main(text_column='text', target_column='target'):
    """
    Main function to load raw data, preprocess it, and save the processed data.
    """
    try:
        # Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Data loaded properly')

        # Transform the data
        train_processed_data = preprocess_df(train_data, text_column, target_column)
        test_processed_data = preprocess_df(test_data, text_column, target_column)

        # Store the data inside data/processed
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)
        
        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        
        logger.debug('Processed data saved to %s', data_path)
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()