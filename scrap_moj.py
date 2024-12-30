import pandas as pd, numpy as np
import logging
import time
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

import csv
import requests
import whisper
import warnings
import torch
from typing import Optional, List, Tuple
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from googletrans import Translator
import streamlit as st

import re
from textblob import TextBlob
import time
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Suppress TensorFlow Lite and WebGL warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configure Chrome WebDriver options
chrome_options = Options()
chrome_options.add_argument("--disable-gpu")  # Disable GPU hardware acceleration
chrome_options.add_argument("--disable-software-rasterizer")  # Disable software rasterizer
chrome_options.add_argument("--disable-usb")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# URL and XPATH configurations
MOJ_URL = "https://mojapp.in/"
LIKE_XPATH = [
    "//div[contains(@class, 'text-sm') and contains(@class, 'font-bold') and not(contains(text(), '@'))]",
    "//div[contains(@class, 'flex') and contains(@class, 'items-center') and contains(text(), 'K')]"
]
COMMENT_XPATH = "//div[@data-testid='comment-button']/div[contains(@class, 'text-sm') and contains(@class, 'font-bold')]"
SHARE_XPATH = "//div[@data-testid='share-button']/div[contains(@class, 'text-sm') and contains(@class, 'font-bold')]"

CONTENT_XPATH = [
    "//div[@data-testid='video-item']/img[@alt]"  # Option for <img> with alt
]
AUDIO_LINK_XPATH = "//a[@data-testid='audio-item' and contains(@href, '/music/')]"
AUDIO_SRC_XPATH = "//audio[@class='lazyload']"

def setup_webdriver() -> webdriver.Chrome:
    try:
        service = Service(r"C:\\Windows\\chromedriver.exe")  # Update with your actual path
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.implicitly_wait(10)
        return driver
    except WebDriverException as e:
        logger.error(f"Failed to initialize WebDriver: {e}")
        raise

def extract_video_details(driver: webdriver.Chrome) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Extract video details (likes, comments, shares, and content) from the current page.
    """
    likes, comments, shares, content = None, None, None, None

    # Extract likes
    for like_xpath in LIKE_XPATH:
        try:
            likes_elements = WebDriverWait(driver, 5).until(
                EC.presence_of_all_elements_located((By.XPATH, like_xpath))
            )
            for element in likes_elements:
                text = element.text.strip()
                if text and (text.endswith('K') or text.replace(',', '').isdigit()):
                    likes = text
                    break
            if likes:
                break
        except (TimeoutException, NoSuchElementException):
            continue

    # Extract comments
    try:
        comment_element = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.XPATH, COMMENT_XPATH))
        )
        comments = comment_element.text.strip()
    except (TimeoutException, NoSuchElementException):
        logger.warning("Comments not found.")
        comments = None

    # Extract shares
    try:
        share_element = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.XPATH, SHARE_XPATH))
        )
        shares = share_element.text.strip()
    except (TimeoutException, NoSuchElementException):
        logger.warning("Shares not found.")
        shares = None

    # Extract content
    for content_xpath in CONTENT_XPATH:
        try:
            content_elements = WebDriverWait(driver, 5).until(
                EC.presence_of_all_elements_located((By.XPATH, content_xpath))
            )
            for element in content_elements:
                content = element.get_attribute("alt").strip() if element.tag_name == "img" else element.text.strip()
                if content:
                    break
        except (TimeoutException, NoSuchElementException):
            continue

    return likes, comments, shares, content

def extract_audio_url(driver: webdriver.Chrome, retries: int = 1, max_wait_time: int = 10) -> Optional[str]:
    """
    Extract the audio URL from the page with a timeout to prevent long delays.
    """
    for attempt in range(retries + 1):
        try:
            # Click the audio link
            audio_link_element = WebDriverWait(driver, max_wait_time).until(
                EC.presence_of_element_located((By.XPATH, AUDIO_LINK_XPATH))
            )
            audio_link_element.click()

            # Wait for the audio page to load and extract the audio source
            WebDriverWait(driver, max_wait_time).until(
                EC.presence_of_element_located((By.XPATH, AUDIO_SRC_XPATH))
            )
            audio_element = driver.find_element(By.XPATH, AUDIO_SRC_XPATH)
            audio_url = audio_element.get_attribute("src")

            # Navigate back to the main page
            driver.back()
            WebDriverWait(driver, max_wait_time).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

            return audio_url
        except (TimeoutException, NoSuchElementException):
            logger.warning(f"Audio not found. Retrying... ({attempt}/{retries})")
            driver.back()
            WebDriverWait(driver, max_wait_time).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            if attempt == retries:
                logger.warning(f"Skipping audio extraction after {retries} retries.")
                return None

def download_audio(audio_url: str, save_path: str) -> bool:
    """
    Download the audio file from the given URL.
    """
    try:
        response = requests.get(audio_url)
        with open(save_path, "wb") as file:
            file.write(response.content)
        return True
    except Exception as e:
        logger.error(f"Failed to download audio: {e}")
        return False

def transcribe_audio(file_path: str, model_type: str = "large", language: str = "hi") -> str:
    # Enforce FP32 globally
    torch.set_default_dtype(torch.float32)
    
    # Determine the device (use GPU if available, otherwise CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the Whisper model
    model = whisper.load_model(model_type, device=device)

    print(f"Using device: {device} with precision: {torch.get_default_dtype()}")

    # Transcribe audio with the specified language
    result = model.transcribe(file_path, language=language)
    
    return result.get("text", "")  # Return the transcribed text

translator = Translator()
def translate_to_english(text: str) -> Optional[str]:
    """
    Translate the given text to English using Google Translate.
    """
    try:
        if text:  # Ensure the text is not empty
            time.sleep(2)
            translation = translator.translate(text, src='hi', dest='en')
            return translation.text
    except Exception as e:
        logger.error(f"Error translating text: {e}")
        return None

def write_to_csv(data: List[Tuple[str, str, str, str, str, str, str]], file_path: str) -> None:
    """
    Write the extracted data to a CSV file, including translations.
    """
    try:
        # If the file doesn't exist, write the header first
        if not os.path.exists(file_path):
            with open(file_path, mode="w", newline="", encoding="utf-8-sig") as file:
                writer = csv.writer(file)
                writer.writerow(["Likes", "Comments", "Shares", "Content", "Audio URL", "Transcription", "English"])  # Add headers

        # Append the data
        with open(file_path, mode="a", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            for row in data:
                transcription = row[5]
                english_translation = translate_to_english(transcription) if transcription else None
                writer.writerow([*row, english_translation])  # Append row with English translation

        logger.info(f"Data successfully written to {file_path}")
    except Exception as e:
        logger.error(f"Error writing to CSV: {e}")

def refresh_and_extract(driver: webdriver.Chrome, file_path: str, refresh_count: int = 3, wait_time: int = 3) -> None:
    """
    Refresh the Moj page, extract video details, and write data to the CSV file after processing each video.
    """
    for i in range(refresh_count):
        try:
            logger.info(f"Refreshing for Video {i+1}...")
            driver.refresh()
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            time.sleep(random.uniform(wait_time + 2, wait_time + 5))

            # Extract video details
            likes, comments, shares, content = extract_video_details(driver)
            audio_url = extract_audio_url(driver, retries=2)

            transcription = None
            if audio_url:
                # Download the audio
                audio_file_path = f"audio_video_{i+1}.mp3"
                if download_audio(audio_url, audio_file_path):
                    # Transcribe the downloaded audio
                    transcription = transcribe_audio(audio_file_path)

            # Write the processed data to the CSV file immediately
            write_to_csv([(likes, comments, shares, content, audio_url, transcription)], file_path)
        except Exception as e:
            logger.error(f"Error processing Video {i+1}: {e}")



# Preprocessing functions
def convert_counts(count):
    count = str(count).lower()
    if 'k' in count:
        return int(float(count.replace('k', '')) * 1000)
    elif 'm' in count:
        return int(float(count.replace('m', '')) * 1000000)
    else:
        return int(count)  # For numbers without suffix

# Sentiment analysis function
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity  # Range: -1 (negative) to +1 (positive)

# Function for sentiment categorization
def categorize_sentiment(score):
    if score > 0.1:
        return 'Positive'
    elif score < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

# Function to preprocess text
def preprocess_text(text):
    if isinstance(text, str):
        text = re.sub(r"#", " ", text)  # Remove hashtags
        text = text.lower().strip()  # Convert to lowercase and strip whitespace
        text = re.findall(r'\w+', text)  # Extract words
        return text
    else:
        return []

model = SentenceTransformer('all-MiniLM-L6-v2')
# Function to generate embeddings
def generate_embeddings(text):
    if text:  # If there are text
        return model.encode(" ".join(text))
    else:  # Return None for empty lists
        return None

# Function for preprocessing
def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Preprocess Likes and Shares
    df['Likes'] = df['Likes'].apply(convert_counts)
    df['Shares'] = df['Shares'].apply(convert_counts)

    # Apply sentiment analysis to the 'English' column
    df['Sentiment'] = df['English'].apply(lambda x: analyze_sentiment(str(x)))
    df['Sentiment Category'] = df['Sentiment'].apply(categorize_sentiment)

    # Preprocess text for hashtags and transcription
    df["Processed_Hashtags"] = df["Content"].apply(preprocess_text)
    df["Processed_Transcription"] = df["English"].apply(preprocess_text)

    # Combine hashtags and transcription
    df["Combined_Text"] = df.apply(
        lambda row: row["Processed_Hashtags"] + row["Processed_Transcription"], axis=1
    )

    # Generate embeddings
    df["Combined_Embeddings"] = df["Combined_Text"].apply(generate_embeddings)

    # Perform KMeans clustering
    valid_embeddings = df["Combined_Embeddings"].dropna().tolist()
    embeddings = np.vstack(valid_embeddings)
    kmeans = KMeans(n_clusters=3, random_state=42)
    empty_rows = df["Combined_Embeddings"].isna()
    df.loc[~empty_rows, "Category"] = kmeans.fit_predict(embeddings)

    # Map categories
    category_mapping = {
        0: "love", 1: "funny", 2: "sad", 3: "motivational", 4: "travel", 
        5: "festival", 6: "cute", 7: "fashion", 8: "food"
    }
    df.loc[~empty_rows, "Generalized_Category"] = df.loc[~empty_rows, "Category"].map(category_mapping)

    # Handle empty rows
    df.loc[empty_rows, "Generalized_Category"] = "unknown"
    #df.drop(columns="Unnamed: 7", inplace=True)

    # Save processed file
    processed_file_path = file_path.replace(".csv", "_processed.csv")
    df.to_csv(processed_file_path, index=False)
    
    return df, processed_file_path

# Streamlit UI integration
def main():
    st.title("Moj Video Scraper with Preprocessing")

    # User input for extraction
    refresh_count = st.number_input("Enter the number of videos to extract:", min_value=1, step=1)
    file_name = st.text_input("Enter the output CSV file name (e.g., moj_data.csv):")

    if st.button("Start Extraction"):
        if file_name:
            driver = None
            try:
                # Extract video data (as per your existing extraction function)
                driver = setup_webdriver()
                driver.get(MOJ_URL)
                time.sleep(2)
                refresh_and_extract(driver, file_name, refresh_count=refresh_count)
                st.success(f"Data extraction completed! Saved to {file_name}")
                
                # Ask user if they want preprocessing
                preprocess_choice = st.radio("Do you want to preprocess the data?", ("Yes", "No"))
                
                if preprocess_choice == "Yes":
                    # Perform preprocessing
                    df, processed_file_path = preprocess_data(file_name)
                    st.success(f"Preprocessing complete! Data saved to {processed_file_path}")
                    st.write(df.head())  # Display a preview of the processed data
                else:
                    st.info("Preprocessing skipped.")
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                if driver:
                    driver.quit()
        else:
            st.error("Please enter a valid file name.")

if __name__ == "__main__":
    main()



# def main():
#     """
#     Main function to run the Moj video scraper with Streamlit integration.
#     """
#     st.title("Moj Video Scraper")
#     st.write("Enter the number of videos to extract and a filename for storing the data.")

#     # Streamlit inputs for user parameters
#     refresh_count = st.number_input("Enter the number of videos to extract:", min_value=1, step=1)
#     file_name = st.text_input("Enter the output CSV file name (e.g., moj_data.csv):")

#     if st.button("Start Extraction"):
#         if file_name:
#             driver = None
#             try:
#                 # Setup WebDriver
#                 driver = setup_webdriver()

#                 # Navigate to Moj URL
#                 driver.get(MOJ_URL)
#                 time.sleep(2)  # Initial page load wait

#                 # Extract video data
#                 st.write(f"Extracting {refresh_count} videos...")
#                 refresh_and_extract(driver, file_name, refresh_count=refresh_count)

#                 st.success(f"Data extraction completed! Saved to {file_name}")
#             except Exception as e:
#                 st.error(f"An error occurred: {e}")
#             finally:
#                 if driver:
#                     driver.quit()
#         else:
#             st.error("Please enter a valid file name.")

# if __name__ == "__main__":
#     main()

