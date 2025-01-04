# Moj Video Data Extraction and Analysis

## Project Overview
This project is a comprehensive Python-based solution designed to scrape data from the Moj video-sharing platform, analyze it, and generate insights. It allows users to extract video details such as likes, comments, shares, audio transcriptions (with translation), and content metadata. Additionally, the data is saved into a CSV file for further analysis. A Streamlit interface is provided to facilitate user interaction and streamline data extraction.

---

## Features

1. **Data Extraction**:
   - Scrapes likes, comments, and shares for each video.
   - Extracts content metadata (e.g., text from alt attributes).
   - Downloads audio files and transcribes them using Whisper.
   - Translates transcriptions from Hindi to English using Google Translate.

2. **Streamlit Integration**:
   - User-friendly interface to input the number of videos to scrape and specify the CSV file name.
   - Displays progress and logs directly in the browser.

3. **Error Handling**:
   - Includes retry mechanisms for audio extraction.
   - Handles Selenium exceptions and audio download/transcription errors.

4. **Data Storage**:
   - Saves data into a CSV file after processing each video, ensuring intermediate results are preserved.
   - Includes columns for likes, comments, shares, content, audio URL, transcription, and English translation.

5. **Efficiency**:
   - Implements timeout limits for each step to avoid long delays.
   - Uses multithreading for processing audio URLs.

---

## Technology Stack

- **Python Libraries**:
  - `selenium`: Web scraping.
  - `whisper`: Audio transcription.
  - `googletrans`: Text translation.
  - `streamlit`: Web-based user interface.
  - `requests`: HTTP requests for downloading audio.
  - `csv`: Data storage.
  - `logging`: Event tracking and debugging.

- **Tools**:
  - ChromeDriver: Selenium WebDriver for interacting with Moj's website.
  - Torch: Backend for Whisper.

---

## Installation

### Prerequisites
1. Install Python (>=3.8).
2. Install Chrome browser and download the matching version of ChromeDriver.

### Python Dependencies
Install required libraries using pip:
```bash
pip install selenium requests googletrans==4.0.0-rc1 torch streamlit openai-whisper
```

---

## How to Run

### Streamlit Interface
1. Clone the repository and navigate to the project directory.
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Enter the number of videos to scrape and the desired CSV file name.
4. Click the start button to begin the extraction process.

### Command-Line Execution
Alternatively, you can run the script directly from the command line for batch processing:
```bash
python scrap_moj.py
```

---

## Project Structure

```plaintext
.
├── scrap_moj.py          # Main scraping script with Streamlit integration
├── requirements.txt      # Required Python libraries
├── moj_data_translated.csv # Sample output CSV file
├── README.md             # Project documentation
```

---

## CSV File Format

The generated CSV file includes the following columns:

| Likes | Comments | Shares | Content | Audio URL | Transcription | English Translation |
|-------|----------|--------|---------|-----------|---------------|----------------------|
| 425K  | 28       | 207    | #funny  | ...       | ...           | ...                  |

---

## Key Functions

### 1. **`extract_video_details(driver)`**
Extracts likes, comments, shares, and content metadata from the current video page.

### 2. **`extract_audio_url(driver)`**
Navigates to the audio page, retrieves the audio URL, and returns to the main page.

### 3. **`transcribe_audio(file_path)`**
Uses Whisper to transcribe the audio file into Hindi.

### 4. **`translate_to_english(text)`**
Translates the transcribed Hindi text into English using Google Translate.

### 5. **`write_to_csv(data, file_path)`**
Writes the extracted data to a CSV file after processing each video.

---

## Challenges and Solutions

### 1. **Slow Audio Transcription**
- **Problem**: Whisper transcription sometimes took excessive time.
- **Solution**: Implemented a timeout for transcription tasks.

### 2. **Dynamic Web Elements**
- **Problem**: Moj's website dynamically updates elements, causing occasional failures.
- **Solution**: Added retries and explicit waits for robust element detection.

### 3. **Translation API Limits**
- **Problem**: Google Translate API introduced delays for large-scale translations.
- **Solution**: Added a short sleep between translations to avoid rate-limiting.

---

## Future Improvements

1. **Content Categorization**:
   - Use NLP techniques to classify video content into predefined categories.

2. **Dashboard Integration**:
   - Create a dashboard with Streamlit for visualizing engagement metrics.

3. **Scalability**:
   - Parallelize the scraping process for faster data collection.
