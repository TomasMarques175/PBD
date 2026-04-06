# Processing Big Data — Portuguese Election Debates Analysis (2024/2025)

A multimodal data science project analysing the 28 one-on-one TV debates between Portugal's 8 major political parties in the lead-up to the March 2024 legislative elections. The project processes visual, audio, and text data to extract meaningful insights about each party and their representatives.

---

## Project Overview

Portugal's 2024 election season featured 28 debates across three national TV stations, with all eight major parties participating. For each debate, rich multimodal data was extracted — including detected objects, human poses, facial expressions, audio features, speaker transcripts, and debate scores. This project explores that data through the full data science pipeline: EDA, representation, visualization, modelling, and performance analysis.

> **Course:** Processing Big Data (PBD) 2024/2025  
> **Dataset:** 28 debate videos × 4 feature types (`_visual.pkl`, `_audio.pkl`, `_speech.pkl`, `_text.pkl`)

---

## Contributors

| Name | GitHub |
|---|---|
| Tomás Marques | [@TomasMarques175](https://github.com/TomasMarques175) |
| Tiago Mira | [@tiagomira29](https://github.com/tiagomira29) |
| Miguel | [@Miguelyin](https://github.com/Miguelyin) |

---

## Dataset Description

Each debate video was processed to extract four types of features, stored as pandas DataFrames in pickle files:

**Visual Features** (`*_visual.pkl`) — sampled at 1 frame/second:
- Detected objects: bounding box `[x, y, width, height]`, class label (up to 80 classes), confidence score
- Human poses: 33 3D keypoints per person with visibility and presence probabilities
- Faces & emotions: bounding box, expression label (8 classes), 128-dimensional facial embedding

**Audio Features** (`*_audio.pkl`) — extracted from speaker-diarized segments (min. 20s):
- Pitch (F0), HNR, jitter, shimmer, formants
- Speech rate, articulation rate, number of pauses, speaking time
- 512-dimensional speaker embedding (pyannote)

**Speech/Text Features** (`*_speech.pkl`):
- Audio transcripts produced by OpenAI Whisper
- Speaker-aligned text segments

**Text Features** (`*_text.pkl`):
- Global word dictionary across all videos
- Debate scores assigned by a Portuguese TV station

---

## Project Structure

```
PBD/
├── PBD_Project.ipynb        # Main analysis notebook (143 cells)
├── PBD_Project_Intro.ipynb  # Introductory notebook
└── Project_Data/
    └── Features/
        ├── *_visual.pkl
        ├── *_audio.pkl
        ├── *_speech.pkl
        ├── *_text.pkl
        ├── dictionary_text.pkl
        ├── pulsometer_text.pkl
        └── grades_text.pkl
```

---

## Analysis Pipeline

### 1. Exploratory Data Analysis & Visualization
- Object detection counts and confidence box plots across all videos
- Pose keypoint variance and person count per frame
- Facial emotion distribution (pie charts per video and overall)
- Audio feature analysis: pitch, HNR, speech rate, articulation rate

### 2. Single Video Analysis (`chega-be` as primary case)

**Visual:**
- Frames grouped by number of people detected (0, 1, 2, 3+ persons)
- Tracklet assignment using cosine similarity and Intersection over Union (IoU) on facial embeddings
- Dimensionality reduction (PCA, ISOMAP) and K-Means clustering to identify and label individual persons
- Screen time estimation per participant

**Audio:**
- Agglomerative clustering on audio features to separate speakers
- Pitch analysis and emotion binning
- Speaker diarization alignment with visual frames
- Word count and speaking turns per identified speaker

**Text:**
- Transcript alignment with speaker IDs
- Word frequency analysis (top 20 words per speaker)
- TF-IDF vectorization and Bag-of-Words representations
- Word-emotion association mapping

### 3. Multi-Video Analysis (All 28 Debates)

**Cross-video person identification:**
- Role assignment (moderator / party representative / sign language interpreter) via brute-force embedding dispersion minimization
- Normalized screen time per party across all debates

**Audio across all videos:**
- Party-to-video mapping for all 8 parties (AD, BE, CDU, Chega, IL, Livre, PAN, PS)
- Cross-video speaker embedding matching to identify consistent orators
- Average speaking time percentage per party

**Text across all videos:**
- Portuguese NLP preprocessing using spaCy (`pt_core_news_lg`) and NLTK stopword removal
- Named entity recognition to augment the vocabulary dictionary
- Per-party word frequency and TF-IDF analysis
- Timeline visualizations of who said what and when

**Multimodal correlations:**
- Audio features vs. facial emotion prediction (Random Forest classifier)
- Pose embeddings vs. emotion embeddings (PCA visualization)
- Emotion co-occurrence at speaker turns (heatmaps and correlation matrices)
- PCA projection of all videos by emotion distribution
- Debate score prediction from multimodal features

---

## Key Technologies

| Library | Purpose |
|---|---|
| `pandas` / `numpy` | Data loading and manipulation |
| `matplotlib` / `seaborn` | Visualization |
| `scikit-learn` | PCA, ISOMAP, K-Means, Random Forest, TF-IDF |
| `scipy` | Chi-square tests, Pearson correlation, audio I/O |
| `nltk` / `spacy` | Text preprocessing and NLP |
| `pyannote` | Speaker diarization and embeddings |
| `Whisper` | Audio transcription |
| `PIL` | Image processing |

---

## Setup & Usage

The notebook is designed to run on **Google Colab** with data stored in Google Drive.

1. Mount your Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

2. Place the dataset under:
```
/content/drive/MyDrive/PBD/Project/Project_Data/Features/
```

3. Open and run `PBD_Project.ipynb` cell by cell. The `multiple_videos` flag at the top controls whether to analyse a single video or all debates:
```python
multiple_videos = True   # Analyse all 28 debates
multiple_videos = False  # Analyse a single video (set video = "chega-be")
```

4. Install any missing dependencies:
```bash
pip install pyannote.audio openai-whisper spacy
python -m spacy download pt_core_news_lg
```

---

## Parties Covered

| Code | Party |
|---|---|
| `ad` | Aliança Democrática |
| `be` | Bloco de Esquerda |
| `cdu` | CDU |
| `chega` | Chega |
| `il` | Iniciativa Liberal |
| `livre` | Livre |
| `pan` | PAN |
| `ps` | Partido Socialista |

---

## Evaluation

| Component | Weight |
|---|---|
| Midterm Review (Week 4) | 15% |
| Continuous Evaluation | 10% |
| Report & Code | 20% |
| Presentation & Discussion | 20% |
| **Total** | **65% of final grade** |
