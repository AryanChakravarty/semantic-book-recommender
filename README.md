# Semantic Book Recommender System

A sophisticated book recommendation system that uses natural language processing and machine learning to provide personalized book recommendations based on semantic similarity and emotional content.

## Features

- **Semantic Search**: Find books similar to your description using advanced NLP
- **Emotion Analysis**: Get recommendations based on emotional tone
- **Category Filtering**: Filter recommendations by book categories
- **Beautiful UI**: Modern Gradio interface with book covers and descriptions
- **Real-time Recommendations**: Get instant results as you type

## Project Structure

```
semantic-book-recommender/
├── notebooks/                    # Jupyter notebooks for development
│   ├── data-exploration.ipynb
│   ├── text-classification.ipynb
│   ├── sentiment-analysis.ipynb
│   └── vector-search.ipynb
├── src/                         # Source code
│   └── gradio-dashboard.py      # Main application
├── docs/                        # Documentation
│   └── code_documentation.txt   # Detailed code documentation
├── requirements.txt             # Python dependencies
├── .env.example                # Example environment variables
└── README.md                   # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/semantic-book-recommender.git
cd semantic-book-recommender
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Usage

1. Run the notebooks in order to set up the data and models:
   - `notebooks/data-exploration.ipynb`
   - `notebooks/text-classification.ipynb`
   - `notebooks/sentiment-analysis.ipynb`
   - `notebooks/vector-search.ipynb`

2. Launch the dashboard:
```bash
python src/gradio-dashboard.py
```

3. Open your browser and navigate to the URL shown in the terminal (usually http://localhost:7860)

## Technical Details

- Uses Hugging Face's DistilRoBERTa for emotion analysis
- Implements sentence transformers for semantic search
- Leverages Chroma DB for vector storage
- Built with Gradio for the user interface

## Requirements

- Python 3.8+
- 8GB+ RAM recommended
- GPU optional but recommended for faster processing

## License

MIT License - see LICENSE file for details

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request 