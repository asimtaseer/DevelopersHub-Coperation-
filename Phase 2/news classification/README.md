# News Topic Classification System

# Introduction
This project is a professional **News Topic Classification System** built using a state-of-the-art Transformer model. The system is designed to automatically categorize news articles, headlines, or paragraphs into specific topics with high accuracy. It leverages a model trained on the widely recognized **AG News dataset** and provides a real-time, interactive web interface for users.

# About the Model
The underlying model is a Transformer-based sequence classifier that was fine-tuned on the **AG News dataset**, which consists of thousands of news articles. After training, the model was optimized and uploaded to the Hugging Face Model Hub for easy accessibility and deployment.

**Model Name:** [asimtaseer/news-classifier](https://huggingface.co/asimtaseer/news-classifier)

# Categories
The system supports the following four major news categories:
*   **World**: Global news, international relations, and world events.
*   **Sports**: Updates from the world of athletics, leagues, and competitions.
*   **Business**: Financial news, economy, corporate updates, and markets.
*   **Science/Technology**: Innovations, gadgets, scientific discoveries, and tech trends.

# Features
*   **News Classification**: Instantly identify the topic of any news snippet.
*   **Confidence Score**: Get a percentage-based confidence level for every prediction.
*   **Hugging Face Integration**: Seamlessly loads the model directly from the cloud.
*   **Streamlit Interface**: A modern, responsive, and user-friendly dashboard.
*   **Real-Time Predictions**: Fast and efficient inference powered by PyTorch.

# Installation
To set up this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/asimtaseer/DevelopersHub-Coperation/Phase 2/Task1-.git
   cd "project folder"
   ```

2. **Create a virtual environment (Optional but Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

# Run Locally
To launch the application, run the following command in your terminal:

```bash
streamlit run app.py
```

# Example Usage
**Input:**
> "The stock market saw a significant rise today as major tech companies reported higher than expected quarterly earnings."

**Output:**
*   **Predicted Category:** Business 💼
*   **Confidence Score:** 98.45%

# Technologies Used
*   **Python**: Core programming language.
*   **Streamlit**: Framework for the web interface.
*   **Transformers**: Hugging Face library for state-of-the-art NLP.
*   **PyTorch**: Deep learning framework for model inference.
*   **Hugging Face**: Platform for sharing and hosting the model.

# Author
**Name:** Asim Taseer
**GitHub:** [github.com/asimtaseer](https://github.com/asimtaseer)
**LinkedIn:** [linkedin.com/in/asim-taseer](https://www.linkedin.com/in/asimtaseer)
**Portfolio:** [asimtaseer.unaux.com](https://asimtaseer.unaux.com)

# License
This project is licensed under the **MIT License**.

```text
MIT License

Copyright (c) 2024 Asim Taseer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
