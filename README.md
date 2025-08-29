# GEN_AI_PROJECT
📄 Research Paper Summarizer (Groq + LangChain + Streamlit)







---

🚀 Overview

This project is a Research Paper Summarizer powered by Groq LLM API + LangChain + Streamlit.
It helps researchers, students, and developers quickly summarize research papers, extract key details, and ask questions about uploaded PDFs.

✅ Upload any research paper (PDF)
✅ Get a structured summary (Problem, Methods, Findings, Conclusion)
✅ Ask custom questions (authors, dataset, metrics, etc.)
✅ If the paper doesn’t mention something → "The paper does not mention this."
✅ Handles terminology: if a term is missing in the paper, it explains it using general knowledge


---

✨ Features

Summarization: Generates a structured summary of the paper.

Q&A Assistant: Answers questions in 3–5 lines, using only paper context.

Detail Extraction: Fetches exact details (authors, year, dataset, methods, metrics).

Contextual Replies: Returns closest matching passage if asked about specific lines.

Terminology Help: Explains concepts not found in the paper.

User-Friendly UI: Built with Streamlit for simple browser-based interaction.



---

🛠 Tech Stack

Python 3.9+

Streamlit – Web UI

LangChain – LLM Orchestration

Groq API – LLM Backend

FAISS – Vector Database

HuggingFace Transformers – Embeddings

PyPDF2 – PDF Text Extraction



---

📂 Project Structure

research-summarizer/
│── app.py              # Main Streamlit app
│── requirements.txt    # Dependencies
│── .env.example        # Example API key file
│── README.md           # Documentation
│── .gitignore          # Ignore env & venv files


---

⚙ Installation & Setup

1. Clone the Repository

git clone https://github.com/your-username/research-summarizer.git
cd research-summarizer

2. Create a Virtual Environment

python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

3. Install Dependencies

pip install -r requirements.txt

4. Setup Environment Variables

Create a .env file in the root directory:

GROQ_API_KEY=your_api_key_here

(⚠ Never share your real API key, and don’t upload .env to GitHub)

5. Run the App

streamlit run app.py

The app will open in your browser at http://localhost:8501.


---

📸 Screenshots (Optional – Add later)

🖼 Upload PDF UI

🖼 Generated Summary Output

🖼 Q&A Section Example



---

🔮 Future Improvements

Support for multiple PDF uploads

Export summaries to Word / PDF format

Add citation extraction (references, DOI, BibTeX)

Add chat history with the paper



---

🤝 Contributing

Contributions are welcome!

1. Fork this repo


2. Create a new branch (feature-new)


3. Commit your changes


4. Push & open a Pull Request

