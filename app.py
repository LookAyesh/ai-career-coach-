import os
import traceback
from flask import Flask, request, render_template, url_for, flash, redirect
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import PyPDF2
import markdown2

# LangChain and Groq imports
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain.text_splitter import CharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

# --- Initialize the Flask App ---
app = Flask(__name__)
# --- NEW: A secret key is required for flash messages ---
app.config['SECRET_KEY'] = 'a_super_secret_key_for_a_great_project'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['FAISS_INDEX_PATH'] = 'faiss_index'

# --- Add a Markdown filter to Flask ---
@app.template_filter('markdown')
def markdown_filter(s):
    return markdown2.markdown(s, extras=["fenced-code-blocks", "tables", "cuddled-lists", "break-on-newline"])

# Create necessary folders if they don't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
if not os.path.exists(app.config['FAISS_INDEX_PATH']):
    os.makedirs(app.config['FAISS_INDEX_PATH'])

# --- Configure the AI Model (LLM) ---
try:
    llm = ChatGroq(
        temperature=0.7,
        model_name="llama3-8b-8192",
        api_key=os.getenv("GROQ_API_KEY")
    )
    print("✅ Groq AI model configured successfully.")
except Exception as e:
    print(f"🛑 ERROR configuring Groq API: {e}")
    llm = None

# --- Configure the Embedding Model ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("✅ Embedding model loaded.")


# --- Helper function to extract text from PDF ---
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            if not reader.pages:
                return None, "Error: PDF has no pages."
            text = "\n".join(p.extract_text() for p in reader.pages if p.extract_text())
            if not text:
                return None, "Error: Could not extract any text from the PDF."
            return text, None
    except Exception as e:
        traceback.print_exc()
        return None, f"Error reading PDF file: {e}"


# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the main upload page."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file upload, analysis, and displays the summary."""
    if not llm:
        return "ERROR: The AI Model is not configured. Please check your GROQ_API_KEY in the .env file.", 500

    # --- NEW: A more robust way to handle the "no file" error ---
    if 'file' not in request.files or not request.files['file'].filename:
        flash('Please choose a PDF file before analyzing.')
        return redirect(url_for('index'))

    file = request.files['file']
    filename = secure_filename(file.filename)
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(pdf_path)

    resume_text, error = extract_text_from_pdf(pdf_path)
    if error:
        flash(error)
        return redirect(url_for('index'))

    try:
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(resume_text)
        vector_store = FAISS.from_texts(chunks, embeddings)
        vector_store.save_local(app.config['FAISS_INDEX_PATH'])
    except Exception as e:
        traceback.print_exc()
        return f"Error creating knowledge base: {e}", 500

    try:
        summary_prompt = PromptTemplate.from_template(
            "You are an expert AI Career Coach. Your task is to provide a clean, professional, and structured summary of the following resume. "
            "Use Markdown for formatting. Specifically, use bold headings for each section and bullet points for lists.\n\n"
            "*Resume Text:*\n---\n{resume}\n---\n\n"
            "*Analysis:*\n"
            "*Career Objective:* (Summarize the candidate's primary career goal in one clear sentence)\n\n"
            "*Key Skills:\n (List a skill)\n* (List another skill)\n\n"
            "*Work Experience Summary:*\n(Provide a brief paragraph summarizing the roles)\n\n"
            "*Education:\n (List degree and institution)"
        )
        summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
        summary = summary_chain.invoke({"resume": resume_text}).get('text', 'Could not generate summary.')
        
        return render_template('results.html', resume_analysis=summary)
    except Exception as e:
        traceback.print_exc()
        return f"Error during AI summary: {e}", 500


@app.route('/ask', methods=['GET', 'POST'])
def ask_query():
    """Handles the Q&A page and answering questions."""
    if request.method == 'POST':
        question = request.form.get('query')
        if not question:
            flash('Please type a question before submitting.')
            return redirect(url_for('ask_query'))

        try:
            vector_store = FAISS.load_local(app.config['FAISS_INDEX_PATH'], embeddings, allow_dangerous_deserialization=True)
            
            qa_template = (
                "You are a helpful AI career coach. Use the following pieces of resume context to answer the user's question. "
                "Provide a clear, well-formatted answer. Use Markdown formatting like bold text and bullet points if it makes the answer easier to read.\n\n"
                "Context:\n{context}\n\n"
                "Question:\n{question}\n\n"
                "Helpful Answer:"
            )
            QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["context", "question"])

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(),
                chain_type_kwargs={"prompt": QA_PROMPT}
            )
            answer = qa_chain.invoke({"query": question}).get('result', 'Could not find an answer.')
            
            return render_template('qa_results.html', query=question, result=answer)
        except Exception as e:
            traceback.print_exc()
            return f"Error during Q&A: {e}", 500
    
    return render_template('ask.html')


# --- Run the App ---
app.run(debug=True)