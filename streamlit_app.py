import streamlit as st
from openai import OpenAI
from pypdf import PdfReader

st.title("📄 HW1: Document Question Answering")
st.write(
    "Upload a document below and ask a question about it. GPT will answer. "
    "You need an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys)."
)

openai_api_key = st.text_input("OpenAI API Key", type="password")

def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    client = OpenAI(api_key=openai_api_key)

    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .pdf)", type=("txt", "pdf")
    )

    model = st.selectbox(
        "Choose a model",
        ("gpt-3.5-turbo", "gpt-4.1", "gpt-5-chat-latest", "gpt-5-nano"),
    )

    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )

    if uploaded_file and question:
        file_extension = uploaded_file.name.split('.')[-1]
        if file_extension == 'txt':
            document = uploaded_file.read().decode()
        elif file_extension == 'pdf':
            document = read_pdf(uploaded_file)
        else:
            st.error("Unsupported file type.")
            document = None

        if document:
            messages = [
                {
                    "role": "user",
                    "content": f"Here's a document: {document} \n\n---\n\n {question}",
                }
            ]

            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
            )

            st.write_stream(stream)