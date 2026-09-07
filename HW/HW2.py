import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import anthropic

st.title("🔗 HW 2: URL Summarizer")

def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        return soup.get_text()
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None

url = st.text_input("Enter a URL to summarize")

llm = st.sidebar.selectbox("LLM", ["OpenAI", "Claude"])
language = st.sidebar.selectbox("Output language", ["English", "Spanish", "French", "Chinese"])
summary_type = st.sidebar.selectbox(
    "Summary type",
    [
        "Summarize in 100 words",
        "Summarize in 2 connecting paragraphs",
        "Summarize in 5 bullet points",
    ],
)
advanced = st.sidebar.checkbox("Use advanced model")

if url:
    document = read_url_content(url)
    if document:
        prompt = f"Here's a web page: {document}\n\n---\n\n{summary_type}. Write the summary in {language}."

        try:
            if llm == "OpenAI":
                client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
                model = "gpt-5.5" if advanced else "gpt-5-nano"
                response = client.responses.create(model=model, input=prompt)
                st.write(response.output_text)
            else:
                client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
                model = "claude-opus-4-5" if advanced else "claude-haiku-4-5"
                response = client.messages.create(
                    model=model,
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}],
                )
                st.write(response.content[0].text)
        except Exception as e:
            st.error(f"Something went wrong: {e}")