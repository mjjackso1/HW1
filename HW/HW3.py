import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import anthropic

st.title("💬 HW 3: Chat about a URL")
st.write(
    "Enter up to two URLs in the sidebar and pick an LLM. The chatbot reads those pages "
    "and answers questions about them. It keeps the last 6 messages (3 exchanges) as memory. "
    "Older messages are dropped, but the page content stays in the system prompt the whole time."
)

@st.cache_data
def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        return soup.get_text()
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None

url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
llm = st.sidebar.selectbox("LLM", ["OpenAI", "Claude"])

context = ""
for url in (url1, url2):
    if url:
        text = read_url_content(url)
        if text:
            context += f"\n\n--- Content from {url} ---\n{text}"

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer questions using the web page content below. "
    "If the answer is not in the content, say so.\n" + context
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask about the pages"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # buffer: last 6 messages (3 exchanges)
    buffer = st.session_state.messages[-6:]
    answer = None

    with st.chat_message("assistant"):
        try:
            if llm == "OpenAI":
                client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
                stream = client.chat.completions.create(
                    model="gpt-5.5",
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}] + buffer,
                    stream=True,
                )
                answer = st.write_stream(stream)
            else:
                client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
                with client.messages.stream(
                    model="claude-opus-4-5",
                    max_tokens=1000,
                    system=SYSTEM_PROMPT,
                    messages=buffer,
                ) as stream:
                    answer = st.write_stream(stream.text_stream)
        except Exception as e:
            st.error(f"Something went wrong: {e}")

    if answer:
        st.session_state.messages.append({"role": "assistant", "content": answer})