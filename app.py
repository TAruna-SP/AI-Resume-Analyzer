import streamlit as st
import tempfile

from main import (
    extract_text_from_pdf,
    chunk_text,
    get_embeddings,
    retrieve_chunks,
    generate_answer,
    create_faiss_index
)

st.title("📄 AI Resume Analyzer")
st.write("Upload your resume and ask questions")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
query = st.text_input("Ask a question")


# ✅ Cache processing (VERY IMPORTANT)
@st.cache_resource
def process_resume(file_path):
    text = extract_text_from_pdf(file_path)
    chunks = chunk_text(text)
    embeddings = get_embeddings(chunks)
    index = create_faiss_index(embeddings)
    return chunks, index


if uploaded_file is not None:

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success("Resume uploaded successfully!")

    # ✅ Process only ONCE using cache
    chunks, index = process_resume(file_path)

    if query:
        retrieved_chunks = retrieve_chunks(query, index, chunks)
        answer = generate_answer(query, retrieved_chunks)

        st.subheader("Answer:")
        st.write(answer)