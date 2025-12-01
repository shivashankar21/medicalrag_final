# ui/streamlit_app.py
import streamlit as st
import os, requests
API_URL = os.getenv("RAG_API_URL", "http://localhost:8000/ask")

st.set_page_config(page_title="MedHelp RAG")
st.title("MedHelp RAG — Medical Assistant (MVP)")

query = st.text_area("Enter medical question", height=140)
if st.button("Ask"):
    if not query.strip():
        st.warning("Type a question.")
    else:
        with st.spinner("Getting answer..."):
            r = requests.post(API_URL, json={"query": query})
            if r.status_code != 200:
                st.error(f"API error: {r.status_code}")
            else:
                j = r.json()
                st.subheader("Answer")
                st.write(j.get("answer"))
                st.markdown(f"**Confidence:** {j.get('score')}")
                st.markdown(f"**Sources:** {j.get('source_ids')}")
