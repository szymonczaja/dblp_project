import streamlit as st
import requests

st.title("DBLP Research Assistant")
st.caption("Ask questions about computer science research papers")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if question := st.chat_input("Ask about research papers..."):
    
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching papers..."):
            response = requests.post(
                "http://localhost:8000/query",
                json={"question": question, "n_results": 5}
            )
            data = response.json()
            
            st.write(data["answer"])
            
            with st.expander("📚 Sources"):
                for title, meta in zip(data["titles"], data["sources"]):
                    st.write(f"**{title}** ({meta['year']}) — {meta['authors']}")
    
    st.session_state.messages.append({"role": "assistant", "content": data["answer"]})