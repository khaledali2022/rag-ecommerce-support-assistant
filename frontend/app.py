import streamlit as st

from api_client import API_BASE_URL, ApiError, ask_question, check_health

st.set_page_config(page_title="E-Commerce Support Assistant", page_icon="🛒", layout="centered")

st.title("🛒 RAG E-Commerce Support Assistant")
st.caption(f"Backend: {API_BASE_URL}")

with st.sidebar:
    st.header("Status")
    try:
        health = check_health()
        if health.get("vector_store_loaded"):
            st.success(f"Backend healthy ✅\n\n{health.get('collection_count', 0)} chunks indexed")
        else:
            st.warning("Backend is up, but the vector store is not loaded yet.")
    except ApiError as exc:
        st.error(str(exc))

    st.markdown("---")
    st.markdown(
        "Ask about **orders, shipping, returns, refunds, payment, account, "
        "cancellation, delivery,** or **products**."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            st.caption("Sources: " + ", ".join(msg["sources"]))

question = st.chat_input("Ask a question, e.g. 'Where is my order?'")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating a grounded answer..."):
            try:
                result = ask_question(question)
                answer = result["answer"]
                sources = result.get("sources", [])
                st.markdown(answer)
                if sources:
                    st.caption("Sources: " + ", ".join(sources))
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except ApiError as exc:
                error_message = f"⚠️ {exc}"
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message, "sources": []}
                )
