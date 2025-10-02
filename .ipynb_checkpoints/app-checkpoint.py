import streamlit as st
# Assuming your main logic is in qa_chain.py or main.py
# from qa_chain import run_qa_chain 

st.set_page_config(page_title="Custom Chatbot")
st.title("🤖 My LangChain Chatbot")

# Initialize chat history (important for memory)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question about the documents..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # --- Integrate your backend logic here ---
    # response = run_qa_chain(prompt) 
    response = f"This is a placeholder response for: **{prompt}**" # Replace with actual call
    # ---------------------------------------

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
