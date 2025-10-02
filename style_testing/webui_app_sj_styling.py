from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import gradio as gr
import uvicorn
import threading
import base64
import os
import argparse

from document_loader import DocumentLoader
from text_processor import TextProcessor
from vector_store import VectorStoreManager
from qa_chain import QAChainBuilder
from chatbot import CRISPRChatbot

# FastAPI app
app = FastAPI(title="LabAssist API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chatbot_instance = None

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[str]
    conversation_id: Optional[str] = None

def initialize_chatbot():
    global chatbot_instance
    print("Initializing LabAssist chatbot...")
    
    vector_store_manager = VectorStoreManager()
    
    try:
        vectorstore = vector_store_manager.load_vectorstore()
        print("✓ Vector store loaded")
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return
    
    retriever = vector_store_manager.get_retriever(k=2)
    qa_builder = QAChainBuilder()
    qa_chain = qa_builder.build_chain(retriever)
    
    chatbot_instance = CRISPRChatbot(qa_chain)
    print("✓ Chatbot ready")

@app.get("/")
async def root():
    return {"message": "LabAssist API is running", "status": "ok"}

@app.get("/health")
async def health():
    if chatbot_instance is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    return {"status": "healthy"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if chatbot_instance is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    try:
        result = chatbot_instance.answer_question(request.message)
        answer = result["answer"].replace("**", "").replace("*", "")
        
        return ChatResponse(
            response=answer,
            sources=result["sources"],
            conversation_id=request.conversation_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Gradio interface
def chat_interface(message, history):
    if chatbot_instance is None:
        return "Chatbot not initialized. Please wait..."
    
    try:
        result = chatbot_instance.answer_question(message)
        answer = result["answer"].replace("**", "").replace("*", "")
        
        # Process sources with page numbers
        unique_sources_info = set()
        for doc in result["source_documents"]:
            source_path = doc.metadata.get("source", "Unknown")
            source_file = os.path.basename(source_path)
            page = doc.metadata.get("page", "n/a")
            
            if isinstance(page, int):
                page_info = f"(Page {page + 1})"
            elif page != "n/a":
                page_info = f"(Page {page})"
            else:
                page_info = "(Page N/A)"
            
            unique_sources_info.add(f"{source_file} {page_info}")
        
        sources = ", ".join(sorted(list(unique_sources_info)))
        
        return {"role": "assistant", "content": f"{answer}\n\n📚 Sources: {sources}"}
    except Exception as e:
        return {"role": "assistant", "content": f"Error: {str(e)}"}

# Create custom theme
custom_theme = gr.themes.Base(
    primary_hue="neutral",
    secondary_hue="red",
    neutral_hue="slate",
    font=("Ariel", "sans-serif"),
    radius_size="md"
).set(
    # Button colors
    button_primary_background_fill="#d11947",
    button_primary_background_fill_hover="#8d0034",
    button_primary_text_color="neutral",
    button_primary_border_color="#d11947",
    
    # Secondary button (like clear, undo)
    button_secondary_background_fill="#6c757d",
    button_secondary_background_fill_hover="#5a6268",
    button_secondary_text_color="neutral",
    
    # Input box
    input_background_fill="#ffffff",
    input_border_color="#ffffff",
    input_border_width="1px",
    input_border_color_focus="#ffffff",
    
    # Chat message colors
    body_background_fill="#f8f9fa",
    block_background_fill="#ffffff",
    
    # User and assistant message colors
    color_accent="#d11947",
    color_accent_soft="#e3f2fd",
)

# Custom CSS for additional styling
custom_css = """
/* Override system theme with higher specificity */
:root {
    --primary-color: #d11947 !important;
    --primary-hover: #8d0034 !important;
    --secondary-color: #6c757d !important;
}

/* Chat message styling - more specific selectors */

/* input repeat */ 
.message-row.user-row .message-bubble-border,
.message-row.user-row > div,
div[data-testid="user"] .message-bubble-border,
div[data-testid="user"] > div > div {
    background: #f1f3f5 !important;
    border: 6px solid #17818F !important;
    border-left: 6px solid #17818F!important;  /* Override the border-left */
    border-radius: 6px !important;
    padding: 0 !important;
    color: white !important;
    outline: none !important;
    box-shadow: none !important;
}

/* Response messages */
.message-row.bot-row .message-bubble-border,
.message-row.bot-row > div,
div[data-testid="bot"] .message-bubble-border,
div[data-testid="bot"] > div > div {
    background: #f1f3f5 !important;
    border: 6px solid #62bb46 !important;
    border-left: 6px solid #62bb46 !important;  /* Override the border-left */
    border-radius: 6px !important;
    padding: 0 !important;
    color: white !important;
    outline: none !important;
    box-shadow: none !important;
}

/* User message text color */
.message-row.user-row p,
.message-row.user-row span,
div[data-testid="user"] p,
div[data-testid="user"] span {
    color: #ffffff !important;
}

/* Assistant message text color */
.message-row.bot-row p,
.message-row.bot-row span,
div[data-testid="bot"] p,
div[data-testid="bot"] span {
    color: #ffffff !important;
}

/* Primary button styling - target all variations */
button.primary,
button[variant="primary"],
.primary-button,
button.submit-btn,
gradio-app button[type="submit"] {
    background: linear-gradient(135deg, #d11947 0%, #8d0034 100%) !important;
    border: none !important;
    color: white !important;
    box-shadow: 0 4px 6px rgba(209, 25, 71, 0.2) !important;
    transition: all 0.3s ease !important;
    font-weight: 600 !important;
}

button.primary:hover,
button[variant="primary"]:hover,
.primary-button:hover,
button.submit-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 12px rgba(209, 25, 71, 0.3) !important;
    background: linear-gradient(135deg, #8d0034 0%, #6a0028 100%) !important;
}

/* Secondary buttons */
button.secondary,
button[variant="secondary"],
.secondary-button {
    background: #6c757d !important;
    color: white !important;
    border: none !important;
    transition: all 0.3s ease !important;
}

button.secondary:hover,
button[variant="secondary"]:hover,
.secondary-button:hover {
    background: #5a6268 !important;
    transform: translateY(-1px) !important;
}

/* Input box styling - more specific */
.gradio-container input[type="text"],
.gradio-container textarea,
gradio-app input[type="text"],
gradio-app textarea {
    border: 2px solid #e5e7eb !important;
    border-radius: 12px !important;
    transition: border-color 0.3s ease !important;
    background: white !important;
    color: #474c55 !important;
}

.gradio-container input[type="text"]:focus,
.gradio-container textarea:focus,
gradio-app input[type="text"]:focus,
gradio-app textarea:focus {
    border-color: #d11947 !important;
    box-shadow: 0 0 0 3px rgba(209, 25, 71, 0.1) !important;
    outline: none !important;
}

/* Header styling */
.gradio-container h1,
gradio-app h1 {
    background: linear-gradient(135deg, #d11947 0%, #8d0034 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    font-weight: bold !important;
}

/* Chat container background */
.gradio-container,
gradio-app {
    background: #f8f9fa !important;
}

/* Chatbot container */
.chatbot,
.chatbot-container,
div[data-testid="chatbot"] {
    background: #474c55 !important;
    border-radius: 12px !important;
    border: 1px solid #474c55 !important;
}

/* Example buttons styling */
.examples {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
}

.examples > div {
    display: flex !important;
    justify-content: center !important;
    flex-wrap: wrap !important;
    gap: 10px !important;
}

/* Target the actual gallery buttons */
.gallery.svelte-p5q82i {
    display: flex !important;
    justify-content: center !important;
    gap: 10px !important;
}

button.gallery-item.svelte-p5q82i {
    background: #474c55 !important;
    background-color: #474c55 !important;
    border: 2px solid #474c55 !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}

button.gallery-item.svelte-p5q82i:hover {
    border-color: #d11947 !important;
    background: #d11947 !important;
    background-color: #d11947 !important;
    color: #ffffff !important;
    transform: translateY(-1px) !important;
}

/* Target the inner div text container */
button.gallery-item.svelte-p5q82i .svelte-1oitfqa {
    color: #ffffff !important;
}

button.gallery-item.svelte-p5q82i:hover .svelte-1oitfqa {
    color: #ffffff !important;
}

/* Examples label styling */
.label.svelte-p5q82i {
    color: #474c55 !important;
    text-align: center !important;
    font-size: 1.5em !important;
    font-weight: 600 !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    gap: 8px !important;
}

.label.svelte-p5q82i svg {
    color: #474c55 !important;
}




"""

# Load logo image
def get_logo_base64():
    try:
        with open("logo.png", "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        print("Warning: logo.png not found")
        return None

logo_base64 = get_logo_base64()

# Build interface with custom blocks for logo
with gr.Blocks(theme=custom_theme, css=custom_css, title="LabAssist") as demo:
    # Custom header with logo
    with gr.Row():
        if logo_base64:
            gr.HTML(f"""
                <div style="display: flex; align-items: center; justify-content: center; gap: 15px; margin-bottom: 20px;">
                    <img src="data:image/png;base64,{logo_base64}" alt="SJCRHLogo" style="width: 60px; height: 60px; object-fit: contain;">
                    <div>
                        <h1 style="margin: 0; font-size: 2em;">LabAssist - Laboratory Protocol Assistant</h1>
                        <p style="margin: 5px 0 0 0; color: #666;">Ask questions about lab protocols and get answers with cited sources</p>
                    </div>
                </div>
            """)
        else:
            gr.HTML("""
                <div style="display: flex; align-items: center; justify-content: center; gap: 15px; margin-bottom: 20px;">
                    <div>
                        <h1 style="margin: 0; font-size: 2em;">🧬 LabAssist - Laboratory Protocol Assistant</h1>
                        <p style="margin: 5px 0 0 0; color: #666;">Ask questions about lab protocols and get answers with cited sources</p>
                    </div>
                </div>
            """)
    
    # Chat interface
    chatbot = gr.Chatbot(type="messages", height=350)
    
    with gr.Row():
        msg = gr.Textbox(
            placeholder="Type your question here...",
            show_label=False,
            scale=4,
            container=False
        )
        submit = gr.Button("Send Message", variant="primary", scale=1)
    
    with gr.Row():
        retry = gr.Button("🔄 Retry", variant="secondary", size="sm")
        undo = gr.Button("↩️ Undo", variant="secondary", size="sm")
        clear = gr.Button("🗑️ Clear Chat", variant="secondary", size="sm")
    
    # Examples
    gr.Examples(
        examples=[
            "What is the protocol for RNP transfection?",
            "How do I prepare plasmid DNA?",
            "What is the correct DNA concentration for NGS submission?",
        ],
        inputs=msg,
        label="Example Questions",
    )
    
    # Event handlers
    def respond(message, chat_history):
        if chatbot_instance is None:
            bot_message = {"role": "assistant", "content": "Chatbot not initialized. Please wait..."}
        else:
            try:
                result = chatbot_instance.answer_question(message)
                answer = result["answer"].replace("**", "").replace("*", "")
                
                # Process sources with page numbers
                unique_sources_info = set()
                for doc in result["source_documents"]:
                    source_path = doc.metadata.get("source", "Unknown")
                    source_file = os.path.basename(source_path)
                    page = doc.metadata.get("page", "n/a")
                    
                    if isinstance(page, int):
                        page_info = f"(Page {page + 1})"
                    elif page != "n/a":
                        page_info = f"(Page {page})"
                    else:
                        page_info = "(Page N/A)"
                    
                    unique_sources_info.add(f"{source_file} {page_info}")
                
                sources = ", ".join(sorted(list(unique_sources_info)))
                bot_message = {"role": "assistant", "content": f"{answer}\n\n📚 Sources: {sources}"}
            except Exception as e:
                bot_message = {"role": "assistant", "content": f"Error: {str(e)}"}
        
        chat_history.append({"role": "user", "content": message})
        chat_history.append(bot_message)
        return "", chat_history
    
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    submit.click(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)
    
    def undo_last(chat_history):
        if len(chat_history) >= 2:
            return chat_history[:-2]
        return chat_history
    
    undo.click(undo_last, chatbot, chatbot)
    
    def retry_last(chat_history):
        if len(chat_history) >= 1 and chat_history[-1]["role"] == "assistant":
            chat_history = chat_history[:-1]
        if len(chat_history) >= 1 and chat_history[-1]["role"] == "user":
            last_msg = chat_history[-1]["content"]
            chat_history = chat_history[:-1]
            return respond(last_msg, chat_history)
        return "", chat_history
    
    retry.click(retry_last, chatbot, [msg, chatbot])

def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8013, log_level="info")

if __name__ == "__main__":
    initialize_chatbot()
    
    api_thread = threading.Thread(target=run_fastapi, daemon=True)
    api_thread.start()
    
    print("\n" + "="*50)
    print("🧬 LabAssist is starting...")
    print("="*50 + "\n")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7890,
        share=True,
        root_path="/user/lead/proxy/7890"
    )