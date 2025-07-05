import gradio as gr
from agent import Agent

def chat_interface(user_input, history):
    # Initialize agent singleton
    if not hasattr(chat_interface, "agent"):
        chat_interface.agent = Agent()

    # Append user message
    chat_interface.agent.history.append({'role': 'user', 'content': user_input})

    # Stream model response
    response = ""
    for token in chat_interface.agent.stream_chat(user_input):
        response += token
        yield history + [[user_input, response]]

    # Save final assistant response
    chat_interface.agent.history.append({'role': 'assistant', 'content': response})

# Build Gradio Blocks
with gr.Blocks(title="🦙 Llama2 7B Chatbot (Gradio)") as demo:
    chatbot = gr.Chatbot()
    user_input = gr.Textbox(placeholder="Type your message...", show_label=False)
    submit_btn = gr.Button("Send")

    submit_btn.click(fn=chat_interface,
                     inputs=[user_input, chatbot],
                     outputs=chatbot,
                     queue=True)

    gr.Markdown("Powered by llama-cpp-python on Apple Silicon")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
