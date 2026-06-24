import os
from groq import Groq
import gradio as gr
client=Groq(api_key=os.getenv('GROQ_API_KEY'))
SYSTEM_PROMPT = ("""You are an ml engineer and you teach ml concepts to students""")
def respond(message, history, system_prompt, temperature):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": message})
    stream = client.chat.completions.create(model="llama-3.1-8b-instant",messages=messages,temperature=temperature,stream=True,)
    partial = ""
    for chunk in stream:
        partial += chunk.choices[0].delta.content or ""
        yield partial
additional_inputs = [gr.Textbox(value="You are an ml engineer and you teach ml concepts to students",label="System Prompt",lines=4,),
                     gr.Slider(minimum=0.0,maximum=2.0,value=0.4,step=0.1,label="Temperature",),]
demo = gr.ChatInterface(fn=respond,type="messages",title="ML Engineer",additional_inputs=additional_inputs,additional_inputs_accordion=gr.Accordion(label="System Settings", open=True),)
demo.launch(debug=True)
