import gradio as gr
import requests
import os
import json

# Load allowed models for dropdown
allowed_models_path = os.path.join(os.path.dirname(__file__), 'src', 'utils', 'allowed_models.json')
with open(allowed_models_path, 'r') as f:
    allowed_models_data = json.load(f)
    allowed_models = allowed_models_data.get("allowed_models", [])

def generate_video_gradio(topic, description, model, use_rag, use_visual_fix_code, use_context_learning, max_retries):
    url = "http://127.0.0.1:8000/generate"
    payload = {
        "topic": topic,
        "description": description,
        "model": model,
        "use_rag": use_rag,
        "use_visual_fix_code": use_visual_fix_code,
        "use_context_learning": use_context_learning,
        "max_retries": max_retries
    }
    try:
        response = requests.post(url, json=payload, timeout=3600)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success" and data.get("video_path"):
                video_path = data["video_path"]
                # Return video file for Gradio video component
                return f"Video generated successfully!", video_path
            else:
                return f"Error: {data.get('message', 'Unknown error')}", None
        else:
            return f"Error: {response.text}", None
    except Exception as e:
        return f"Exception: {str(e)}", None

def get_demo_videos():
    demo_dir = os.path.join(os.path.dirname(__file__), 'output')
    demos = []
    for topic in os.listdir(demo_dir):
        topic_dir = os.path.join(demo_dir, topic)
        if os.path.isdir(topic_dir):
            for file in os.listdir(topic_dir):
                if file.endswith('_combined.mp4'):
                    demos.append({
                        'topic': topic.replace('_', ' ').title(),
                        'video': os.path.join(topic_dir, file)
                    })
    return demos[:4]  # Show up to 4 demos

with gr.Blocks(theme=gr.themes.Soft(), css=".gradio-container {max-width: 1400px !important; margin: auto;} .demo-gallery video {height: 220px; border-radius: 12px;} .gradio-container {background: #f8fafc;} .gr-row, .gr-column {gap: 24px;}") as demo:
    gr.Markdown("""
    # <span style='color:#4F8EF7;font-weight:bold;'>Theory2Manim</span>
    <div style='font-size:1.1em;margin-bottom:1em;'>Generate beautiful math/science explainer videos from a single prompt. Powered by LLMs, RAG, and Manim.</div>
    """, elem_id="header")

    # Demo gallery
    gr.Markdown("## Demo Videos")
    demos = get_demo_videos()
    with gr.Row(elem_classes="demo-gallery"):
        for demo_item in demos:
            with gr.Column(scale=1):
                gr.Markdown(f"**{demo_item['topic']}**")
                gr.Video(value=demo_item['video'], show_label=False, interactive=False)

    gr.Markdown("---")
    gr.Markdown("## Try it yourself!")
    with gr.Row():
        with gr.Column():
            topic = gr.Textbox(label="Topic", placeholder="Enter topic, e.g. Chain Rule")
            description = gr.Textbox(label="Description", placeholder="Describe the video content")
            model = gr.Dropdown(allowed_models, value=allowed_models[0] if allowed_models else None, label="Model")
            use_rag = gr.Checkbox(label="Use RAG", value=False)
            use_visual_fix_code = gr.Checkbox(label="Use Visual Fix Code", value=False)
            use_context_learning = gr.Checkbox(label="Use Context Learning", value=False)
            max_retries = gr.Slider(1, 10, value=5, step=1, label="Max Retries")
            submit_btn = gr.Button("Generate Video", elem_id="submit-btn")
        with gr.Column():
            output_text = gr.Textbox(label="Status / Message", interactive=False)
            output_video = gr.Video(label="Generated Video")
    submit_btn.click(
        generate_video_gradio,
        inputs=[topic, description, model, use_rag, use_visual_fix_code, use_context_learning, max_retries],
        outputs=[output_text, output_video]
    )

demo.launch()
