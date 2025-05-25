import streamlit as st
import requests
import os
import time
from pathlib import Path
from streamlit_extras.stylable_container import stylable_container
from streamlit_shadcn_ui import card

# --- Config ---
FASTAPI_URL = "http://127.0.0.1:8000/generate"
OUTPUT_DIR = Path("output")

# Initialize session state for tracking submission
if 'submitted' not in st.session_state:
    st.session_state.submitted = False
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'checking' not in st.session_state:
    st.session_state.checking = False

# --- Helper: Get demo videos ---
def get_demo_videos(max_demos=4):
    demos = []
    if OUTPUT_DIR.exists():
        for topic_dir in OUTPUT_DIR.iterdir():
            if topic_dir.is_dir():
                for file in topic_dir.iterdir():
                    if file.name.endswith('_combined.mp4'):
                        demos.append({
                            'topic': topic_dir.name.replace('_', ' ').title(),
                            'video': str(file)
                        })
    return demos[:max_demos]

def submit_request(topic, description, model, use_rag, use_visual_fix_code, use_context_learning, max_retries):
    st.session_state.submitted = True
    st.session_state.checking = True
    
    # Normalize topic for file path prediction
    file_prefix = topic.lower()
    import re
    file_prefix = re.sub(r'[^a-z0-9_]+', '_', file_prefix)
    st.session_state.video_path = os.path.join(str(OUTPUT_DIR), file_prefix, f"{file_prefix}_combined.mp4")

def check_video_ready():
    st.session_state.checking = True

# --- UI ---
st.set_page_config(page_title="Theory2Manim", layout="wide")

# Add custom CSS for card styling
st.markdown("""
<style>
.card-container {
    min-height: 280px;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

with stylable_container(
    key="header",
    css_styles="""
        {
            background: linear-gradient(90deg,#4F8EF7 0,#a1c4fd 100%);
            border-radius: 16px;
            padding: 2rem 1rem;
            margin-bottom: 2rem;
        }
    """
):
    st.markdown("""
    <h1 style='color:white; margin-bottom:0;'>Theory2Manim</h1>
    <div style='color:white; font-size:1.2em;'>Generate beautiful math/science explainer videos from a single prompt. Powered by LLMs, RAG, and Manim.</div>
    """, unsafe_allow_html=True)

# --- Demo Gallery ---
st.subheader("Demo Videos")
demos = get_demo_videos()
if demos:
    cols = st.columns(len(demos))
    for i, demo in enumerate(demos):
        with cols[i]:
            with stylable_container(key=f"card_{i}", css_styles="{min-height: 280px; margin-bottom: 1rem;}"):
                card(
                    title=demo['topic'],
                    content=st.video(demo['video'])
                )
else:
    st.info("No demo videos found in the output directory.")

st.markdown("---")
st.subheader("Try it yourself!")

# --- User Input/Output ---
col1, col2 = st.columns(2)
with col1:
    topic = st.text_input("Topic", placeholder="Enter topic, e.g. Chain Rule")
    description = st.text_area("Description", placeholder="Describe the video content")
    model = st.selectbox("Model", options=["gemini/gemini-2.0-flash-001", "gpt-4", "gemini-pro"], index=0)
    
    # Use standard Streamlit components
    use_rag = st.checkbox("Use RAG", value=False)
    use_visual_fix_code = st.checkbox("Use Visual Fix Code", value=False)
    use_context_learning = st.checkbox("Use Context Learning", value=False)
    
    # Replace shadcn slider with standard Streamlit slider
    max_retries = st.slider("Max Retries", min_value=1, max_value=10, value=5)
    
    # Replace shadcn button with standard Streamlit button
    with stylable_container(
        key="submit_button",
        css_styles="""
            {
                width: 100%;
                margin-top: 1rem;
            }
            button {
                width: 100%;
                background-color: #4F8EF7;
                color: white;
            }
        """
    ):
        submit = st.button("Generate Video", key="submit_btn", disabled=st.session_state.submitted and st.session_state.checking)

with col2:
    with stylable_container(key="output_container", css_styles="{padding: 1.5rem; border-radius: 8px; border: 1px solid #e0e0e0;}"):
        status_placeholder = st.empty()
        progress_placeholder = st.empty()
        video_placeholder = st.empty()

# Handle submission
if submit and not st.session_state.submitted:
    if not topic or not description:
        status_placeholder.warning("Please enter both topic and description.")
    else:
        # Mark as submitted and set up UI
        submit_request(topic, description, model, use_rag, use_visual_fix_code, use_context_learning, max_retries)
        status_placeholder.info("Submitting your request...")
        
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
            # Send request to backend
            resp = requests.post(FASTAPI_URL, json=payload, timeout=120)
            if resp.status_code == 200:
                data = resp.json()
                status_placeholder.success(f"Request submitted: {data.get('message', '')}")
                progress_placeholder.info("Video generation started. This may take several minutes.")
            else:
                status_placeholder.error(f"Error: {resp.text}")
                st.session_state.submitted = False
        except Exception as e:
            status_placeholder.error(f"Error: {str(e)}")
            progress_placeholder.info("Please make sure the backend server is running.")
            st.session_state.submitted = False

# Check if video is ready (either from button click or already submitted)
if st.session_state.submitted:
    # Add a check status button
    check_button = st.button("Check if video is ready", key="check_btn", on_click=check_video_ready)
    
    # If checking triggered or first submission
    if st.session_state.checking:
        if os.path.exists(st.session_state.video_path):
            # Video is ready
            progress_placeholder.success("Video generation completed!")
            video_placeholder.video(st.session_state.video_path)
            st.session_state.checking = False
            
            # Add a reset button to start over
            if st.button("Generate another video", key="reset_btn"):
                st.session_state.submitted = False
                st.session_state.checking = False
                st.session_state.video_path = None
                st.experimental_rerun()
        else:
            # Video not ready yet
            progress_placeholder.warning("Video is still being generated. Check back in a few minutes.")
            st.session_state.checking = False
