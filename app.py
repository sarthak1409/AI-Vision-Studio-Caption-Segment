import streamlit as st
from caption_model import ImageCaptioner
from segment_model import ImageSegmenter
from PIL import Image
import numpy as np
import io
import random
from streamlit_image_comparison import image_comparison

# setting up streamlit page so that we can actually see our work practically

st.set_page_config(page_title="AI Vision Studio: Caption & Segment", layout="wide")

# adding custom css for cool look and best UI
st.markdown("""
<style>
.stApp {
    background: linear-gradient(-45deg, #1e3c72, #2a5298, #6f86d6, #48c6ef, #00f2fe, #6a11cb, #2575fc);
    background-size: 500% 500%;
    animation: auroraFlow 20s ease infinite;
    color: #f8f9fa;
    font-family: "Poppins", sans-serif;
}
@keyframes auroraFlow {
    0% { background-position: 0% 50%; filter: brightness(0.9); }
    25% { background-position: 50% 100%; filter: brightness(1.1); }
    50% { background-position: 100% 50%; filter: brightness(0.95); }
    75% { background-position: 50% 0%; filter: brightness(1.05); }
    100% { background-position: 0% 50%; filter: brightness(0.9); }
}
.main {
    background-color: rgba(255, 255, 255, 0.10);
    backdrop-filter: blur(18px);
    border-radius: 22px;
    padding: 2.5rem;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.45);
    margin-top: 30px;
    font-size: 1.05rem;
}
h1, h2, h3, h4 {
    font-family: "Poppins", sans-serif;
    font-weight: 700;
    letter-spacing: 0.7px;
    color: #ffffff;
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
}
p, li, div {
    font-family: "Poppins", sans-serif;
    font-size: 1.05rem;
    color: #f0f0f0;
}
.stButton > button {
    background: linear-gradient(90deg, #6a11cb, #2575fc);
    color: white;
    border: none;
    padding: 0.9rem 1.8rem;
    border-radius: 16px;
    font-weight: 600;
    font-size: 1.05rem;
    box-shadow: 0px 4px 15px rgba(37, 117, 252, 0.4);
    transition: all 0.3s ease;
}
.stButton > button:hover {
    transform: scale(1.08);
    background: linear-gradient(90deg, #2575fc, #6a11cb);
    box-shadow: 0px 6px 20px rgba(37, 117, 252, 0.6);
}
.caption-box {
    background: rgba(255,255,255,0.18);
    padding: 20px;
    border-radius: 14px;
    margin-top: 12px;
    font-size: 1.1rem;
    color: #f9f9f9;
    border-left: 6px solid #48c6ef;
    box-shadow: 0px 4px 25px rgba(0,0,0,0.3);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f2027, #203a43, #2c5364, #4b6cb7, #182848);
    background-size: 400% 400%;
    animation: sidebarAurora 25s ease infinite;
    color: white;
    border-right: 2px solid rgba(255,255,255,0.15);
    box-shadow: 4px 0 18px rgba(0,0,0,0.35);
}
@keyframes sidebarAurora {
    0% { background-position: 0% 0%; }
    50% { background-position: 100% 100%; }
    100% { background-position: 0% 0%; }
}
[data-testid="stSidebar"] * {
    color: #ffffff !important;
}
.footer {
    text-align: center;
    margin-top: 45px;
    font-size: 1rem;
    color: #e3e3e3;
    letter-spacing: 0.3px;
}
</style>
""", unsafe_allow_html=True)

# just making the sidebar info and sliders like it will provide info and it is also user interactive
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=100)
st.sidebar.markdown("""
## ⚙️ **AI Vision Studio**
**Created by:** Sarthak Maddi  
**Internship:** Zidio Tech  
**Powered by:** PyTorch ⚡ + HuggingFace 🤗  

### Features:
- 🧠 Deep Caption Generation (BLIP + CLIP)
- 🎯 Smart Image Segmentation (Mask R-CNN)
- 💾 Downloadable Results
""")
st.sidebar.markdown("---")

threshold = st.sidebar.slider("Segmentation Confidence", 0.3, 0.95, 0.7, 0.05)
num_captions = st.sidebar.slider("Number of Caption Suggestions", 1, 5, 3)

if "captioner" not in st.session_state:
    st.session_state.captioner = ImageCaptioner()
if "segmenter" not in st.session_state:
    st.session_state.segmenter = ImageSegmenter(threshold=threshold)

st.markdown("<h1 style='text-align:center; font-size:46px;'>🧠 AI Vision Studio: Caption & Segment</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:20px;'>A Streamlit-based Web App integrating Image Captioning and Instance Segmentation.</p>", unsafe_allow_html=True)
st.markdown("---")

# uploading image now
uploaded_file = st.file_uploader("📸 Upload your image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=520)
    st.markdown("---")

    st.subheader("📝 Step 1: Deep Image Captioning")
    with st.spinner("Generating deep captions..."):
        captions = []
        for _ in range(num_captions):
            caption = st.session_state.captioner.generate_caption(uploaded_file)
            captions.append(caption)
        captions = list(set(captions))

    best_caption = random.choice(captions)
    st.markdown(f"<div class='caption-box'><b>🗣️ Suggested Caption:</b> {best_caption}</div>", unsafe_allow_html=True)

    if len(captions) > 1:
        with st.expander("💡 See More Suggestions"):
            for cap in captions:
                st.write(f"- {cap}")

    if st.button("🔄 Regenerate Caption"):
        with st.spinner("Regenerating caption..."):
            new_caption = st.session_state.captioner.generate_caption(uploaded_file)
            st.markdown(f"<div class='caption-box'><b>✨ New Caption:</b> {new_caption}</div>", unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("🎯 Step 2: Image Segmentation")
    with st.spinner("Performing segmentation..."):
        segmented_img = st.session_state.segmenter.segment_image(uploaded_file)

    segmented_pil = Image.fromarray(np.uint8(segmented_img))

    # showing comparison of both images so that by using slider user can see difference
    image_comparison(
        img1=image,
        img2=segmented_pil,
        label1="Original",
        label2="Segmented",
        width=700
    )

    buf = io.BytesIO()
    segmented_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="⬇️ Download Segmented Image",
        data=byte_im,
        file_name="segmented_output.png",
        mime="image/png",
    )

st.markdown("<p class='footer'>Developed by <b>Sarthak Maddi</b> | © 2025 Zidio Internship Project</p>", unsafe_allow_html=True)

# done by sarthak maddi for zidio internship project
