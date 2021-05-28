import onnxruntime as onnx_rt
import torch
import numpy as np
from PIL import Image
import cv2
import os
import pathlib
import streamlit as st
from utils import non_max_suppression


# Constant vars
MODEL_PATH = os.path.join(pathlib.Path(__file__).parent.absolute(), "best.onnx")
IMG_SIZE = 256
SAMPLE_IMG = os.path.join(pathlib.Path(__file__).parent.absolute(), "sample.gif")
favicon = Image.open(os.path.join(pathlib.Path(__file__).parent.absolute(), "frame_001371.jpg"))
st.set_page_config(page_title='Pastai', page_icon="🍉", layout='wide', initial_sidebar_state='auto')


@st.cache
def load_model():
    return onnx_rt.InferenceSession(MODEL_PATH)


@st.cache
def load_sample_image():
    return Image.open(SAMPLE_IMG)


def process_img(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.asarray(img).transpose((2, 0, 1))/255.0
    img = np.expand_dims(img, 0).astype(np.float32)
    return img


def post_process_img(img, bboxes):
    for bbox in bboxes:
        img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), thickness=1)
    return img


def predict(model, img, conf_thres, iou_thres):
    img_processed = process_img(img)
    preds = model.run(None, {"images": img_processed})[0]
    preds = torch.tensor(preds)
    preds = non_max_suppression(preds, conf_thres=conf_thres, iou_thres=iou_thres)
    if len(preds) == 1: preds = preds[0]
    preds = [item.squeeze().numpy() for item in preds]
    return preds


def main():

    st.sidebar.header("Prediction calibration")
    st.sidebar.write("""First upload an image using the file upload widget on the right. Then, using the below sliders, change the bounding boxes 
    prediction thresholds to see how model prediction are changing.
     The lower the thresholds, the more (uncertain) bounding boxes the model will yield.""")
    conf_thres = st.sidebar.slider("Confidence Threshold", min_value=0., max_value=1.0, value=0.25, step=0.05)
    iou_thres = st.sidebar.slider("Intersection Over Uninon Threshold", min_value=0., max_value=1.0, value=0.45, step=0.05)

    # st.components.v1.html("""<h1 style="text-align:center; font-family: system-ui;">Pastai : watermelon detection in the wild</h1>""")
    st.write("# Pastai : watermelon detection in the wild")
    
    st.write("## Gimme a 🍉, I'll give ya bbox 📦! ┌( ಠ‿ಠ )┘")
    st.info("(Disclaimer! bbox = bounding boxes, not bouygues telecom box)")
    st.info("Waiting for you to upload a 🍉 image below ⬇️⬇️⬇️")
    with st.spinner("Loading the 🍉 detector ..."):
        model = load_model()
    file_uploader = st.file_uploader("", type=["jpg", "png"])
    col1, col2, col3 = st.beta_columns([1,6,1])
    if file_uploader is None:
        with col1:
            st.write("")
        with col2:
            st.write("Displaying sample image")
            st.image(SAMPLE_IMG)
        with col3:
            st.write("")

    else:
        img = Image.open(file_uploader)
        w, h = img.size
        new_h = 400
        new_w = int(new_h*w/h)
        preds = predict(model, img, conf_thres, iou_thres)
        img = np.array(img.resize((IMG_SIZE, IMG_SIZE)))
        img = post_process_img(img, preds)
        
        with col1:
            st.write("")
        with col2:
            st.image(Image.fromarray(img).resize((new_w, new_h)))
        with col3:
            st.write("")
    
    with open(os.path.join(pathlib.Path(__file__).parent.parent.absolute(), "README.md")) as f:
        readme = f.read()
    st.markdown(readme, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
