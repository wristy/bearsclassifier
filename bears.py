import streamlit as st
from fastai.vision.all import *
from fastai.vision.widgets import *
import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

learner = load_learner('export.pkl')


class Predict:
    def __init__(self, filename):
        self.learn_inference = learner
        self.img = self.get_image()
        if self.img is not None:
            self.display_output()
            self.get_prediction()

    @staticmethod
    def get_image():
        uploaded_file = st.file_uploader("Upload an image of a black or grizzly bear.", type=['png', 'jpeg', 'jpg'])
        if uploaded_file is not None:
            return PILImage.create((uploaded_file))
        return None

    def display_output(self):
        st.image(self.img.to_thumb(500, 500), caption='Uploaded Image')

    def get_prediction(self):
        if st.button('Classify'):
            pred, pred_idx, probs = self.learn_inference.predict(self.img)
            st.write(f'Prediction: {pred} Probability: {probs[pred_idx]:.04f}')
        else:
            st.write('Classification result will appear here')


if __name__ == "__main__":

    file_name = 'export.pkl'
    predictor = Predict(file_name)
