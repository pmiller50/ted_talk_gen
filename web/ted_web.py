import streamlit as st

from PIL import Image 

logo = Image.open('./images/ted_logo.png')

st.image(logo, caption='(This site is in no way associated with TED.)')
st.title('TED Talk Text Generation')

'''

In markdown
[text](http://www.ted.com)

![Image](./images/ted_logo.png)
This app allows you to generate text based on several compiled neural network machine learning models.

The models have been fed a body of text that comprises of the top 25 most viewed TED Talks (through Jan, 2021.)
         
'''


# https://github.com/pmiller50/ted_talk_gen/blob/main/transcripts/brain_sentence_case.txt