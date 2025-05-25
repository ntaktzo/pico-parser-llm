from transformers import pipeline
import torch
from langchain.llms import OpenAI

print("Torch version:", torch.__version__)
print("Using CUDA:", torch.cuda.is_available())

from transformers import pipeline
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

print(summarizer("This is a long text that needs to be shortened. " * 10)[0]['summary_text'])
