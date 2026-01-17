import gradio as gr
import pandas as pd
import pickle
import numpy as np

# load the model
with open('laptop_rf_pipeline.pkl',"rb") as f:
    model=pickle.load(f)

# main logic
def predict_price(
    brand, name, processor, CPU,
    Ram, Ram_type, ROM, ROM_type,
    GPU, OS, display_size,
    resolution, warranty, spec_rating
):

    # split resolution
    resolution_width, resolution_height = map(int, resolution.split("x"))

    input_df = pd.DataFrame([{
        "brand": brand,
        "name": name,
        "spec_rating": spec_rating,
        "processor": processor,
        "CPU": CPU,
        "Ram": Ram,
        "Ram_type": Ram_type,
        "ROM": ROM,
        "ROM_type": ROM_type,
        "GPU": GPU,
        "display_size": display_size,
        "resolution_width": resolution_width,
        "resolution_height": resolution_height,
        "OS": OS,
        "warranty": warranty
    }])

    prediction = model.predict(input_df)[0]
    return float(np.clip(prediction, 0, 500000))
    
# interface

# define inputs 

inputs = [
    gr.Dropdown(["Apple", "Dell", "HP", "Lenovo", "Asus", "Acer", "MSI", "Samsung"], label="Brand"),
    gr.Textbox(label="Laptop Model Name"),

    gr.Textbox(label="Processor"),
    gr.Textbox(label="CPU (Cores & Threads)"),

    gr.Number(label="RAM (GB)", precision=0),
    gr.Dropdown(["DDR4", "DDR5", "LPDDR4", "LPDDR4X", "LPDDR5"], label="RAM Type"),

    gr.Number(label="Storage Size (GB)", precision=0),
    gr.Dropdown(["SSD", "Hard-Disk"], label="Storage Type"),

    gr.Textbox(label="GPU"),
    gr.Dropdown(["Windows 11 OS", "Windows 10 OS", "Mac OS", "Android 11 OS", "DOS OS"], label="OS"),

    gr.Number(label="Display Size (inches)"),
    gr.Dropdown(["1366x768", "1920x1080", "2560x1600", "3200x2000"], label="Resolution"),

    gr.Number(label="Warranty (Years)", precision=0),
    gr.Number(label="Specification Rating")
]

app = gr.Interface(
    fn=predict_price,
    inputs=inputs,
    outputs=gr.Number(label="Predicted Price"),
    title="Laptop Price Predictor"
)

# launch
app.launch(share=True)