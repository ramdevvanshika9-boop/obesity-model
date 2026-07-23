import gradio as gr
import pickle
import numpy as np

# Load the trained model
with open("obesity_knn_model.pkl", "rb") as f:
    model = pickle.load(f)

# Prediction function
def predict_obesity(age, gender, height, weight, family_history, physical_activity):
    # Convert categorical values
    gender = 1 if gender == "Male" else 0
    family_history = 1 if family_history == "Yes" else 0

    # Input array (must match the order used during training)
    features = np.array([[age, gender, height, weight,
                          family_history, physical_activity]])

    prediction = model.predict(features)[0]

    return f"Predicted Obesity Level: {prediction}"

# Gradio Interface
demo = gr.Interface(
    fn=predict_obesity,
    inputs=[
        gr.Number(label="Age"),
        gr.Radio(["Male", "Female"], label="Gender"),
        gr.Number(label="Height (meters)"),
        gr.Number(label="Weight (kg)"),
        gr.Radio(["Yes", "No"], label="Family History of Obesity"),
        gr.Number(label="Physical Activity (Hours/Week)")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Developed by Vanshika",
    description="Roll No.: 241404\n\nObesity Prediction using KNN Model"
)

if __name__ == "__main__":
    demo.launch()
