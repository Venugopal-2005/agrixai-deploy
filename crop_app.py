import tkinter as tk
from tkinter import messagebox
import pandas as pd
import shap
import numpy as np
import joblib

# ----------------------------
# LOAD TRAINED MODEL FILES
# ----------------------------

try:
    model = joblib.load("model/yield_model.pkl")
    encoders = joblib.load("model/encoders.pkl")
    y = joblib.load("model/target_values.pkl")
except Exception as e:
    print("Error loading model files:", e)
    exit()

explainer = shap.TreeExplainer(model)

# ----------------------------
# PREDICTION FUNCTION
# ----------------------------

def predict_crop():
    try:
        # Validate inputs
        if (not crop_entry.get() or
            not season_entry.get() or
            not state_entry.get() or
            not area_entry.get() or
            not rainfall_entry.get() or
            not fertilizer_entry.get() or
            not pesticide_entry.get()):
            
            messagebox.showerror("Input Error", "Please fill all fields.")
            return

        farmer_input = {
            "Crop": crop_entry.get(),
            "Crop_Year": 2020,   # 🔥 Automatically set default year
            "Season": season_entry.get(),
            "State": state_entry.get(),
            "Area": float(area_entry.get()),
            "Annual_Rainfall": float(rainfall_entry.get()),
            "Fertilizer": float(fertilizer_entry.get()),
            "Pesticide": float(pesticide_entry.get())
        }

        input_df = pd.DataFrame([farmer_input])

        # Clean categorical text
        for col in ["Crop", "Season", "State"]:
            input_df[col] = input_df[col].astype(str).str.strip().str.title()

        # Encode categorical
        for col in ["Crop", "Season", "State"]:
            input_df[col] = encoders[col].transform(input_df[col])

        # Match feature order
        input_df = input_df[model.feature_names_in_]

        # Predict
        predicted_yield = model.predict(input_df)[0]

        average_yield = y.mean()
        yield_status = "HIGH" if predicted_yield >= average_yield else "LOW"

        # SHAP explanation
        shap_values = explainer.shap_values(input_df)[0]
        feature_names = input_df.columns

        positive = []
        negative = []

        for feature, value in zip(feature_names, shap_values):
            if value > 0:
                positive.append(feature)
            elif value < 0:
                negative.append(feature)

        result_text = "\n=============================\n"
        result_text += "Crop Yield Prediction Result\n"
        result_text += "=============================\n\n"
        result_text += f"Predicted Yield: {round(predicted_yield,2)}\n"
        result_text += f"Yield Status: {yield_status}\n\n"

        if yield_status == "HIGH":
            result_text += "Yield is high mainly because:\n"
            for f in positive:
                result_text += f"+ {f}\n"
        else:
            result_text += "Yield is low mainly because:\n"
            for f in negative:
                result_text += f"- {f}\n"

        result_text += "\nSuggestions:\n"

        for f in negative:
            if f == "Fertilizer":
                result_text += "• Consider adjusting fertilizer levels\n"
            elif f == "Pesticide":
                result_text += "• Improve pest management practices\n"
            elif f == "Area":
                result_text += "• Optimize land usage\n"
            elif f == "Annual_Rainfall":
                result_text += "• Improve irrigation planning\n"
            else:
                result_text += f"• Review {f}\n"

        result_label.config(text=result_text)

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values.")
    except Exception as e:
        messagebox.showerror("Error", str(e))


# ----------------------------
# GUI DESIGN
# ----------------------------

root = tk.Tk()
root.title("XAI Crop Yield Prediction System")
root.geometry("600x700")

tk.Label(root, text="Crop").pack()
crop_entry = tk.Entry(root)
crop_entry.pack()

tk.Label(root, text="Season").pack()
season_entry = tk.Entry(root)
season_entry.pack()

tk.Label(root, text="State").pack()
state_entry = tk.Entry(root)
state_entry.pack()

tk.Label(root, text="Area (hectares)").pack()
area_entry = tk.Entry(root)
area_entry.pack()

tk.Label(root, text="Annual Rainfall (mm)").pack()
rainfall_entry = tk.Entry(root)
rainfall_entry.pack()

tk.Label(root, text="Fertilizer (kg)").pack()
fertilizer_entry = tk.Entry(root)
fertilizer_entry.pack()

tk.Label(root, text="Pesticide (kg)").pack()
pesticide_entry = tk.Entry(root)
pesticide_entry.pack()

tk.Button(root, text="Predict Yield", command=predict_crop).pack(pady=15)

result_label = tk.Label(root, text="", justify="left", wraplength=550)
result_label.pack()

root.mainloop()
