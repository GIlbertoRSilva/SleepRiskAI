# SleepRiskAI

"Don't just track, predict. AI that forecasts sleep disorders and simulates how lifestyle changes improve your health."

## About The Project

SleepRiskAI goes beyond passive sleep tracking. It is an intelligent circadian health monitor that uses Machine Learning to analyze a user's biometric profile and lifestyle habits to predict the probability of developing sleep disorders (such as Insomnia or Sleep Apnea).

Unlike "black box" AI, SleepRiskAI is built on Explainability (XAI). It tells you why you are at risk and provides a Lifestyle Lab—a real-time simulation engine where you can test how changing your habits (e.g., sleeping 1 hour more, reducing stress) mathematically lowers your risk.

## Key Features

Predictive Risk Engine: Calculates the probability of sleep disorders based on 12+ clinical markers (BMI, Blood Pressure, Heart Rate, etc.).

Explainable AI (SHAP): Visualizes the "why" behind the prediction. See exactly which factors (e.g., High Stress, Low Activity) are driving your risk up.

The Lifestyle Lab: A "What-If" simulation environment. Tweak sliders to simulate lifestyle improvements and instantly see the projected risk reduction.

Smart Medical Report: Generates a professional, printable HTML report with clinical insights for your doctor.

## Built With

This project relies on a robust Python stack for Data Science and Web Development.

Streamlit - For the interactive web interface and custom CSS injection.

Scikit-Learn - For training the Random Forest Classifier model.

SHAP - For Explainable AI (XAI) and feature importance visualization.

Plotly - For interactive Radar Charts and Gauge Meters.

## Getting Started

Follow these steps to set up the project locally on your machine.

Prerequisites

Python 3.9 or higher

pip (Python package manager)

## Installation

Clone the repository

git clone [https://github.com/GIlbertoRSilva/SleepRiskAI.git](https://github.com/GIlbertoRSilva/SleepRiskAI.git)
cd SleepRiskAI


Create a Virtual Environment (Optional but Recommended)

python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate


Install Dependencies

pip install -r requirements.txt


Run the App

streamlit run src/sleep_risk_app.py


## How It Works

Data Input: The user enters demographics (Age, Gender, BMI) and lifestyle data (Sleep Duration, Stress Level, Steps).

Prediction: The pre-trained RandomForestClassifier processes the input vector and outputs a probability score.

Explanation: The SHAP explainer calculates the contribution of each feature to that score (Log-Odds).

Simulation: In the Lifestyle Lab, the app clones the user's data, applies the slider adjustments, and re-runs the prediction pipeline to show the delta (Risk Reduction).

## Contributing

Contributions make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request

## License

Distributed under the MIT License. See LICENSE for more information.

👨‍💻 Author

Gilberto R. Silva

GitHub Profile

<p align="center">
<i>Built with 💻 and ☕ for the Hackathon.</i>
