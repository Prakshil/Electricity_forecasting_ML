from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# 1. Initialize the FastAPI app
app = FastAPI(title="Electricity Consumption Forecaster", description="API to predict electricity usage in kWh")

# 2. Load the saved model upon startup
try:
    with open("electricity_forecaster.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None
    print("Warning: electricity_forecaster.pkl not found. Please run the notebook first to save the model.")

# 3. Define the expected input using Pydantic
# These must match the features used during training: 
# ["hour", "day", "weekday", "month", "lag1", "lag2", "lag3", "rolling_mean"]
class ForecastRequest(BaseModel):
    hour: int
    day: int
    weekday: int
    month: int
    lag1: float
    lag2: float
    lag3: float
    rolling_mean: float
    
    # Adding an example so the dashboard is easy to test
    class Config:
        schema_extra = {
            "example": {
                "hour": 14,
                "day": 15,
                "weekday": 2,
                "month": 6,
                "lag1": 250.5,
                "lag2": 245.0,
                "lag3": 240.2,
                "rolling_mean": 245.23
            }
        }

# 4. Create the prediction endpoint
@app.post("/predict")
def predict_electricity(request: ForecastRequest):
    if model is None:
        return {"error": "Model not loaded on server."}
    
    # Convert the requested data into a pandas DataFrame 
    # (Scikit-learn pipelines usually expect DataFrames with the exact column names)
    input_data = pd.DataFrame([request.model_dump()])
    
    # Make the prediction
    prediction = model.predict(input_data)[0]
    
    # Return the result
    return {
        "predicted_energy_kwh": round(float(prediction), 2),
        "input_features": request.model_dump()
    }

from fastapi.responses import HTMLResponse

# Web Dashboard endpoint
@app.get("/", response_class=HTMLResponse)
def read_root():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Electricity Forecaster Dashboard</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-slate-100 min-h-screen text-slate-800 font-sans">
        <div class="max-w-4xl mx-auto py-10 px-4">
            <div class="bg-white rounded-2xl shadow-xl overflow-hidden">
                <div class="bg-blue-600 text-white py-6 px-8">
                    <h1 class="text-3xl font-bold">⚡ Electricity Forecaster</h1>
                    <p class="mt-2 text-blue-100">FastAPI Powered Dashboard</p>
                </div>
                
                <div class="p-8">
                    <form id="predictionForm" class="space-y-6">
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                            
                            <!-- Temporal Features -->
                            <div class="bg-slate-50 p-4 rounded-xl border border-slate-200">
                                <h3 class="font-semibold text-lg mb-4 text-slate-700 border-b pb-2">Time Features</h3>
                                <div class="grid grid-cols-2 gap-4">
                                    <div>
                                        <label class="block text-sm font-medium mb-1">Hour (0-23)</label>
                                        <input type="number" id="hour" value="14" class="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500">
                                    </div>
                                    <div>
                                        <label class="block text-sm font-medium mb-1">Day (1-31)</label>
                                        <input type="number" id="day" value="15" class="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500">
                                    </div>
                                    <div>
                                        <label class="block text-sm font-medium mb-1">Weekday (0-6)</label>
                                        <input type="number" id="weekday" value="2" class="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500">
                                    </div>
                                    <div>
                                        <label class="block text-sm font-medium mb-1">Month (1-12)</label>
                                        <input type="number" id="month" value="6" class="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500">
                                    </div>
                                </div>
                            </div>

                            <!-- Historical Features -->
                            <div class="bg-slate-50 p-4 rounded-xl border border-slate-200">
                                <h3 class="font-semibold text-lg mb-4 text-slate-700 border-b pb-2">Historical Usage (kWh)</h3>
                                <div class="space-y-4">
                                    <div>
                                        <label class="block text-sm font-medium mb-1">Lag 1 (1 hour ago)</label>
                                        <input type="number" step="0.1" id="lag1" value="250.5" class="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500">
                                    </div>
                                    <div class="grid grid-cols-2 gap-4">
                                        <div>
                                            <label class="block text-sm font-medium mb-1">Lag 2</label>
                                            <input type="number" step="0.1" id="lag2" value="245.0" class="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500">
                                        </div>
                                        <div>
                                            <label class="block text-sm font-medium mb-1">Lag 3</label>
                                            <input type="number" step="0.1" id="lag3" value="240.2" class="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500">
                                        </div>
                                    </div>
                                    <div>
                                        <label class="block text-sm font-medium mb-1">Rolling Mean (3-hour)</label>
                                        <input type="number" step="0.1" id="rolling_mean" value="245.2" class="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500">
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div class="pt-4">
                            <button type="submit" class="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded-xl transition duration-200">
                                Predict Energy Consumption
                            </button>
                        </div>
                    </form>

                    <!-- Result Card -->
                    <div id="resultCard" class="mt-8 bg-green-50 border border-green-200 rounded-xl p-6 hidden">
                        <p class="text-green-700 font-medium mb-1">Forecast Result</p>
                        <div class="flex items-baseline space-x-2">
                            <span id="predictionValue" class="text-4xl font-bold text-green-800">0.00</span>
                            <span class="text-xl text-green-700">kWh</span>
                        </div>
                    </div>
                    
                    <!-- Error Box -->
                    <div id="errorBox" class="mt-8 bg-red-50 border border-red-200 text-red-700 rounded-xl p-4 hidden">
                        <p id="errorMessage"></p>
                    </div>
                </div>
            </div>
        </div>

        <script>
            document.getElementById('predictionForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const data = {
                    hour: parseInt(document.getElementById('hour').value),
                    day: parseInt(document.getElementById('day').value),
                    weekday: parseInt(document.getElementById('weekday').value),
                    month: parseInt(document.getElementById('month').value),
                    lag1: parseFloat(document.getElementById('lag1').value),
                    lag2: parseFloat(document.getElementById('lag2').value),
                    lag3: parseFloat(document.getElementById('lag3').value),
                    rolling_mean: parseFloat(document.getElementById('rolling_mean').value)
                };

                const resultCard = document.getElementById('resultCard');
                const errorBox = document.getElementById('errorBox');
                const predValue = document.getElementById('predictionValue');

                resultCard.classList.add('hidden');
                errorBox.classList.add('hidden');

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify(data)
                    });

                    const result = await response.json();

                    if (!response.ok || result.error) {
                        document.getElementById('errorMessage').innerText = result.error || "An error occurred";
                        errorBox.classList.remove('hidden');
                    } else {
                        predValue.innerText = result.predicted_energy_kwh.toFixed(2);
                        resultCard.classList.remove('hidden');
                    }
                } catch (error) {
                    document.getElementById('errorMessage').innerText = error.message;
                    errorBox.classList.remove('hidden');
                }
            });
        </script>
    </body>
    </html>
    """
    return html_content
