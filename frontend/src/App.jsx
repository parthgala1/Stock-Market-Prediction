import React, { useState } from "react";

const App = () => {
  const [formData, setFormData] = useState({
    open: "",
    high: "",
    low: "",
  });
  const [prediction, setPrediction] = useState(null);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch("http://localhost:3000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });
      const data = await response.json();
      let predict = data.prediction;
      setPrediction(predict[0 ]);
      console.log(typeof predict[0]);
    } catch (error) {
      console.error("Error fetching prediction:", error);
      setPrediction("Error: Unable to fetch prediction");
    }
  };

  const formatPrediction = (value) => {
    if (typeof value === "number") {
      return value.toFixed(2);
    }
    return value;
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col justify-center items-center p-4">
      <div className="bg-white rounded-lg shadow-xl p-8 max-w-md w-full">
        <h1 className="text-4xl text-gray-800 font-serif mb-2 text-center">
          Stock Prediction
        </h1>
        <p className="text-lg text-gray-600 mb-6 text-center">
          Using machine learning to predict stock prices
        </p>
        <form onSubmit={handleSubmit} className="space-y-4">
          {["open", "high", "low"].map((field) => (
            <div key={field}>
              <label
                htmlFor={field}
                className="block text-sm font-medium text-gray-700 capitalize"
              >
                {field} Price
              </label>
              <input
                type="number"
                id={field}
                name={field}
                value={formData[field]}
                onChange={handleInputChange}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
                required
              />
            </div>
          ))}
          <button
            type="submit"
            className="w-full bg-indigo-600 text-white py-2 px-4 rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 transition duration-150 ease-in-out"
          >
            Predict
          </button>
        </form>
        {prediction !== null && (
          <div className="mt-6 p-4 bg-gray-50 rounded-md">
            <p className="text-lg font-medium text-gray-800">
              Predicted Closing Price:
            </p>
            <p className="text-2xl font-bold text-indigo-600">
              {formatPrediction(prediction)}
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
