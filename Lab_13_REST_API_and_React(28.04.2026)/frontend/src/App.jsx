import { useEffect, useMemo, useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const fields = [
  ["alcohol", "Alcohol"],
  ["malic_acid", "Malic acid"],
  ["ash", "Ash"],
  ["alcalinity_of_ash", "Alcalinity"],
  ["magnesium", "Magnesium"],
  ["total_phenols", "Phenols"],
  ["flavanoids", "Flavanoids"],
  ["nonflavanoid_phenols", "Nonflavanoid"],
  ["proanthocyanins", "Proanthocyanins"],
  ["color_intensity", "Color"],
  ["hue", "Hue"],
  ["od280_od315_of_diluted_wines", "OD280/OD315"],
  ["proline", "Proline"]
];

const startValues = {
  alcohol: 13.2,
  malic_acid: 1.78,
  ash: 2.14,
  alcalinity_of_ash: 11.2,
  magnesium: 100,
  total_phenols: 2.65,
  flavanoids: 2.76,
  nonflavanoid_phenols: 0.26,
  proanthocyanins: 1.28,
  color_intensity: 4.38,
  hue: 1.05,
  od280_od315_of_diluted_wines: 3.4,
  proline: 1050
};

function App() {
  const [form, setForm] = useState(startValues);
  const [modelInfo, setModelInfo] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [examples, setExamples] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [status, setStatus] = useState("loading");
  const [error, setError] = useState("");

  const accuracy = useMemo(() => {
    if (!modelInfo?.accuracy) {
      return "-";
    }

    return modelInfo.accuracy.toFixed(4);
  }, [modelInfo]);

  useEffect(() => {
    async function loadData() {
      try {
        const [infoResponse, metricsResponse, examplesResponse] = await Promise.all([
          fetch(`${API_URL}/api/model/info`),
          fetch(`${API_URL}/api/model/metrics`),
          fetch(`${API_URL}/api/examples`)
        ]);

        if (!infoResponse.ok || !metricsResponse.ok || !examplesResponse.ok) {
          throw new Error("API response error");
        }

        const infoData = await infoResponse.json();
        const metricsData = await metricsResponse.json();
        const examplesData = await examplesResponse.json();

        setModelInfo(infoData);
        setMetrics(metricsData);
        setExamples(examplesData.examples);
        setStatus("ready");
      } catch (loadError) {
        setStatus("error");
        setError("FastAPI server is not available");
      }
    }

    loadData();
  }, []);

  function changeValue(name, value) {
    setForm((currentForm) => ({
      ...currentForm,
      [name]: Number(value)
    }));
  }

  function loadExample(example) {
    setForm(example.features);
    setPrediction(null);
  }

  async function predictClass(event) {
    event.preventDefault();
    setError("");

    try {
      const response = await fetch(`${API_URL}/api/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(form)
      });

      if (!response.ok) {
        throw new Error("Prediction error");
      }

      setPrediction(await response.json());
    } catch (predictError) {
      setError("Prediction request failed");
    }
  }

  return (
    <main className="page">
      <section className="topbar">
        <div>
          <p className="eyebrow">Lab 13</p>
          <h1>Wine REST Client</h1>
        </div>
        <div className={`status ${status}`}>{status}</div>
      </section>

      <section className="metrics">
        <article>
          <span>Model</span>
          <strong>{modelInfo?.model || "-"}</strong>
        </article>
        <article>
          <span>Dataset</span>
          <strong>{modelInfo?.dataset || "-"}</strong>
        </article>
        <article>
          <span>Accuracy</span>
          <strong>{accuracy}</strong>
        </article>
        <article>
          <span>Classes</span>
          <strong>{modelInfo?.classes?.length || "-"}</strong>
        </article>
      </section>

      <section className="workspace">
        <form className="panel form-panel" onSubmit={predictClass}>
          <div className="panel-head">
            <h2>Input Data</h2>
            <button type="submit">Predict</button>
          </div>

          <div className="examples">
            {examples.map((example) => (
              <button
                type="button"
                key={example.name}
                onClick={() => loadExample(example)}
              >
                {example.expected_class_name}
              </button>
            ))}
          </div>

          <div className="field-grid">
            {fields.map(([name, label]) => (
              <label key={name}>
                <span>{label}</span>
                <input
                  type="number"
                  step="0.01"
                  value={form[name]}
                  onChange={(event) => changeValue(name, event.target.value)}
                />
              </label>
            ))}
          </div>
        </form>

        <section className="panel result-panel">
          <div className="panel-head">
            <h2>Result</h2>
            <span>{prediction?.predicted_class_name || "waiting"}</span>
          </div>

          {prediction ? (
            <div className="prediction">
              <div className="class-box">
                <span>Predicted class</span>
                <strong>{prediction.predicted_class_name}</strong>
              </div>

              <div className="probabilities">
                {Object.entries(prediction.probabilities).map(([name, value]) => (
                  <div className="probability" key={name}>
                    <div>
                      <span>{name}</span>
                      <strong>{value.toFixed(4)}</strong>
                    </div>
                    <div className="bar">
                      <span style={{ width: `${value * 100}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="empty-state">No prediction</div>
          )}

          {metrics && (
            <div className="matrix">
              <h3>Confusion Matrix</h3>
              <div className="matrix-grid">
                {metrics.confusion_matrix.flat().map((value, index) => (
                  <span key={`${value}-${index}`}>{value}</span>
                ))}
              </div>
            </div>
          )}

          {error && <p className="error">{error}</p>}
        </section>
      </section>
    </main>
  );
}

export default App;
