const numericFields = [
  "original_publication_year", "books_count", "ratings_count",
  "work_ratings_count", "work_text_reviews_count"
];
const descriptions = {
  regression: "Estimate the novel's average reader rating on a scale of 0 to 5.",
  classification: "Predict whether the novel belongs to the Low, Medium, or High rating group."
};
let appInfo = null;
let activeTask = "regression";
const byId = id => document.getElementById(id);

function makeCell(tag, value) {
  const node = document.createElement(tag);
  node.textContent = value;
  return node;
}
function createTable(headers, rows, bestIndex = -1) {
  const table = document.createElement("table");
  table.className = "score-table";
  const head = document.createElement("thead");
  const tr = document.createElement("tr");
  headers.forEach(item => tr.appendChild(makeCell("th", item)));
  head.appendChild(tr);
  table.appendChild(head);
  const body = document.createElement("tbody");
  rows.forEach((row, index) => {
    const line = document.createElement("tr");
    if (index === bestIndex) line.className = "best-row";
    row.forEach(item => line.appendChild(makeCell("td", item)));
    body.appendChild(line);
  });
  table.appendChild(body);
  return table;
}
function showError(message) {
  const result = byId("result");
  result.replaceChildren();
  const p = makeCell("p", message);
  p.className = "error-text";
  result.appendChild(p);
}
function readInput() {
  const data = {};
  numericFields.forEach(field => {
    const value = byId(field).value.trim();
    if (value === "" || !Number.isFinite(Number(value))) {
      throw new Error(`Please enter a valid value for ${field.replaceAll("_", " ")}.`);
    }
    data[field] = Number(value);
  });
  data.language_code = byId("language_code").value || "unknown";
  return data;
}
async function requestApi(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data.detail;
    throw new Error(typeof detail === "string" ? detail : "The server could not process this request.");
  }
  return data;
}
function renderSingle(data) {
  const card = document.createElement("div");
  card.className = "result-card";
  card.appendChild(makeCell("div", data.name)).className = "result-kicker";
  if (activeTask === "regression") {
    const number = makeCell("div", `${Number(data.predicted_average_rating).toFixed(2)} / 5`);
    number.className = "result-number";
    card.appendChild(number);
    card.appendChild(makeCell("div", "Estimated average reader rating")).className = "result-sub";
  } else {
    const number = makeCell("div", data.category);
    number.className = "result-number";
    card.appendChild(number);
    card.appendChild(makeCell("div", `Rating band: ${data.range}`)).className = "result-sub";
    if (data.probabilities) {
      const rows = Object.entries(data.probabilities).map(([label, value]) =>
        [label, `${(value * 100).toFixed(1)}%`]);
      card.appendChild(createTable(["Rating group", "Model probability"], rows));
    }
  }
  byId("result").replaceChildren(card);
}
function renderComparison(results) {
  const card = document.createElement("div");
  card.className = "result-card";
  card.appendChild(makeCell("div", "MODEL COMPARISON")).className = "result-kicker";
  let rows;
  if (activeTask === "regression") {
    rows = results.map(item => [item.name, `${item.predicted_average_rating.toFixed(2)} / 5`]);
  } else {
    rows = results.map(item => {
      const confidence = item.probabilities
        ? `${(Math.max(...Object.values(item.probabilities)) * 100).toFixed(1)}%` : "Not available";
      return [item.name, item.category, confidence];
    });
  }
  const headers = activeTask === "regression"
    ? ["Machine-learning model", "Predicted rating"]
    : ["Machine-learning model", "Predicted group", "Highest probability"];
  card.appendChild(createTable(headers, rows));
  byId("result").replaceChildren(card);
}
function renderMetrics(task, elementId) {
  const metrics = appInfo.metrics[task] || [];
  const target = byId(elementId);
  target.replaceChildren();
  if (!metrics.length) {
    target.textContent = "No model scores are available yet.";
    return;
  }
  const regression = task === "regression";
  const bestIndex = metrics.reduce((best, current, index) => {
    const score = regression ? current.r2 : current.accuracy;
    const bestScore = regression ? metrics[best].r2 : metrics[best].accuracy;
    return score > bestScore ? index : best;
  }, 0);
  const headers = regression ? ["Model", "R²", "MAE", "RMSE"] : ["Model", "Accuracy", "F1 score"];
  const rows = metrics.map(item => regression
    ? [item.name, item.r2.toFixed(3), item.mae.toFixed(3), item.rmse.toFixed(3)]
    : [item.name, item.accuracy.toFixed(3), item.f1.toFixed(3)]);
  target.appendChild(createTable(headers, rows, bestIndex));
}
function setTask(task) {
  activeTask = task;
  document.querySelectorAll(".tab").forEach(button => {
    const selected = button.dataset.task === task;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-selected", String(selected));
  });
  byId("task-description").textContent = descriptions[task];
  const select = byId("algorithm");
  select.replaceChildren();
  select.add(new Option("Compare all models", "all"));
  Object.entries(appInfo.models[task] || {}).forEach(([key, name]) => {
    select.add(new Option(name, key));
  });
  byId("result").replaceChildren();
}
async function initialize() {
  try {
    const response = await fetch("/api/info");
    const info = await response.json();
    if (!response.ok) throw new Error(info.detail || "Could not load trained models.");
    appInfo = info;
    byId("book-count").textContent = Number(info.dataset_rows).toLocaleString();
    const count = Object.keys(info.models.regression).length + Object.keys(info.models.classification).length;
    byId("model-count").textContent = count;
    const language = byId("language_code");
    language.replaceChildren(new Option("Unknown", "unknown"));
    (info.languages || []).forEach(code => {
      if (code !== "unknown") language.add(new Option(code, code));
    });
    renderMetrics("regression", "regression-metrics");
    renderMetrics("classification", "classification-metrics");
    setTask("regression");
  } catch (error) {
    byId("notice").textContent = `${error.message} Open the Colab training notebook, run all cells, and then start the app again.`;
    byId("notice").classList.remove("hidden");
    byId("regression-metrics").textContent = "Model information unavailable.";
    byId("classification-metrics").textContent = "Model information unavailable.";
  }
}
document.querySelectorAll(".tab").forEach(button => {
  button.addEventListener("click", () => { if (appInfo) setTask(button.dataset.task); });
});
byId("prediction-form").addEventListener("submit", async event => {
  event.preventDefault();
  if (!appInfo) return;
  let payload;
  try { payload = readInput(); } catch (error) { showError(error.message); return; }
  const button = byId("predict-button");
  button.disabled = true;
  button.textContent = "Working…";
  try {
    const selected = byId("algorithm").value;
    if (selected === "all") {
      const response = await requestApi(`/api/compare/${activeTask}`, payload);
      renderComparison(response.results);
    } else {
      const response = await requestApi(`/api/predict/${activeTask}`, {...payload, algorithm: selected});
      renderSingle(response);
    }
  } catch (error) {
    showError(error.message);
  } finally {
    button.disabled = false;
    button.innerHTML = 'Generate prediction <span>↗</span>';
  }
});
initialize();
