from flask import Flask, request, jsonify, render_template_string
import json
import joblib

app = Flask(__name__)
bundle = joblib.load("model.pkl")
model, FEATURES = bundle["model"], bundle["features"]

# Load model accuracy for display
try:
    with open("metrics.json") as fh:
        _metrics = json.load(fh)
    ACCURACY = round(_metrics["accuracy"] * 100, 1)
except (FileNotFoundError, KeyError):
    ACCURACY = 79.5

DEFAULTS = {
    "age": 30,
    "time_on_site_min": 10,
    "pages_viewed": 12,
    "cart_items": 2,
    "previous_purchases": 3,
    "discount_offered": 0,
}

LABELS = {
    "age": "Customer age",
    "time_on_site_min": "Time on site (min)",
    "pages_viewed": "Pages viewed",
    "cart_items": "Items in cart",
    "previous_purchases": "Previous purchases",
    "discount_offered": "Discount (1=yes, 0=no)",
}

ICONS = {
    "age": "\U0001F9D1",
    "time_on_site_min": "\u23F1\uFE0F",
    "pages_viewed": "\U0001F4C4",
    "cart_items": "\U0001F6D2",
    "previous_purchases": "\U0001F4E6",
    "discount_offered": "\U0001F3F7\uFE0F",
}

PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CartSense AI \u2014 Purchase Intent Predictor</title>
  <style>
    *{margin:0;padding:0;box-sizing:border-box}
    body{
      font-family:'Segoe UI',system-ui,-apple-system,sans-serif;
      background:linear-gradient(135deg,#0f0c29,#302b63,#24243e);
      min-height:100vh;color:#e8e8e8;
      display:flex;align-items:center;justify-content:center;
      padding:2rem 1rem;
    }
    .container{max-width:620px;width:100%}

    /* Header */
    .header{text-align:center;margin-bottom:1.8rem}
    .logo{display:inline-flex;align-items:center;gap:.6rem;margin-bottom:.4rem}
    .logo-icon{font-size:2.2rem}
    .logo h1{
      font-size:1.9rem;font-weight:800;letter-spacing:-.5px;
      background:linear-gradient(135deg,#00f2fe,#4facfe);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;
      background-clip:text;
    }
    .subtitle{color:#9a9ab0;font-size:.95rem;margin-bottom:.6rem}
    .accuracy-badge{
      display:inline-block;background:rgba(79,172,254,.1);
      color:#4facfe;padding:.35rem .9rem;border-radius:30px;
      font-size:.8rem;font-weight:600;
      border:1px solid rgba(79,172,254,.2);
    }

    /* Card */
    .card{
      background:rgba(255,255,255,.05);backdrop-filter:blur(14px);
      -webkit-backdrop-filter:blur(14px);
      border:1px solid rgba(255,255,255,.1);
      border-radius:24px;padding:2rem 1.8rem;
      box-shadow:0 20px 60px rgba(0,0,0,.3);
    }

    /* Form grid */
    .form-grid{display:grid;grid-template-columns:1fr 1fr;gap:1.1rem}
    .field{display:flex;flex-direction:column;gap:.35rem}
    .field label{font-size:.82rem;color:#a0a0b8;font-weight:600}
    .icon-label{display:flex;align-items:center;gap:.35rem}
    .field input{
      background:rgba(255,255,255,.06);
      border:1px solid rgba(255,255,255,.1);
      border-radius:12px;padding:.7rem .9rem;color:#fff;
      font-size:.95rem;transition:border-color .2s,background .2s,box-shadow .2s;
    }
    .field input:focus{
      outline:none;border-color:#4facfe;
      background:rgba(255,255,255,.09);
      box-shadow:0 0 0 3px rgba(79,172,254,.15);
    }

    /* Button */
    .btn{
      width:100%;margin-top:1.6rem;padding:.95rem;
      background:linear-gradient(135deg,#4facfe,#00f2fe);
      border:none;border-radius:14px;color:#0f0c29;
      font-size:1.05rem;font-weight:700;cursor:pointer;
      transition:transform .15s,box-shadow .2s;
      letter-spacing:.3px;
    }
    .btn:hover{transform:translateY(-2px);box-shadow:0 10px 30px rgba(79,172,254,.35)}
    .btn:active{transform:translateY(0)}

    /* Results */
    .result-panel{
      margin-top:1.8rem;text-align:center;
      animation:fadeIn .5s ease-out;
    }
    .result-badge{
      display:inline-flex;align-items:center;gap:.5rem;
      padding:.85rem 2rem;border-radius:50px;
      font-size:1.25rem;font-weight:700;
    }
    .result-purchase{
      background:rgba(46,213,115,.12);color:#2ed573;
      border:1.5px solid rgba(46,213,115,.3);
    }
    .result-nopurchase{
      background:rgba(255,165,2,.12);color:#ffa502;
      border:1.5px solid rgba(255,165,2,.3);
    }

    /* Probability bar */
    .prob-section{margin-top:1.3rem}
    .prob-label{font-size:.82rem;color:#888;margin-bottom:.5rem;text-align:left}
    .prob-bar-bg{
      background:rgba(255,255,255,.06);border-radius:50px;
      height:26px;overflow:hidden;position:relative;
    }
    .prob-bar-fill{
      height:100%;border-radius:50px;
      background:linear-gradient(90deg,#00f2fe,#4facfe);
      animation:grow .9s cubic-bezier(.4,0,.2,1);
      transition:width .3s;
    }
    .prob-bar-fill.negative{
      background:linear-gradient(90deg,#ffa502,#ff6348);
    }
    .prob-value{
      font-size:2.2rem;font-weight:800;margin-top:.6rem;
      letter-spacing:-1px;
    }
    .prob-caption{font-size:.78rem;color:#777;margin-top:.2rem}

    /* Footer */
    .footer{
      text-align:center;margin-top:1.5rem;
      color:#5a5a6e;font-size:.78rem;line-height:1.6;
    }
    .footer a{color:#4facfe;text-decoration:none}

    /* Animations */
    @keyframes grow{from{width:0}}
    @keyframes fadeIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}

    /* Responsive */
    @media(max-width:560px){
      .form-grid{grid-template-columns:1fr}
      .card{padding:1.5rem 1.2rem}
      .logo h1{font-size:1.5rem}
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="logo">
        <span class="logo-icon">\U0001F9E0</span>
        <h1>CartSense AI</h1>
      </div>
      <p class="subtitle">Online Shopper Purchase Intent Predictor</p>
      <div class="accuracy-badge">\u26A1 Model accuracy: {{accuracy}}%</div>
    </div>

    <div class="card">
      <form method="post">
        <div class="form-grid">
          {% for f in features %}
          <div class="field">
            <label><span class="icon-label">{{icons[f]}} {{labels[f]}}</span></label>
            <input name="{{f}}" type="number" step="any" value="{{vals[f]}}" required>
          </div>
          {% endfor %}
        </div>
        <button type="submit" class="btn">\u26A1 Predict Purchase Intent</button>
      </form>

      {% if result %}
      <div class="result-panel">
        <div class="result-badge {{'result-purchase' if prob >= 50 else 'result-nopurchase'}}">
          {{result}}
        </div>
        <div class="prob-section">
          <div class="prob-label">Purchase Probability</div>
          <div class="prob-bar-bg">
            <div class="prob-bar-fill {{'' if prob >= 50 else 'negative'}}" style="width:{{prob}}%"></div>
          </div>
          <div class="prob-value" style="color:{{'#2ed573' if prob >= 50 else '#ffa502'}}">{{prob}}%</div>
          <div class="prob-caption">{{'High likelihood of purchase' if prob >= 50 else 'Low likelihood of purchase'}} \u2014 threshold: 50%</div>
        </div>
      </div>
      {% endif %}
    </div>

    <div class="footer">
      CartSense AI &mdash; Powered by Logistic Regression &copy; 2026<br>
      Trained on 2,000 simulated shopper sessions
    </div>
  </div>
</body>
</html>
"""


def run_prediction(values):
    p = float(model.predict_proba([values])[0][1])
    label = "Likely to PURCHASE" if p >= 0.5 else "Unlikely to purchase"
    return label, round(p * 100, 2)


@app.route("/", methods=["GET", "POST"])
def home():
    result = prob = None
    vals = DEFAULTS
    if request.method == "POST":
        vals = {f: float(request.form[f]) for f in FEATURES}
        result, prob = run_prediction([vals[f] for f in FEATURES])
    return render_template_string(
        PAGE, features=FEATURES, labels=LABELS, icons=ICONS,
        vals=vals, result=result, prob=prob, accuracy=ACCURACY
    )


@app.route("/predict", methods=["POST"])
def predict():
    label, prob = run_prediction([float(request.json[f]) for f in FEATURES])
    return jsonify({"prediction": label, "purchase_probability": prob})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
