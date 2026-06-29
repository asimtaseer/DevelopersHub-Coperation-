import os
import gradio as gr
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download


def load_model():
    """Load the trained Scikit-Learn pipeline from Hugging Face Hub."""
    try:
        model_path = hf_hub_download(
            repo_id="asimtaseer/customer-churn-pipeline",
            filename="customer_churn_pipeline.pkl",
        )
        return joblib.load(model_path)
    except Exception as exc:
        print(f"[ERROR] Could not load model from Hub: {exc}")
        return None


model = load_model()


CUSTOM_CSS = """
/* ── Google Fonts ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root Tokens ──────────────────────────────────────────────────────────── */
:root {
    --primary:        #1a73e8;
    --primary-hover:  #1557b0;
    --primary-light:  #e8f0fe;
    --success:        #1e8e3e;
    --success-light:  #e6f4ea;
    --danger:         #d93025;
    --danger-light:   #fce8e6;
    --neutral-50:     #f8f9fa;
    --neutral-100:    #f1f3f4;
    --neutral-200:    #e8eaed;
    --neutral-500:    #9aa0a6;
    --neutral-700:    #5f6368;
    --neutral-900:    #202124;
    --radius-card:    16px;
    --radius-input:   10px;
    --shadow-card:    0 1px 3px rgba(0,0,0,.08), 0 4px 16px rgba(0,0,0,.06);
    --shadow-lg:      0 4px 20px rgba(26,115,232,.18);
    --transition:     .2s ease;
}

/* ── Base Reset ───────────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    font-family: 'Inter', sans-serif !important;
    background: #f0f4f9 !important;
    color: var(--neutral-900) !important;
    margin: 0 !important;
    padding: 0 !important;
}

.gradio-container { max-width: 960px !important; margin: 0 auto !important; padding: 0 16px 48px !important; }

/* ── Hero Banner ──────────────────────────────────────────────────────────── */
#hero-banner {
    background: linear-gradient(135deg, #1a73e8 0%, #0d47a1 100%);
    border-radius: var(--radius-card);
    padding: 40px 48px;
    margin: 28px 0 24px;
    color: #fff;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-lg);
}
#hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    border-radius: 50%;
    background: rgba(255,255,255,.07);
}
#hero-banner::after {
    content: '';
    position: absolute;
    bottom: -80px; left: -40px;
    width: 220px; height: 220px;
    border-radius: 50%;
    background: rgba(255,255,255,.05);
}
.hero-title {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -.5px;
    margin: 0 0 10px;
    line-height: 1.2;
    position: relative; z-index: 1;
}
.hero-subtitle {
    font-size: 1rem;
    font-weight: 400;
    opacity: .88;
    margin: 0;
    position: relative; z-index: 1;
    max-width: 540px;
    line-height: 1.55;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,255,255,.15);
    border: 1px solid rgba(255,255,255,.25);
    border-radius: 20px;
    padding: 5px 14px;
    font-size: .78rem;
    font-weight: 600;
    letter-spacing: .3px;
    margin-bottom: 16px;
    position: relative; z-index: 1;
    backdrop-filter: blur(4px);
}

/* ── Section Headers ──────────────────────────────────────────────────────── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 18px;
}
.section-icon {
    width: 36px; height: 36px;
    border-radius: 10px;
    background: var(--primary-light);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
}
.section-title {
    font-size: 1.08rem;
    font-weight: 700;
    color: var(--neutral-900);
    margin: 0;
}
.section-desc {
    font-size: .82rem;
    color: var(--neutral-500);
    margin: 0;
}

/* ── Cards ────────────────────────────────────────────────────────────────── */
.card {
    background: #fff;
    border-radius: var(--radius-card);
    box-shadow: var(--shadow-card);
    padding: 28px 32px;
    margin-bottom: 20px;
    border: 1px solid var(--neutral-200);
    transition: box-shadow var(--transition);
}
.card:hover { box-shadow: 0 2px 8px rgba(0,0,0,.10), 0 8px 24px rgba(0,0,0,.08); }

/* ── Gradio Component Overrides ───────────────────────────────────────────── */
.gradio-container label span,
.gradio-container .label-wrap span {
    font-size: .84rem !important;
    font-weight: 600 !important;
    color: var(--neutral-700) !important;
    letter-spacing: .15px !important;
    margin-bottom: 4px !important;
}

/* Dropdowns & Number inputs */
.gradio-container select,
.gradio-container input[type="number"] {
    border: 1.5px solid var(--neutral-200) !important;
    border-radius: var(--radius-input) !important;
    padding: 10px 14px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: .9rem !important;
    color: var(--neutral-900) !important;
    background: #fff !important;
    transition: border-color var(--transition), box-shadow var(--transition) !important;
    outline: none !important;
}
.gradio-container select:focus,
.gradio-container input[type="number"]:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(26,115,232,.12) !important;
}

/* Radio buttons */
.gradio-container .wrap label {
    padding: 8px 14px !important;
    border-radius: 8px !important;
    border: 1.5px solid var(--neutral-200) !important;
    margin: 0 4px 0 0 !important;
    cursor: pointer !important;
    font-size: .87rem !important;
    font-weight: 500 !important;
    transition: all var(--transition) !important;
}
.gradio-container .wrap label:hover { border-color: var(--primary) !important; background: var(--primary-light) !important; }
.gradio-container .wrap input[type="radio"]:checked + span { color: var(--primary) !important; }

/* Slider */
.gradio-container input[type="range"] {
    accent-color: var(--primary) !important;
    height: 6px !important;
}

/* ── Predict Button ───────────────────────────────────────────────────────── */
#predict-btn {
    background: linear-gradient(135deg, #1a73e8, #0d47a1) !important;
    color: #fff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1.02rem !important;
    font-weight: 700 !important;
    letter-spacing: .4px !important;
    padding: 16px 40px !important;
    border-radius: 12px !important;
    border: none !important;
    cursor: pointer !important;
    width: 100% !important;
    transition: all var(--transition) !important;
    box-shadow: 0 4px 14px rgba(26,115,232,.35) !important;
    margin-top: 4px !important;
}
#predict-btn:hover {
    background: linear-gradient(135deg, #1557b0, #0a3880) !important;
    box-shadow: 0 6px 20px rgba(26,115,232,.45) !important;
    transform: translateY(-1px) !important;
}
#predict-btn:active { transform: translateY(0) !important; }

/* ── Result Cards ─────────────────────────────────────────────────────────── */
#result-output .prose, #result-output { background: transparent !important; }
.result-churn {
    background: #fff;
    border: 2px solid #f5c6c4;
    border-left: 6px solid var(--danger);
    border-radius: var(--radius-card);
    padding: 28px 32px;
    margin-bottom: 16px;
    box-shadow: 0 2px 12px rgba(217,48,37,.10);
}
.result-safe {
    background: #fff;
    border: 2px solid #a8d5b5;
    border-left: 6px solid var(--success);
    border-radius: var(--radius-card);
    padding: 28px 32px;
    margin-bottom: 16px;
    box-shadow: 0 2px 12px rgba(30,142,62,.10);
}
.result-title { font-size: 1.45rem; font-weight: 800; margin: 0 0 6px; }
.result-sub   { font-size: .9rem; color: var(--neutral-700); margin: 0 0 20px; }

/* Progress bar */
.progress-wrap { margin-top: 12px; }
.progress-label {
    display: flex; justify-content: space-between;
    font-size: .82rem; font-weight: 600; margin-bottom: 6px;
    color: var(--neutral-700);
}
.progress-bar-bg {
    background: var(--neutral-100);
    border-radius: 100px;
    height: 10px;
    overflow: hidden;
}
.progress-bar-fill {
    height: 100%;
    border-radius: 100px;
    transition: width .6s cubic-bezier(.4,0,.2,1);
}

/* Model info chips */
.model-card {
    background: #fff;
    border-radius: var(--radius-card);
    box-shadow: var(--shadow-card);
    padding: 24px 32px;
    border: 1px solid var(--neutral-200);
    margin-bottom: 20px;
}
.model-grid { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 12px; }
.model-chip {
    background: var(--neutral-50);
    border: 1px solid var(--neutral-200);
    border-radius: 10px;
    padding: 10px 18px;
    flex: 1 1 140px;
    min-width: 130px;
}
.chip-label { font-size: .73rem; font-weight: 700; color: var(--neutral-500); text-transform: uppercase; letter-spacing: .6px; margin-bottom: 4px; }
.chip-value { font-size: 1.05rem; font-weight: 700; color: var(--neutral-900); }
.chip-value.primary { color: var(--primary); }

/* ── Footer ───────────────────────────────────────────────────────────────── */
#footer {
    text-align: center;
    padding: 24px 0 8px;
    color: var(--neutral-500);
    font-size: .83rem;
}
#footer strong { color: var(--neutral-700); }
.social-links { display: flex; justify-content: center; gap: 14px; margin-top: 10px; }
.social-link {
    display: inline-flex; align-items: center; justify-content: center;
    width: 34px; height: 34px; border-radius: 8px;
    background: var(--neutral-100); color: var(--neutral-700);
    text-decoration: none; font-size: 1rem;
    transition: all var(--transition);
    border: 1px solid var(--neutral-200);
}
.social-link:hover { background: var(--primary); color: #fff; border-color: var(--primary); transform: translateY(-2px); }

/* ── Responsive ───────────────────────────────────────────────────────────── */
@media (max-width: 640px) {
    #hero-banner { padding: 28px 22px; }
    .hero-title  { font-size: 1.45rem; }
    .card        { padding: 20px 18px; }
    .model-grid  { flex-direction: column; }
}
"""

# HTML HELPERS


HERO_HTML = """
<div id="hero-banner">
  <div class="hero-badge">🤖 &nbsp;Powered by Scikit-Learn</div>
  <h1 class="hero-title">🔮 Customer Churn Prediction System</h1>
  <p class="hero-subtitle">
    Predict whether a telecom customer is likely to churn using a
    trained Machine Learning Pipeline — instantly and accurately.
  </p>
</div>
"""

SECTION_CUSTOMER = """
<div class="section-header">
  <div class="section-icon">👤</div>
  <div>
    <p class="section-title">Customer Information</p>
    <p class="section-desc">Basic demographic details about the customer</p>
  </div>
</div>
"""

SECTION_SERVICES = """
<div class="section-header">
  <div class="section-icon">📡</div>
  <div>
    <p class="section-title">Telecom Services</p>
    <p class="section-desc">Current subscribed services and add-ons</p>
  </div>
</div>
"""

SECTION_BILLING = """
<div class="section-header">
  <div class="section-icon">💳</div>
  <div>
    <p class="section-title">Billing Details</p>
    <p class="section-desc">Contract, payment method &amp; financial info</p>
  </div>
</div>
"""

FOOTER_HTML = """
<div id="footer">
  <strong>Developed by Asim Qurashi</strong> &nbsp;·&nbsp; AI Engineer<br/>
  <div class="social-links">
    <a class="social-link" href="https://github.com/asimtaseer" target="_blank" title="GitHub">
      <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
        <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483
        0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466
        -.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832
        .092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688
        -.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115
        2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595
        1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012
        2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"/>
      </svg>
    </a>
    <a class="social-link" href="https://linkedin.com/in/asimtaseer" target="_blank" title="LinkedIn">
      <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
        <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136
        2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267
        5.455v6.286zM5.337 7.433a2.062 2.062 0 01-2.063-2.065 2.064 2.064 0 112.063 2.065zm1.782
        13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24
        1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
      </svg>
    </a>
    <a class="social-link" href="https://huggingface.co/asimtaseer" target="_blank" title="Hugging Face">
      🤗
    </a>
  </div>
</div>
"""

MODEL_INFO_HTML = """
<div class="model-card">
  <div class="section-header" style="margin-bottom:0">
    <div class="section-icon">📊</div>
    <div>
      <p class="section-title">Model Information</p>
      <p class="section-desc">Statistics for the deployed pipeline</p>
    </div>
  </div>
  <div class="model-grid">
    <div class="model-chip">
      <div class="chip-label">Algorithm</div>
      <div class="chip-value primary">Logistic Regression</div>
    </div>
    <div class="model-chip">
      <div class="chip-label">Accuracy</div>
      <div class="chip-value">80.55 %</div>
    </div>
    <div class="model-chip">
      <div class="chip-label">ROC-AUC</div>
      <div class="chip-value">84.18 %</div>
    </div>
    <div class="model-chip">
      <div class="chip-label">Pipeline</div>
      <div class="chip-value">Scikit-Learn</div>
    </div>
  </div>
</div>
"""


# PREDICTION LOGIC


def build_result_html(prediction: int, probability: float) -> str:

    pct = round(probability * 100, 2)
    bar_pct = round(pct)

    if prediction == 1:
        card_class   = "result-churn"
        emoji        = "🔴"
        verdict      = "Customer is Likely to Churn"
        note         = "This customer shows a high risk of leaving. Consider proactive retention strategies."
        bar_color    = "#d93025"
        confidence_label = "Churn Probability"
    else:
        card_class   = "result-safe"
        emoji        = "🟢"
        verdict      = "Customer is NOT Likely to Churn"
        note         = "This customer appears satisfied and loyal. Keep up the great service!"
        bar_color    = "#1e8e3e"
        confidence_label = "Retention Confidence"

    html = f"""
    <div class="{card_class}">
      <div class="result-title">{emoji} {verdict}</div>
      <div class="result-sub">{note}</div>
      <div class="progress-wrap">
        <div class="progress-label">
          <span>{confidence_label}</span>
          <span style="font-size:1.1rem;font-weight:800;color:{bar_color}">{pct}%</span>
        </div>
        <div class="progress-bar-bg">
          <div class="progress-bar-fill" style="width:{bar_pct}%;background:{bar_color};"></div>
        </div>
      </div>
    </div>
    {MODEL_INFO_HTML}
    """
    return html


def predict_churn(
    gender, senior_citizen, partner, dependents,
    phone_service, multiple_lines, internet_service,
    online_security, online_backup, device_protection,
    tech_support, streaming_tv, streaming_movies,
    contract, paperless_billing, payment_method,
    tenure, monthly_charges, total_charges,
):


    # ── Guard: model not loaded ──────────────────────────────────────────────
    if model is None:
        return """
        <div class="result-churn">
          <div class="result-title">⚠️ Model Unavailable</div>
          <div class="result-sub">
            Could not load the prediction model. Please check the Hugging Face
            Hub connection or try again in a moment.
          </div>
        </div>
        """

    # Map UI labels → model-expected values
    # SeniorCitizen is int in original dataset (0 / 1)
    senior_map = {"Yes": 1, "No": 0}

    input_df = pd.DataFrame([{
        "gender":          gender,
        "SeniorCitizen":   senior_map.get(senior_citizen, 0),
        "Partner":         partner,
        "Dependents":      dependents,
        "tenure":          int(tenure),
        "PhoneService":    phone_service,
        "MultipleLines":   multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity":  online_security,
        "OnlineBackup":    online_backup,
        "DeviceProtection": device_protection,
        "TechSupport":     tech_support,
        "StreamingTV":     streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract":        contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod":   payment_method,
        "MonthlyCharges":  float(monthly_charges),
        "TotalCharges":    float(total_charges),
    }])

    # Inference 
    try:
        prediction  = int(model.predict(input_df)[0])
        proba_array = model.predict_proba(input_df)[0]
        # Index 1 = probability of churn; index 0 = probability of no churn
        probability = float(proba_array[1]) if prediction == 1 else float(proba_array[0])
    except Exception as exc:
        return f"""
        <div class="result-churn">
          <div class="result-title">❌ Prediction Error</div>
          <div class="result-sub">{str(exc)}</div>
        </div>
        """

    return build_result_html(prediction, probability)



# GRADIO INTERFACE


with gr.Blocks(css=CUSTOM_CSS, title="Customer Churn Prediction System") as demo:

    # Hero 
    gr.HTML(HERO_HTML)

    # SECTION 1  Customer Information
    with gr.Group(elem_classes="card"):
        gr.HTML(SECTION_CUSTOMER)

        with gr.Row():
            gender = gr.Dropdown(
                label="Gender",
                choices=["Male", "Female"],
                value="Male",
                interactive=True,
            )
            senior_citizen = gr.Radio(
                label="Senior Citizen",
                choices=["Yes", "No"],
                value="No",
                interactive=True,
            )

        with gr.Row():
            partner = gr.Radio(
                label="Partner",
                choices=["Yes", "No"],
                value="No",
                interactive=True,
            )
            dependents = gr.Radio(
                label="Dependents",
                choices=["Yes", "No"],
                value="No",
                interactive=True,
            )


    # SECTION 2 Telecom Services
    with gr.Group(elem_classes="card"):
        gr.HTML(SECTION_SERVICES)

        with gr.Row():
            phone_service = gr.Dropdown(
                label="Phone Service",
                choices=["Yes", "No"],
                value="Yes",
                interactive=True,
            )
            multiple_lines = gr.Dropdown(
                label="Multiple Lines",
                choices=["Yes", "No", "No phone service"],
                value="No",
                interactive=True,
            )
            internet_service = gr.Dropdown(
                label="Internet Service",
                choices=["DSL", "Fiber optic", "No"],
                value="DSL",
                interactive=True,
            )

        with gr.Row():
            online_security = gr.Dropdown(
                label="Online Security",
                choices=["Yes", "No", "No internet service"],
                value="No",
                interactive=True,
            )
            online_backup = gr.Dropdown(
                label="Online Backup",
                choices=["Yes", "No", "No internet service"],
                value="No",
                interactive=True,
            )
            device_protection = gr.Dropdown(
                label="Device Protection",
                choices=["Yes", "No", "No internet service"],
                value="No",
                interactive=True,
            )

        with gr.Row():
            tech_support = gr.Dropdown(
                label="Tech Support",
                choices=["Yes", "No", "No internet service"],
                value="No",
                interactive=True,
            )
            streaming_tv = gr.Dropdown(
                label="Streaming TV",
                choices=["Yes", "No", "No internet service"],
                value="No",
                interactive=True,
            )
            streaming_movies = gr.Dropdown(
                label="Streaming Movies",
                choices=["Yes", "No", "No internet service"],
                value="No",
                interactive=True,
            )

    # SECTION 3 — Billing Details
    with gr.Group(elem_classes="card"):
        gr.HTML(SECTION_BILLING)

        with gr.Row():
            contract = gr.Dropdown(
                label="Contract",
                choices=["Month-to-month", "One year", "Two year"],
                value="Month-to-month",
                interactive=True,
            )
            paperless_billing = gr.Radio(
                label="Paperless Billing",
                choices=["Yes", "No"],
                value="Yes",
                interactive=True,
            )

        with gr.Row():
            payment_method = gr.Dropdown(
                label="Payment Method",
                choices=[
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                value="Electronic check",
                interactive=True,
            )

        with gr.Row():
            tenure = gr.Slider(
                label="Tenure (months)",
                minimum=0,
                maximum=72,
                step=1,
                value=12,
                interactive=True,
            )

        with gr.Row():
            monthly_charges = gr.Number(
                label="Monthly Charges ($)",
                value=65.0,
                minimum=0.0,
                interactive=True,
            )
            total_charges = gr.Number(
                label="Total Charges ($)",
                value=780.0,
                minimum=0.0,
                interactive=True,
            )

    # PREDICT BUTTON
    predict_btn = gr.Button(
        "🔍 Predict Churn",
        elem_id="predict-btn",
        variant="primary",
    )

    # RESULT OUTPUT
    result_output = gr.HTML(
        label="Prediction Result",
        elem_id="result-output",
    )

    # Wire up 
    predict_btn.click(
        fn=predict_churn,
        inputs=[
            gender, senior_citizen, partner, dependents,
            phone_service, multiple_lines, internet_service,
            online_security, online_backup, device_protection,
            tech_support, streaming_tv, streaming_movies,
            contract, paperless_billing, payment_method,
            tenure, monthly_charges, total_charges,
        ],
        outputs=result_output,
    )

    # Footer 
    gr.HTML(FOOTER_HTML)


if __name__ == "__main__":
    demo.launch()
