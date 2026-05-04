"""FraudShield — Complete Fraud Detection Command Center"""
import streamlit as st
import streamlit_shadcn_ui as sui
from streamlit_echarts import st_echarts
import pandas as pd, numpy as np, json, os, sys, joblib
sys.path.insert(0, os.path.dirname(__file__))

st.set_page_config(page_title="Fraud Detection Pipeline", page_icon=":material/shield:", layout="wide", initial_sidebar_state="collapsed")

# ── Minimal CSS ──
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
:root{--bg:#09090B;--card:#18181B;--border:#27272A;--text:#FAFAFA;--muted:#71717A;--cyan:#22D3EE;--emerald:#34D399;--rose:#FB7185;--amber:#FBBF24;--blue:#60A5FA;}
html,body,.stApp{background:var(--bg)!important;color:var(--text);font-family:'DM Sans',sans-serif!important;}
#MainMenu,footer,header{visibility:hidden;}.stDeployButton{display:none;}
.top-bar{display:flex;align-items:center;justify-content:space-between;padding:0.5rem 0 1rem;border-bottom:1px solid var(--border);margin-bottom:1.2rem;}
.brand{font-family:'JetBrains Mono',monospace;font-size:1rem;font-weight:700;color:var(--cyan);letter-spacing:0.12em;}
.glass{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:1.1rem 1.3rem;}
.glass:hover{border-color:#3F3F46;}
.lbl{font-size:0.65rem;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.2rem;}
.val{font-family:'JetBrains Mono',monospace;font-size:1.4rem;font-weight:700;line-height:1.1;}
.sec{font-size:0.7rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:0.1em;margin:1.5rem 0 0.7rem;padding-bottom:0.5rem;border-bottom:1px solid var(--border);}
.verdict{border-radius:14px;padding:1.6rem;text-align:center;border:1px solid;}
.v-fraud{background:linear-gradient(160deg,#1C0A0E,#2D0F16);border-color:#F43F5E;box-shadow:0 0 30px rgba(244,63,94,0.1);}
.v-safe{background:linear-gradient(160deg,#051F14,#0A2E1C);border-color:#10B981;box-shadow:0 0 30px rgba(16,185,129,0.08);}
.pill{display:inline-block;padding:0.2rem 0.8rem;border-radius:100px;font-size:0.6rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;margin-top:0.4rem;}
.stTabs [data-baseweb="tab-list"]{background:var(--card);border-radius:10px;padding:4px;border:1px solid var(--border);gap:2px;}
.stTabs [data-baseweb="tab"]{font-family:'DM Sans',sans-serif!important;font-size:0.78rem!important;font-weight:600!important;color:var(--muted)!important;border-radius:8px!important;}
.stTabs [aria-selected="true"]{background:rgba(34,211,238,0.1)!important;color:var(--cyan)!important;}
div[data-testid="stForm"]{border:1px solid var(--border)!important;border-radius:12px!important;background:var(--card)!important;}
.app-footer{text-align:center;color:#3F3F46;font-size:0.6rem;font-family:'JetBrains Mono',monospace;padding:2rem 0 1rem;border-top:1px solid var(--border);margin-top:2rem;letter-spacing:0.04em;}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    d = os.path.join(os.path.dirname(__file__), 'models')
    try:
        model = joblib.load(os.path.join(d, 'best_model.joblib'))
        scaler = joblib.load(os.path.join(d, 'scaler.joblib'))
        le = joblib.load(os.path.join(d, 'label_encoders.joblib'))
        fn = joblib.load(os.path.join(d, 'feature_names.joblib'))
        with open(os.path.join(d, 'model_metadata.json')) as f: meta = json.load(f)
        with open(os.path.join(d, 'confusion_matrix.json')) as f: cm = json.load(f)
        return model, scaler, le, fn, meta, cm
    except: return None, None, None, None, None, None

def predict_single(model, scaler, le, fn, txn):
    from src.preprocess import NUMERIC_FEATURES
    features = {f: 0 for f in fn}
    for k, v in txn.items():
        if k in features: features[k] = v
    cats = ['ProductCD','card4','card6','P_emaildomain','R_emaildomain','DeviceType','DeviceInfo_cleaned']
    for c in cats:
        if c in features and c in le:
            val = str(features[c])
            features[c] = le[c].transform([val])[0] if val in le[c].classes_ else 0
    df = pd.DataFrame([features], columns=fn)
    if 'TransactionAmt_log' in df.columns: df['TransactionAmt_log'] = np.log1p(txn.get('TransactionAmt', 0))
    if 'Transaction_dayofweek' in df.columns: df['Transaction_dayofweek'] = 3
    nc = [c for c in NUMERIC_FEATURES + ['TransactionAmt_log','Transaction_hour','Transaction_dayofweek'] if c in df.columns]
    df[nc] = scaler.transform(df[nc])
    prob = model.predict_proba(df)[0][1]
    return df, prob

model, scaler, le, fn, meta, cm_data = load_artifacts()

# Header
st.markdown('<div class="top-bar"><div><span class="brand">FRAUD DETECTION PIPELINE</span><br><span style="font-size:0.68rem;color:#71717A;letter-spacing:0.06em;">Real-time transaction intelligence</span></div><div style="font-size:0.75rem;color:#34D399;">&#9679; Model active</div></div>', unsafe_allow_html=True)

if model is None:
    st.error("Model not loaded. Run the training pipeline first.", icon=":material/error:")
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs([":material/search: Analyze", ":material/science: Explainability", ":material/upload_file: Batch", ":material/analytics: Performance", ":material/database: Dataset"])

# ═══ TAB 1: ANALYZE ═══
with tab1:
    c1, c2 = st.columns([2, 3], gap="large")
    with c1:
        st.markdown('<div class="sec">Transaction input</div>', unsafe_allow_html=True)
        with st.form("txn_form"):
            amt = st.number_input("Amount (USD)", 0.0, 50000.0, 150.0, 10.0)
            r1, r2 = st.columns(2)
            product = r1.selectbox("Product", ["W","H","C","R","S"])
            email = r2.selectbox("Email", ["Gmail","Yahoo","Microsoft","Apple","AOL","Protonmail","Other","Unknown"])
            r3, r4 = st.columns(2)
            card_net = r3.selectbox("Card network", ["visa","mastercard","discover","american express"])
            card_type = r4.selectbox("Card type", ["debit","credit","charge"])
            r5, r6 = st.columns(2)
            dev_type = r5.selectbox("Device", ["desktop","mobile"])
            dev_os = r6.selectbox("OS", ["Windows","iOS","MacOS","Linux","Samsung","Android_Other","Other"])
            hour = st.slider("Hour", 0, 23, 14)
            submitted = st.form_submit_button("Run analysis", use_container_width=True, type="primary", icon=":material/play_arrow:")
    with c2:
        if submitted:
            txn = {'TransactionAmt':amt,'ProductCD':product,'card4':card_net,'card6':card_type,'P_emaildomain':email,'R_emaildomain':'Unknown','DeviceType':dev_type,'DeviceInfo_cleaned':dev_os,'Transaction_hour':hour,'Transaction_dayofweek':3}
            df_input, prob = predict_single(model, scaler, le, fn, txn)
            is_fraud = prob >= 0.5
            color = "#FB7185" if is_fraud else "#34D399"
            cls = "v-fraud" if is_fraud else "v-safe"
            conf = f"{prob*100:.1f}" if is_fraud else f"{(1-prob)*100:.1f}"
            risk = "CRITICAL" if prob>=0.85 else "HIGH" if prob>=0.6 else "MEDIUM" if prob>=0.3 else "LOW"
            st.markdown('<div class="sec">Verdict</div>', unsafe_allow_html=True)
            v1, v2 = st.columns(2)
            with v1:
                st.markdown(f'<div class="verdict {cls}"><div style="font-size:0.68rem;letter-spacing:0.15em;color:{color};font-family:JetBrains Mono,monospace;">{"BLOCKED" if is_fraud else "APPROVED"}</div><h2 style="color:{color};font-size:1.3rem;font-weight:800;margin:0.2rem 0;">{"Fraudulent" if is_fraud else "Legitimate"}</h2><div style="font-family:JetBrains Mono,monospace;font-size:2rem;font-weight:700;color:{color};">{conf}%</div><div class="pill" style="background:rgba({"244,63,94" if is_fraud else "16,185,129"},0.15);color:{color};border:1px solid rgba({"244,63,94" if is_fraud else "16,185,129"},0.25);">{risk} risk</div></div>', unsafe_allow_html=True)
            with v2:
                gauge = {"series":[{"type":"gauge","startAngle":220,"endAngle":-40,"min":0,"max":100,"pointer":{"length":"55%","width":4,"itemStyle":{"color":color}},"axisLine":{"lineStyle":{"width":10,"color":[[0.3,"#34D399"],[0.6,"#FBBF24"],[0.85,"#FB923C"],[1,"#FB7185"]]}},"axisTick":{"show":False},"splitLine":{"length":8,"lineStyle":{"color":"#3F3F46"}},"axisLabel":{"distance":16,"color":"#71717A","fontSize":9},"detail":{"valueAnimation":True,"formatter":"{value}%","color":color,"fontSize":20,"fontFamily":"JetBrains Mono","offsetCenter":[0,"70%"]},"data":[{"value":round(float(prob*100),1),"name":"Fraud score"}],"title":{"offsetCenter":[0,"90%"],"color":"#71717A","fontSize":10}}]}
                st_echarts(gauge, height="240px", theme="dark")
            st.markdown('<div class="sec">Summary</div>', unsafe_allow_html=True)
            tc = st.columns(4)
            for col,l,v,c in [(tc[0],"Amount",f"${amt:,.2f}","var(--cyan)"),(tc[1],"Product",product,"var(--blue)"),(tc[2],"Network",card_net.title(),"var(--amber)"),(tc[3],"Device",dev_os,"var(--emerald)")]:
                col.markdown(f'<div class="glass"><div class="lbl">{l}</div><div class="val" style="color:{c};">{v}</div></div>', unsafe_allow_html=True)
            st.session_state['last_input'] = df_input
            st.session_state['last_prob'] = prob
        else:
            st.markdown('<div style="text-align:center;padding:5rem 2rem;color:var(--muted);"><div style="font-size:2.5rem;opacity:0.3;margin-bottom:0.8rem;">&#9737;</div><h3 style="color:#A1A1AA;font-weight:600;font-size:1rem;">Awaiting transaction</h3><p style="font-size:0.8rem;max-width:340px;margin:0 auto;line-height:1.5;">Fill in details and press Run analysis.</p></div>', unsafe_allow_html=True)

# ═══ TAB 2: EXPLAINABILITY ═══
with tab2:
    st.markdown('<div class="sec">Why was this decision made?</div>', unsafe_allow_html=True)
    if 'last_input' in st.session_state:
        prob = st.session_state['last_prob']
        input_df = st.session_state['last_input']
        input_vals = input_df.values[0]

        # Use feature importance * feature value as contribution proxy (robust, no crash)
        importances = model.feature_importances_
        contributions = importances * np.abs(input_vals)
        # Sign: positive input values in important features push toward fraud
        signs = np.sign(input_vals) * np.sign(importances)
        signed_contrib = contributions * signs

        shap_df = pd.DataFrame({'feature': fn, 'contribution': signed_contrib})
        shap_df = shap_df.reindex(shap_df['contribution'].abs().sort_values(ascending=False).index).head(12)
        shap_df = shap_df.iloc[::-1]  # reverse for horizontal bar
        colors = ['#FB7185' if v > 0 else '#34D399' for v in shap_df['contribution']]

        opt = {
            "grid":{"left":"28%","right":"8%","top":"4%","bottom":"6%"},
            "xAxis":{"type":"value","axisLabel":{"color":"#71717A"},"splitLine":{"lineStyle":{"color":"#27272A"}},"name":"Feature contribution (importance x value)","nameTextStyle":{"color":"#71717A","fontSize":10}},
            "yAxis":{"type":"category","data":shap_df['feature'].tolist(),"axisLabel":{"color":"#A1A1AA","fontFamily":"JetBrains Mono","fontSize":10}},
            "series":[{"type":"bar","data":[{"value":round(float(v),4),"itemStyle":{"color":c}} for v,c in zip(shap_df['contribution'],colors)],"barWidth":"55%"}],
            "tooltip":{"trigger":"axis"}
        }
        st_echarts(opt, height="420px", theme="dark")
        st.caption("Red bars push toward FRAUD. Green bars push toward LEGITIMATE. The longer the bar, the stronger that feature influenced this specific prediction.")

        st.markdown(f'<div class="glass" style="margin-top:1rem;"><div class="lbl">Interpretation</div><p style="color:#A1A1AA;font-size:0.82rem;line-height:1.6;margin:0.3rem 0 0 0;">The model predicted a fraud probability of <strong style="color:{"var(--rose)" if prob>=0.5 else "var(--emerald)"};">{prob:.2%}</strong>. The chart above shows which features had the largest influence on this specific transaction. Features are ranked by <code>importance x scaled_value</code>.</p></div>', unsafe_allow_html=True)
    else:
        st.info("Run a prediction in the Analyze tab first to see explanations here.", icon=":material/info:")

# ═══ TAB 3: BATCH PREDICTION ═══
with tab3:
    st.markdown('<div class="sec">Upload transactions for batch scoring</div>', unsafe_allow_html=True)
    st.caption("Upload a CSV with columns: TransactionAmt, ProductCD, card4, card6, P_emaildomain, DeviceType, DeviceInfo_cleaned, Transaction_hour")
    uploaded = st.file_uploader("Choose CSV file", type="csv", label_visibility="collapsed")
    if uploaded:
        batch_df = pd.read_csv(uploaded)
        st.markdown(f'<div class="glass"><div class="lbl">Uploaded</div><div class="val" style="color:var(--cyan);">{len(batch_df):,} rows</div></div>', unsafe_allow_html=True)
        if st.button("Score all transactions", type="primary", icon=":material/play_arrow:", use_container_width=True):
            results = []
            prog = st.progress(0, text="Scoring...")
            for i, row in batch_df.iterrows():
                txn = row.to_dict()
                txn.setdefault('R_emaildomain', 'Unknown')
                txn.setdefault('Transaction_dayofweek', 3)
                try:
                    _, prob = predict_single(model, scaler, le, fn, txn)
                except: prob = 0.0
                results.append({'index': i, 'fraud_probability': round(prob, 4), 'prediction': 'FRAUD' if prob >= 0.5 else 'LEGITIMATE', 'risk': 'CRITICAL' if prob>=0.85 else 'HIGH' if prob>=0.6 else 'MEDIUM' if prob>=0.3 else 'LOW'})
                if i % max(1, len(batch_df)//20) == 0: prog.progress(min((i+1)/len(batch_df), 1.0), text=f"Scoring {i+1}/{len(batch_df)}...")
            prog.empty()
            res_df = pd.DataFrame(results)
            out = pd.concat([batch_df, res_df[['fraud_probability','prediction','risk']]], axis=1)
            fraud_count = (res_df['prediction'] == 'FRAUD').sum()
            rc = st.columns(3)
            rc[0].markdown(f'<div class="glass"><div class="lbl">Total</div><div class="val" style="color:var(--cyan);">{len(res_df):,}</div></div>', unsafe_allow_html=True)
            rc[1].markdown(f'<div class="glass"><div class="lbl">Flagged fraud</div><div class="val" style="color:var(--rose);">{fraud_count}</div></div>', unsafe_allow_html=True)
            rc[2].markdown(f'<div class="glass"><div class="lbl">Fraud rate</div><div class="val" style="color:var(--amber);">{fraud_count/len(res_df)*100:.1f}%</div></div>', unsafe_allow_html=True)
            st.dataframe(out, use_container_width=True, height=400)
            st.download_button("Download results CSV", out.to_csv(index=False), "fraud_predictions.csv", "text/csv", use_container_width=True, icon=":material/download:")

# ═══ TAB 4: PERFORMANCE ═══
with tab4:
    m = meta['metrics']
    st.markdown('<div class="sec">Key metrics</div>', unsafe_allow_html=True)
    mc = st.columns(4)
    for col,l,v,c in [(mc[0],"ROC-AUC",f"{m['roc_auc']:.4f}","var(--cyan)"),(mc[1],"F1 score",f"{m['f1_score']:.4f}","var(--blue)"),(mc[2],"Precision",f"{m['precision']:.4f}","var(--amber)"),(mc[3],"Recall",f"{m['recall']:.4f}","var(--emerald)")]:
        col.markdown(f'<div class="glass"><div class="lbl">{l}</div><div class="val" style="color:{c};">{v}</div></div>', unsafe_allow_html=True)
    p1, p2 = st.columns(2)
    with p1:
        st.markdown('<div class="sec">Confusion matrix</div>', unsafe_allow_html=True)
        cm_vals = [[0,0,cm_data['true_negatives']],[0,1,cm_data['false_positives']],[1,0,cm_data['false_negatives']],[1,1,cm_data['true_positives']]]
        cm_max = max(v[2] for v in cm_vals)
        st_echarts({"xAxis":{"type":"category","data":["Pred legit","Pred fraud"],"axisLabel":{"color":"#A1A1AA"}},"yAxis":{"type":"category","data":["Actual legit","Actual fraud"],"axisLabel":{"color":"#A1A1AA"}},"visualMap":{"min":0,"max":int(cm_max),"show":False,"inRange":{"color":["#18181B","#164E63","#22D3EE"]}},"series":[{"type":"heatmap","data":cm_vals,"label":{"show":True,"color":"#FAFAFA","fontSize":14,"fontFamily":"JetBrains Mono"}}]}, height="300px", theme="dark")
    with p2:
        st.markdown('<div class="sec">Model specification</div>', unsafe_allow_html=True)
        for k,v in {"Algorithm":meta['model_name'],"Train samples":f"{meta['training_samples']:,}","Test samples":f"{meta['test_samples']:,}","Features":str(meta['n_features']),"Train fraud":f"{meta['fraud_rate_train']:.2f}%","Test fraud":f"{meta['fraud_rate_test']:.2f}%"}.items():
            st.markdown(f'<div style="display:flex;justify-content:space-between;padding:0.5rem 0;border-bottom:1px solid var(--border);"><span style="color:var(--muted);font-size:0.8rem;">{k}</span><span style="color:var(--text);font-family:JetBrains Mono,monospace;font-size:0.8rem;">{v}</span></div>', unsafe_allow_html=True)
    # Feature importance
    st.markdown('<div class="sec">Feature importance — top 15</div>', unsafe_allow_html=True)
    if hasattr(model, 'feature_importances_'):
        imp = pd.DataFrame({'feature':fn,'importance':model.feature_importances_}).sort_values('importance',ascending=True).tail(15)
        st_echarts({"grid":{"left":"22%","right":"6%","top":"3%","bottom":"5%"},"xAxis":{"type":"value","axisLabel":{"color":"#71717A"},"splitLine":{"lineStyle":{"color":"#27272A"}}},"yAxis":{"type":"category","data":imp['feature'].tolist(),"axisLabel":{"color":"#A1A1AA","fontFamily":"JetBrains Mono","fontSize":10}},"series":[{"type":"bar","data":imp['importance'].round(4).tolist(),"itemStyle":{"color":{"type":"linear","x":0,"y":0,"x2":1,"y2":0,"colorStops":[{"offset":0,"color":"#164E63"},{"offset":1,"color":"#22D3EE"}]}},"barWidth":"55%"}],"tooltip":{"trigger":"axis"}}, height="400px", theme="dark")
    # ROC curve
    st.markdown('<div class="sec">ROC curve</div>', unsafe_allow_html=True)
    proc = os.path.join(os.path.dirname(__file__), 'data', 'processed')
    try:
        preds = pd.read_csv(os.path.join(proc, 'test_predictions.csv'))
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(preds['y_true'], preds['y_prob'])
        roc_val = auc(fpr, tpr)
        step = max(1, len(fpr)//200)
        st_echarts({"grid":{"left":"8%","right":"4%","bottom":"10%"},"xAxis":{"type":"value","name":"FPR","nameTextStyle":{"color":"#71717A"},"axisLabel":{"color":"#71717A"},"splitLine":{"lineStyle":{"color":"#27272A"}}},"yAxis":{"type":"value","name":"TPR","nameTextStyle":{"color":"#71717A"},"axisLabel":{"color":"#71717A"},"splitLine":{"lineStyle":{"color":"#27272A"}}},"series":[{"type":"line","data":list(zip(fpr[::step].tolist(),tpr[::step].tolist())),"smooth":True,"symbol":"none","lineStyle":{"color":"#22D3EE","width":2.5},"areaStyle":{"color":{"type":"linear","x":0,"y":0,"x2":0,"y2":1,"colorStops":[{"offset":0,"color":"rgba(34,211,238,0.12)"},{"offset":1,"color":"rgba(34,211,238,0)"}]}},"name":f"AUC={roc_val:.4f}"},{"type":"line","data":[[0,0],[1,1]],"lineStyle":{"color":"#3F3F46","type":"dashed"},"symbol":"none","name":"Baseline"}],"tooltip":{"trigger":"axis"},"legend":{"bottom":"0%","textStyle":{"color":"#A1A1AA"}}}, height="360px", theme="dark")
    except: pass

# ═══ TAB 5: DATASET OVERVIEW ═══
with tab5:
    proc = os.path.join(os.path.dirname(__file__), 'data', 'processed')
    try:
        preds = pd.read_csv(os.path.join(proc, 'test_predictions.csv'))
        st.markdown('<div class="sec">Class distribution</div>', unsafe_allow_html=True)
        d1, d2 = st.columns(2)
        with d1:
            fraud_n = int((preds['y_true']==1).sum()); legit_n = int((preds['y_true']==0).sum())
            st_echarts({"tooltip":{"trigger":"item"},"series":[{"type":"pie","radius":["40%","68%"],"itemStyle":{"borderRadius":6,"borderColor":"#09090B","borderWidth":2},"label":{"color":"#FAFAFA","fontSize":12},"data":[{"value":legit_n,"name":"Legitimate","itemStyle":{"color":"#34D399"}},{"value":fraud_n,"name":"Fraud","itemStyle":{"color":"#FB7185"}}]}]}, height="300px", theme="dark")
        with d2:
            hist_data = np.histogram(preds['y_prob'], bins=50)
            st_echarts({"xAxis":{"type":"category","data":[f"{v:.2f}" for v in hist_data[1][:-1]],"axisLabel":{"color":"#71717A","interval":9}},"yAxis":{"type":"value","axisLabel":{"color":"#71717A"},"splitLine":{"lineStyle":{"color":"#27272A"}}},"series":[{"type":"bar","data":hist_data[0].tolist(),"itemStyle":{"color":"#0891B2"},"barWidth":"90%"}],"tooltip":{"trigger":"axis"},"grid":{"left":"10%","right":"4%","bottom":"12%"}}, height="300px", theme="dark")
        st.markdown('<div class="sec">Dataset statistics</div>', unsafe_allow_html=True)
        ds = st.columns(4)
        ds[0].markdown(f'<div class="glass"><div class="lbl">Train samples</div><div class="val" style="color:var(--cyan);">{meta["training_samples"]:,}</div></div>', unsafe_allow_html=True)
        ds[1].markdown(f'<div class="glass"><div class="lbl">Test samples</div><div class="val" style="color:var(--blue);">{meta["test_samples"]:,}</div></div>', unsafe_allow_html=True)
        ds[2].markdown(f'<div class="glass"><div class="lbl">Features</div><div class="val" style="color:var(--amber);">{meta["n_features"]}</div></div>', unsafe_allow_html=True)
        ds[3].markdown(f'<div class="glass"><div class="lbl">Fraud rate</div><div class="val" style="color:var(--rose);">{meta["fraud_rate_test"]:.2f}%</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec">Sample predictions</div>', unsafe_allow_html=True)
        st.dataframe(preds.head(100), use_container_width=True, height=300)
    except: st.info("Run training pipeline to see dataset insights.", icon=":material/info:")

st.markdown('<div class="app-footer">FRAUD DETECTION PIPELINE v1.0 · IEEE-CIS Dataset · XGBoost + MLflow + FastAPI + Streamlit · SHAP Explainability</div>', unsafe_allow_html=True)
