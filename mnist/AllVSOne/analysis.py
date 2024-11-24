import streamlit as st
from PIL import Image
import json
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")

st.title('Deep Models for Anomaly detections on MNIST : ')
st.subheader('One class of digit is considered as the anomaly class')

st.sidebar.title('Parameters: ')
anormal = st.sidebar.slider("Anormal digit: ", min_value=0, max_value=9, value=0, step=1)
threshold = st.sidebar.slider("Threshold: ", min_value=0., max_value=0.2, step=0.01, value=0.05)

deepsvd_err = Image.open(f"results/figures/deepsvdd/mean_scores_{anormal}.jpg")
vae_err = Image.open(f"results/figures/vae/mean_scores_{anormal}.jpg")
cvae_err = Image.open(f"results/figures/cvae/mean_scores_{anormal}.jpg")
fanogan_err = Image.open(f"results/figures/fanogan/mean_scores_{anormal}.jpg")

col1, col2 = st.columns(2)
with col1:
    st.header('Deep One Class')
    st.image(deepsvd_err)

with col2:
    st.header('Linear VAE')
    st.image(vae_err)

st.divider()

col1, col2 = st.columns(2)
with col1:
    st.header('Conv VAE')
    st.image(cvae_err)

with col2:
    st.header('f-ANOGAN')
    st.image(fanogan_err)

st.header('Statistical tests')

with open(f"results/p_values/deepsvdd/pval_{anormal}.json", "r") as file:
    p_values_deepsvdd = json.load(file)

with open(f"results/p_values/vae/pval_{anormal}.json", "r") as file:
    p_values_vae = json.load(file)

with open(f"results/p_values/cvae/pval_{anormal}.json", "r") as file:
    p_values_cvae = json.load(file)

with open(f"results/p_values/fanogan/pval_{anormal}.json", "r") as file:
    p_values_fanogan = json.load(file)

results = []
for digit in range(10):
    
    p_values_deepsvdd_val, len_test_deepsvdd = p_values_deepsvdd[str(digit)]
    p_values_deepsvdd_val = np.asarray(p_values_deepsvdd_val)

    n_rejets_deepsvdd = (p_values_deepsvdd_val < threshold).sum().item()
    percentage_rejected_deepsvdd = n_rejets_deepsvdd / len_test_deepsvdd


    p_values_vae_val, len_test_vae = p_values_vae[str(digit)]
    p_values_vae_val = np.asarray(p_values_vae_val)

    n_rejets_vae = (p_values_vae_val < threshold).sum().item()
    percentage_rejected_vae = n_rejets_vae / len_test_vae


    p_values_cvae_val, len_test_cvae = p_values_cvae[str(digit)]
    p_values_cvae_val = np.asarray(p_values_cvae_val)

    n_rejets_cvae = (p_values_cvae_val < threshold).sum().item()
    percentage_rejected_cvae = n_rejets_cvae / len_test_cvae


    p_values_fanogan_val, len_test_fanogan = p_values_fanogan[str(digit)]
    p_values_fanogan_val = np.asarray(p_values_fanogan_val)

    n_rejets_fanogan = (p_values_fanogan_val < threshold).sum().item()
    percentage_rejected_fanogan = n_rejets_fanogan / len_test_fanogan

    results.append({
        "Digit": digit,
        "Anormal": "Yes" if digit == anormal else "No",
        "Threshold": threshold,

        "Rejections Deep SVDD": f"{n_rejets_deepsvdd}/{len_test_deepsvdd}",
        "Rejections VAE": f"{n_rejets_vae}/{len_test_vae}",
        "Rejections CVAE": f"{n_rejets_cvae}/{len_test_cvae}",
        "Rejections f-anogan": f"{n_rejets_fanogan}/{len_test_fanogan}",

        "Rejection Rate Deep SVDD": f"{percentage_rejected_deepsvdd:.3%}",
        "Rejection Rate VAE": f"{percentage_rejected_vae:.3%}",
        "Rejection Rate CVAE": f"{percentage_rejected_cvae:.3%}",
        "Rejection Rate f-anogan": f"{percentage_rejected_fanogan:.3%}"
    })

df_results = pd.DataFrame(results)

st.table(df_results)