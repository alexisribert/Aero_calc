import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import fsolve

st.set_page_config(page_title="Appli Aéro Turbo-Boostée", layout="wide")

st.title("🚀 Appli Aéro Turbo-Boostée")
st.markdown("Cette application reproduit et améliore les calculs de l'onglet **Expérimental**.")

# --- Dictionnaires de conversion ---
unit_factors = {
    "Pression": {"Pa": 1.0, "bar": 1e5, "atm": 101325.0},
    "Vitesse": {"m/s": 1.0, "km/h": 1/3.6, "kt": 1/1.94384},
    "Distance": {"m": 1.0, "ft": 0.3048, "km": 1000.0},
    "Temperature": {"K": (1.0, 0.0), "°C": (1.0, 273.15)} # (multiplier, offset)
}

def convert_to_SI(val, unit, category):
    if pd.isna(val): return val
    if category == "Temperature":
        return val * unit_factors[category][unit][0] + unit_factors[category][unit][1]
    return val * unit_factors[category][unit]

def convert_from_SI(val, unit, category):
    if pd.isna(val): return val
    if category == "Temperature":
        return (val - unit_factors[category][unit][1]) / unit_factors[category][unit][0]
    return val / unit_factors[category][unit]

# =========================================================================
st.header("1. Hydrostatique & Manomètres")

col1, col2 = st.columns(2)

with col1:
    with st.expander("ℹ️ Manomètre à Eau"):
        st.latex(r"\Delta P = \rho_{eau} \cdot g \cdot \Delta h")
        
    st.markdown("**Manomètre à Eau**")
    
    col_u1_p, col_u1_h = st.columns(2)
    with col_u1_p: u_p1 = st.selectbox("Unité Pression", ["Pa", "bar"], key="u_p1")
    with col_u1_h: u_h1 = st.selectbox("Unité Hauteur", ["m", "cm", "mm"], key="u_h1")
    factor_h1 = {"m":1.0, "cm":0.01, "mm":0.001}[u_h1]
    
    if 'eau_data' not in st.session_state:
        st.session_state.eau_data = pd.DataFrame({"Rho_eau (kg/m3)": [997.0, 997.0], f"Delta P ({u_p1})": [9780.0, np.nan], f"Delta h ({u_h1})": [np.nan, 1.5]})
        
    # On force la mise à jour des entêtes au cas où l'utilisateur change les selectbox
    st.session_state.eau_data.columns = ["Rho_eau (kg/m3)", f"Delta P ({u_p1})", f"Delta h ({u_h1})"]

    with st.form("form_eau"):
        df_eau = st.data_editor(st.session_state.eau_data, num_rows="dynamic", use_container_width=True)
        if st.form_submit_button("Calculer"):
            for i, row in df_eau.iterrows():
                rho = row["Rho_eau (kg/m3)"]
                p_val = convert_to_SI(row[f"Delta P ({u_p1})"], u_p1, "Pression")
                h_val = row[f"Delta h ({u_h1})"] * factor_h1
                
                if pd.isna(p_val) and not pd.isna(h_val) and not pd.isna(rho):
                    p_val = rho * 9.81 * h_val
                    df_eau.at[i, f"Delta P ({u_p1})"] = convert_from_SI(p_val, u_p1, "Pression")
                elif pd.isna(h_val) and not pd.isna(p_val) and not pd.isna(rho):
                    h_val = p_val / (rho * 9.81)
                    df_eau.at[i, f"Delta h ({u_h1})"] = h_val / factor_h1
            st.session_state.eau_data = df_eau
            st.rerun()

with col2:
    with st.expander("ℹ️ Manomètre à Mercure"):
        st.latex(r"\Delta P = \rho_{Hg} \cdot g \cdot \Delta h")
        
    st.markdown("**Manomètre à Mercure**")
    
    col_u2_p, col_u2_h = st.columns(2)
    with col_u2_p: u_p2 = st.selectbox("Unité Pression", ["Pa", "bar"], key="u_p2")
    with col_u2_h: u_h2 = st.selectbox("Unité Hauteur", ["m", "cm", "mm"], key="u_h2")
    factor_h2 = {"m":1.0, "cm":0.01, "mm":0.001}[u_h2]
    
    if 'hg_data' not in st.session_state:
        st.session_state.hg_data = pd.DataFrame({"Rho_Hg (kg/m3)": [13600.0, 13600.0], f"Delta P ({u_p2})": [101325.0, np.nan], f"Delta h ({u_h2})": [np.nan, 0.76]})
        
    st.session_state.hg_data.columns = ["Rho_Hg (kg/m3)", f"Delta P ({u_p2})", f"Delta h ({u_h2})"]

    with st.form("form_hg"):
        df_hg = st.data_editor(st.session_state.hg_data, num_rows="dynamic", use_container_width=True)
        if st.form_submit_button("Calculer"):
            for i, row in df_hg.iterrows():
                rho = row["Rho_Hg (kg/m3)"]
                p_val = convert_to_SI(row[f"Delta P ({u_p2})"], u_p2, "Pression")
                h_val = row[f"Delta h ({u_h2})"] * factor_h2
                
                if pd.isna(p_val) and not pd.isna(h_val) and not pd.isna(rho):
                    p_val = rho * 9.81 * h_val
                    df_hg.at[i, f"Delta P ({u_p2})"] = convert_from_SI(p_val, u_p2, "Pression")
                elif pd.isna(h_val) and not pd.isna(p_val) and not pd.isna(rho):
                    h_val = p_val / (rho * 9.81)
                    df_hg.at[i, f"Delta h ({u_h2})"] = h_val / factor_h2
            st.session_state.hg_data = df_hg
            st.rerun()

# =========================================================================
st.divider()
st.header("2. Dynamique des Fluides")

col3, col4 = st.columns(2)

with col3:
    with st.expander("ℹ️ Loi de Torricelli"):
        st.write("**Cas d'usage:** Vitesse d'écoulement d'un fluide sous l'effet de la gravité.")
        st.latex(r"U = \sqrt{2 \cdot g \cdot h}")
        
    st.markdown("**Loi de Torricelli**")
    
    col_u3_v, col_u3_h = st.columns(2)
    with col_u3_v: u_v3 = st.selectbox("Unité Vitesse", ["m/s", "km/h"], key="u_v3")
    with col_u3_h: u_h3 = st.selectbox("Unité Hauteur", ["m", "ft"], key="u_h3")
    
    if 'torri_data' not in st.session_state:
        st.session_state.torri_data = pd.DataFrame({f"Vitesse U ({u_v3})": [np.nan, 10.0], f"Hauteur h ({u_h3})": [5.0, np.nan]})
        
    st.session_state.torri_data.columns = [f"Vitesse U ({u_v3})", f"Hauteur h ({u_h3})"]

    with st.form("form_torri"):
        df_torri = st.data_editor(st.session_state.torri_data, num_rows="dynamic", use_container_width=True)
        if st.form_submit_button("Calculer"):
            for i, row in df_torri.iterrows():
                u_val = convert_to_SI(row[f"Vitesse U ({u_v3})"], u_v3, "Vitesse")
                h_val = convert_to_SI(row[f"Hauteur h ({u_h3})"], u_h3, "Distance")
                
                if pd.isna(u_val) and not pd.isna(h_val) and h_val >= 0:
                    u_val = np.sqrt(2 * 9.81 * h_val)
                    df_torri.at[i, f"Vitesse U ({u_v3})"] = convert_from_SI(u_val, u_v3, "Vitesse")
                elif pd.isna(h_val) and not pd.isna(u_val):
                    h_val = (u_val**2) / (2 * 9.81)
                    df_torri.at[i, f"Hauteur h ({u_h3})"] = convert_from_SI(h_val, u_h3, "Distance")
            st.session_state.torri_data = df_torri
            st.rerun()

with col4:
    with st.expander("ℹ️ Sonde Pitot (Incompressible)"):
        st.write("**Cas d'usage:** Mesure de la vitesse via la pression dynamique (Mach < 0.3).")
        st.latex(r"U = \sqrt{\frac{2 \cdot \Delta P}{\rho}}")
        
    st.markdown("**Sonde Pitot**")
    
    col_u4_v, col_u4_p = st.columns(2)
    with col_u4_v: u_v4 = st.selectbox("Unité Vitesse", ["m/s", "km/h", "kt"], key="u_v4")
    with col_u4_p: u_p4 = st.selectbox("Unité Pression", ["Pa", "bar"], key="u_p4")
    
    if 'pitot_data' not in st.session_state:
        st.session_state.pitot_data = pd.DataFrame({f"Vitesse U ({u_v4})": [np.nan, 250.0], f"Delta P ({u_p4})": [44500.0, np.nan], "Rho (kg/m3)": [1.225, 1.225]})
        
    st.session_state.pitot_data.columns = [f"Vitesse U ({u_v4})", f"Delta P ({u_p4})", "Rho (kg/m3)"]

    with st.form("form_pitot"):
        df_pitot = st.data_editor(st.session_state.pitot_data, num_rows="dynamic", use_container_width=True)
        if st.form_submit_button("Calculer"):
            for i, row in df_pitot.iterrows():
                u_val = convert_to_SI(row[f"Vitesse U ({u_v4})"], u_v4, "Vitesse")
                p_val = convert_to_SI(row[f"Delta P ({u_p4})"], u_p4, "Pression")
                rho = row["Rho (kg/m3)"]
                
                if pd.isna(u_val) and not pd.isna(p_val) and not pd.isna(rho) and p_val >= 0 and rho > 0:
                    u_val = np.sqrt(2 * p_val / rho)
                    df_pitot.at[i, f"Vitesse U ({u_v4})"] = convert_from_SI(u_val, u_v4, "Vitesse")
                elif pd.isna(p_val) and not pd.isna(u_val) and not pd.isna(rho):
                    p_val = 0.5 * rho * (u_val**2)
                    df_pitot.at[i, f"Delta P ({u_p4})"] = convert_from_SI(p_val, u_p4, "Pression")
            st.session_state.pitot_data = df_pitot
            st.rerun()

# =========================================================================
st.divider()
st.header("3. Thermodynamique & Atmosphère")

with st.expander("ℹ️ Loi des Gaz Parfaits"):
    st.latex(r"P = \rho \cdot r \cdot T")

st.markdown("**Loi des Gaz Parfaits**")
col_u5_p, col_u5_t = st.columns(2)
with col_u5_p: u_p5 = st.selectbox("Unité Pression", ["Pa", "bar", "atm"], key="u_p5")
with col_u5_t: u_t5 = st.selectbox("Unité Température", ["K", "°C"], key="u_t5")

if 'gaz_data' not in st.session_state:
    st.session_state.gaz_data = pd.DataFrame({
        f"Pression P ({u_p5})": [101325.0, np.nan, 75600.0], 
        "Densité Rho (kg/m3)": [1.225, 1.225, np.nan], 
        f"Température T ({u_t5})": [np.nan, 288.15, 293.15],
        "Constante r (J/kg.K)": [287.1, 287.1, 287.1]
    })
    
st.session_state.gaz_data.columns = [f"Pression P ({u_p5})", "Densité Rho (kg/m3)", f"Température T ({u_t5})", "Constante r (J/kg.K)"]

with st.form("form_gaz"):
    df_gaz = st.data_editor(st.session_state.gaz_data, num_rows="dynamic", use_container_width=True)
    if st.form_submit_button("Calculer"):
        for i, row in df_gaz.iterrows():
            p = convert_to_SI(row[f"Pression P ({u_p5})"], u_p5, "Pression")
            rho = row["Densité Rho (kg/m3)"]
            t = convert_to_SI(row[f"Température T ({u_t5})"], u_t5, "Temperature")
            r = row["Constante r (J/kg.K)"]
            
            if pd.isna(p) and not pd.isna(rho) and not pd.isna(t) and not pd.isna(r):
                p = rho * r * t
                df_gaz.at[i, f"Pression P ({u_p5})"] = convert_from_SI(p, u_p5, "Pression")
            elif pd.isna(rho) and not pd.isna(p) and not pd.isna(t) and not pd.isna(r) and t > 0:
                rho = p / (r * t)
                df_gaz.at[i, "Densité Rho (kg/m3)"] = rho
            elif pd.isna(t) and not pd.isna(p) and not pd.isna(rho) and not pd.isna(r) and rho > 0:
                t = p / (rho * r)
                df_gaz.at[i, f"Température T ({u_t5})"] = convert_from_SI(t, u_t5, "Temperature")
        st.session_state.gaz_data = df_gaz
        st.rerun()

col5, col6 = st.columns(2)

with col5:
    with st.expander("ℹ️ Loi de Sutherland"):
        st.write("**Cas d'usage:** Calcul de la viscosité dynamique de l'air selon la température.")
        st.latex(r"\mu \approx \frac{B \sqrt{T}}{1 + A/T}")
        
    st.markdown("**Loi de Sutherland (Air)**")
    col_u6_t = st.columns(1)[0]
    u_t6 = col_u6_t.selectbox("Unité Température", ["K", "°C"], key="u_t6")
    
    if 'suth_data' not in st.session_state:
        st.session_state.suth_data = pd.DataFrame({
            f"Température T ({u_t6})": [293.15, np.nan], 
            "Viscosité Mu (Pa.s)": [np.nan, 1.81e-5],
            "Constante A": [114.0, 114.0],
            "Constante B": [1.458e-6, 1.458e-6]
        })
        
    st.session_state.suth_data.columns = [f"Température T ({u_t6})", "Viscosité Mu (Pa.s)", "Constante A", "Constante B"]

    with st.form("form_suth"):
        df_suth = st.data_editor(st.session_state.suth_data, num_rows="dynamic", use_container_width=True)
        if st.form_submit_button("Calculer"):
            for i, row in df_suth.iterrows():
                t = convert_to_SI(row[f"Température T ({u_t6})"], u_t6, "Temperature")
                mu = row["Viscosité Mu (Pa.s)"]
                A, B = row["Constante A"], row["Constante B"]
                
                if pd.isna(mu) and not pd.isna(t) and t > 0:
                    mu = (B * np.sqrt(t)) / (1 + A/t)
                    df_suth.at[i, "Viscosité Mu (Pa.s)"] = mu
                elif pd.isna(t) and not pd.isna(mu) and mu > 0:
                    # Sutherland inverse -> fsolve (résolution numérique)
                    def eq(T_guess): return ((B * np.sqrt(T_guess)) / (1 + A/T_guess)) - mu
                    t_sol = fsolve(eq, 288.15)[0]
                    df_suth.at[i, f"Température T ({u_t6})"] = convert_from_SI(t_sol, u_t6, "Temperature")
            st.session_state.suth_data = df_suth
            st.rerun()

with col6:
    with st.expander("ℹ️ Atmosphère Standard (ISA)"):
        st.write("**Cas d'usage:** Pression en fonction de l'altitude géopotentielle.")
        st.latex(r"Z < 11 km : P = P_0 \left( 1 - \frac{s \cdot Z}{100 \cdot T_0} \right)^{\frac{100 \cdot g}{r \cdot s}}")
        st.latex(r"Z > 11 km : P = P_{11} \cdot \exp\left(-\frac{g \cdot (Z - 11000)}{r \cdot T_{11}}\right)")
        
    st.markdown("**Modèle ISA (Pression / Altitude)**")
    
    col_u7_z, col_u7_p = st.columns(2)
    with col_u7_z: u_z7 = st.selectbox("Unité Altitude", ["m", "ft", "km"], key="u_z7")
    with col_u7_p: u_p7 = st.selectbox("Unité Pression", ["Pa", "bar", "atm"], key="u_p7")
    
    if 'isa_data' not in st.session_state:
        st.session_state.isa_data = pd.DataFrame({
            f"Altitude Z ({u_z7})": [2400.0, 20000.0, np.nan], 
            f"Pression P ({u_p7})": [np.nan, np.nan, 50000.0],
            "P0 (Pa)": [101325.0]*3, "T0 (K)": [288.15]*3, 
            "s (K/100m)": [0.65]*3
        })
        
    st.session_state.isa_data.columns = [f"Altitude Z ({u_z7})", f"Pression P ({u_p7})", "P0 (Pa)", "T0 (K)", "s (K/100m)"]

    with st.form("form_isa"):
        df_isa = st.data_editor(st.session_state.isa_data, num_rows="dynamic", use_container_width=True)
        if st.form_submit_button("Calculer"):
            for i, row in df_isa.iterrows():
                z = convert_to_SI(row[f"Altitude Z ({u_z7})"], u_z7, "Distance")
                p = convert_to_SI(row[f"Pression P ({u_p7})"], u_p7, "Pression")
                p0, t0, s = row["P0 (Pa)"], row["T0 (K)"], row["s (K/100m)"]
                g, r = 9.81, 287.1
                
                # Constantes stratosphère
                p11 = p0 * (1 - (s*11000)/(100*t0))**((100*g)/(r*s))
                t11 = t0 - (s*11000/100)
                
                if pd.isna(p) and not pd.isna(z):
                    if z <= 11000:
                        p = p0 * (1 - (s*z)/(100*t0))**((100*g)/(r*s))
                    else:
                        p = p11 * np.exp(-g*(z-11000)/(r*t11))
                    df_isa.at[i, f"Pression P ({u_p7})"] = convert_from_SI(p, u_p7, "Pression")
                
                elif pd.isna(z) and not pd.isna(p):
                    if p >= p11:
                        z = (100*t0/s) * (1 - (p/p0)**((r*s)/(100*g)))
                    else:
                        z = 11000 - (np.log(p/p11) * r * t11 / g)
                    df_isa.at[i, f"Altitude Z ({u_z7})"] = convert_from_SI(z, u_z7, "Distance")
                    
            st.session_state.isa_data = df_isa
            st.rerun()
