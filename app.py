import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import fsolve

st.set_page_config(page_title="Appli Aéro Turbo-Boostée", layout="wide", page_icon="🚀")

st.title("🚀 Appli Aéro Turbo-Boostée")
st.markdown("Outil de résolution aérodynamique et mécanique des fluides multi-régimes (Solveur bidirectionnel).")

# --- Dictionnaires de conversion ---
unit_factors = {
    "Pression": {"Pa": 1.0, "bar": 1e5, "atm": 101325.0},
    "Vitesse": {"m/s": 1.0, "km/h": 1/3.6, "kt": 1/1.94384},
    "Distance": {"m": 1.0, "ft": 0.3048, "km": 1000.0},
    "Temperature": {"K": (1.0, 0.0), "°C": (1.0, 273.15)}
}

def convert_to_SI(val, unit, category):
    if pd.isna(val): return val
    if category == "Temperature": return val * unit_factors[category][unit][0] + unit_factors[category][unit][1]
    return val * unit_factors[category][unit]

def convert_from_SI(val, unit, category):
    if pd.isna(val): return val
    if category == "Temperature": return (val - unit_factors[category][unit][1]) / unit_factors[category][unit][0]
    return val / unit_factors[category][unit]

# --- Moteur du solveur : Détection des changements ---
def get_changed_cols(row_new, row_old):
    changed = []
    for col in row_new.index:
        v_new, v_old = row_new[col], row_old[col]
        if pd.isna(v_new) and pd.isna(v_old): continue
        if pd.isna(v_new) or pd.isna(v_old) or v_new != v_old: changed.append(col)
    return changed

def get_target(vars_list, changed_cols, row):
    for v in vars_list:
        if pd.isna(row[v]): return v
    unmodified = [v for v in vars_list if v not in changed_cols]
    if unmodified: return unmodified[0]
    return vars_list[0]

def init_state(key, default_df):
    if key not in st.session_state:
        st.session_state[key] = default_df.copy()
        st.session_state[f"{key}_prev"] = default_df.copy()
    else:
        st.session_state[f"{key}_prev"].columns = st.session_state[key].columns

# --- Création des onglets ---
tab1, tab2, tab3 = st.tabs(["📊 Expérimental", "📐 Similitude & Hélices", "💨 Compressible"])

# =========================================================================
# ONGLET 1 : EXPERIMENTAL
# =========================================================================
with tab1:
    st.header("1. Hydrostatique & Manomètres")
    col1, col2 = st.columns(2)

    with col1:
        with st.expander("ℹ️ Manomètre à Eau"):
            st.markdown("**Cas d'usage:** Mesure de faibles différences de pression.\n\n- **Hypothèses:** Fluide incompressible, équilibre hydrostatique.\n- **Limites:** Inadapté aux hautes pressions.")
            st.latex(r"\Delta P = \rho_{eau} \cdot g \cdot \Delta h")
            
        col_u1_p, col_u1_h = st.columns(2)
        with col_u1_p: u_p1 = st.selectbox("Unité Pression", ["Pa", "bar"], key="u_p1")
        with col_u1_h: u_h1 = st.selectbox("Unité Hauteur", ["m", "cm", "mm"], key="u_h1")
        factor_h1 = {"m":1.0, "cm":0.01, "mm":0.001}[u_h1]
        
        cols = ["Rho_eau (kg/m3)", "g (m/s2)", f"Delta P ({u_p1})", f"Delta h ({u_h1})"]
        init_state('eau', pd.DataFrame([[997.0, 9.81, np.nan, np.nan]], columns=cols))
        st.session_state['eau'].columns = cols

        with st.form("form_eau"):
            df_eau = st.data_editor(st.session_state['eau'], num_rows="dynamic", use_container_width=True)
            if st.form_submit_button("Calculer Eau"):
                for i, row in df_eau.iterrows():
                    changed = get_changed_cols(row, st.session_state['eau_prev'].iloc[i]) if i < len(st.session_state['eau_prev']) else list(row.index)
                    target = get_target([f"Delta P ({u_p1})", f"Delta h ({u_h1})"], changed, row)
                    rho, g = row["Rho_eau (kg/m3)"], row["g (m/s2)"]
                    
                    if target == f"Delta P ({u_p1})" and not pd.isna(row[f"Delta h ({u_h1})"]):
                        df_eau.at[i, target] = convert_from_SI(rho * g * (row[f"Delta h ({u_h1})"] * factor_h1), u_p1, "Pression")
                    elif target == f"Delta h ({u_h1})" and not pd.isna(row[f"Delta P ({u_p1})"]):
                        df_eau.at[i, target] = (convert_to_SI(row[f"Delta P ({u_p1})"], u_p1, "Pression") / (rho * g)) / factor_h1
                st.session_state['eau'] = df_eau
                st.session_state['eau_prev'] = df_eau.copy()
                st.rerun()

    with col2:
        with st.expander("ℹ️ Manomètre à Mercure"):
            st.markdown("**Cas d'usage:** Mesure de pressions atmosphériques.\n\n- **Hypothèses:** Incompressible, densité constante.\n- **Limites:** Hauteur de colonne limitée.")
            st.latex(r"\Delta P = \rho_{Hg} \cdot g \cdot \Delta h")
            
        col_u2_p, col_u2_h = st.columns(2)
        with col_u2_p: u_p2 = st.selectbox("Unité Pression", ["Pa", "bar"], key="u_p2")
        with col_u2_h: u_h2 = st.selectbox("Unité Hauteur", ["m", "cm", "mm"], key="u_h2")
        factor_h2 = {"m":1.0, "cm":0.01, "mm":0.001}[u_h2]
        
        cols = ["Rho_Hg (kg/m3)", "g (m/s2)", f"Delta P ({u_p2})", f"Delta h ({u_h2})"]
        init_state('hg', pd.DataFrame([[13600.0, 9.81, np.nan, np.nan]], columns=cols))
        st.session_state['hg'].columns = cols

        with st.form("form_hg"):
            df_hg = st.data_editor(st.session_state['hg'], num_rows="dynamic", use_container_width=True)
            if st.form_submit_button("Calculer Mercure"):
                for i, row in df_hg.iterrows():
                    changed = get_changed_cols(row, st.session_state['hg_prev'].iloc[i]) if i < len(st.session_state['hg_prev']) else list(row.index)
                    target = get_target([f"Delta P ({u_p2})", f"Delta h ({u_h2})"], changed, row)
                    rho, g = row["Rho_Hg (kg/m3)"], row["g (m/s2)"]
                    
                    if target == f"Delta P ({u_p2})" and not pd.isna(row[f"Delta h ({u_h2})"]):
                        df_hg.at[i, target] = convert_from_SI(rho * g * (row[f"Delta h ({u_h2})"] * factor_h2), u_p2, "Pression")
                    elif target == f"Delta h ({u_h2})" and not pd.isna(row[f"Delta P ({u_p2})"]):
                        df_hg.at[i, target] = (convert_to_SI(row[f"Delta P ({u_p2})"], u_p2, "Pression") / (rho * g)) / factor_h2
                st.session_state['hg'] = df_hg
                st.session_state['hg_prev'] = df_hg.copy()
                st.rerun()

    st.divider()
    st.header("2. Dynamique & Vitesse")
    col3, col4 = st.columns(2)

    with col3:
        with st.expander("ℹ️ Loi de Torricelli"):
            st.markdown("**Cas d'usage:** Vitesse d'écoulement d'un réservoir.\n\n- **Hypothèses:** Fluide parfait, charge constante.\n- **Limites:** Néglige les pertes de charge et la contraction de la veine fluide.")
            st.latex(r"U = \sqrt{2 \cdot g \cdot h}")
            
        col_u3_v, col_u3_h = st.columns(2)
        with col_u3_v: u_v3 = st.selectbox("Unité Vitesse", ["m/s", "km/h"], key="u_v3")
        with col_u3_h: u_h3 = st.selectbox("Unité Hauteur", ["m", "ft"], key="u_h3")
        
        cols = ["g (m/s2)", f"Vitesse U ({u_v3})", f"Hauteur h ({u_h3})"]
        init_state('torri', pd.DataFrame([[9.81, np.nan, np.nan]], columns=cols))
        st.session_state['torri'].columns = cols

        with st.form("form_torri"):
            df_torri = st.data_editor(st.session_state['torri'], num_rows="dynamic", use_container_width=True)
            if st.form_submit_button("Calculer Torricelli"):
                for i, row in df_torri.iterrows():
                    changed = get_changed_cols(row, st.session_state['torri_prev'].iloc[i]) if i < len(st.session_state['torri_prev']) else list(row.index)
                    target = get_target([f"Vitesse U ({u_v3})", f"Hauteur h ({u_h3})"], changed, row)
                    g = row["g (m/s2)"]
                    
                    if target == f"Vitesse U ({u_v3})" and not pd.isna(row[f"Hauteur h ({u_h3})"]):
                        h = convert_to_SI(row[f"Hauteur h ({u_h3})"], u_h3, "Distance")
                        if h >= 0: df_torri.at[i, target] = convert_from_SI(np.sqrt(2 * g * h), u_v3, "Vitesse")
                    elif target == f"Hauteur h ({u_h3})" and not pd.isna(row[f"Vitesse U ({u_v3})"]):
                        u = convert_to_SI(row[f"Vitesse U ({u_v3})"], u_v3, "Vitesse")
                        df_torri.at[i, target] = convert_from_SI((u**2) / (2 * g), u_h3, "Distance")
                st.session_state['torri'] = df_torri
                st.session_state['torri_prev'] = df_torri.copy()
                st.rerun()

    with col4:
        with st.expander("ℹ️ Sonde Pitot (Incompressible)"):
            st.markdown("**Cas d'usage:** Calcul de la vitesse via la pression dynamique.\n\n- **Hypothèses:** Mach < 0.3, isentropique.\n- **Limites:** Invalide si fluides compressibles (utiliser les lois de Saint-Venant).")
            st.latex(r"U = \sqrt{\frac{2 \cdot \Delta P}{\rho}}")
            
        col_u4_v, col_u4_p = st.columns(2)
        with col_u4_v: u_v4 = st.selectbox("Unité Vitesse", ["m/s", "km/h", "kt"], key="u_v4")
        with col_u4_p: u_p4 = st.selectbox("Unité Pression", ["Pa", "bar"], key="u_p4")
        
        cols = ["Rho_air (kg/m3)", f"Delta P ({u_p4})", f"Vitesse U ({u_v4})"]
        init_state('pitot', pd.DataFrame([[1.225, np.nan, np.nan]], columns=cols))
        st.session_state['pitot'].columns = cols

        with st.form("form_pitot"):
            df_pitot = st.data_editor(st.session_state['pitot'], num_rows="dynamic", use_container_width=True)
            if st.form_submit_button("Calculer Pitot"):
                for i, row in df_pitot.iterrows():
                    changed = get_changed_cols(row, st.session_state['pitot_prev'].iloc[i]) if i < len(st.session_state['pitot_prev']) else list(row.index)
                    target = get_target([f"Vitesse U ({u_v4})", f"Delta P ({u_p4})"], changed, row)
                    rho = row["Rho_air (kg/m3)"]
                    
                    if target == f"Vitesse U ({u_v4})" and not pd.isna(row[f"Delta P ({u_p4})"]):
                        p = convert_to_SI(row[f"Delta P ({u_p4})"], u_p4, "Pression")
                        if p >= 0: df_pitot.at[i, target] = convert_from_SI(np.sqrt(2 * p / rho), u_v4, "Vitesse")
                    elif target == f"Delta P ({u_p4})" and not pd.isna(row[f"Vitesse U ({u_v4})"]):
                        u = convert_to_SI(row[f"Vitesse U ({u_v4})"], u_v4, "Vitesse")
                        df_pitot.at[i, target] = convert_from_SI(0.5 * rho * (u**2), u_p4, "Pression")
                st.session_state['pitot'] = df_pitot
                st.session_state['pitot_prev'] = df_pitot.copy()
                st.rerun()

    st.divider()
    st.header("3. Thermodynamique & Atmosphère")
    
    with st.expander("ℹ️ Loi des Gaz Parfaits"):
        st.markdown("**Cas d'usage:** Équation d'état.\n\n- **Hypothèses:** Particules ponctuelles, chocs élastiques.\n- **Limites:** Faux sous très hautes pressions.")
        st.latex(r"P = \rho \cdot r \cdot T")

    col_u5_p, col_u5_t = st.columns(2)
    with col_u5_p: u_p5 = st.selectbox("Unité Pression", ["Pa", "bar", "atm"], key="u_p5_2")
    with col_u5_t: u_t5 = st.selectbox("Unité Température", ["K", "°C"], key="u_t5_2")

    cols = ["Constante r", f"Pression ({u_p5})", "Rho (kg/m3)", f"Température ({u_t5})"]
    init_state('gaz', pd.DataFrame([[287.05, np.nan, np.nan, np.nan]], columns=cols))
    st.session_state['gaz'].columns = cols

    with st.form("form_gaz"):
        df_gaz = st.data_editor(st.session_state['gaz'], num_rows="dynamic", use_container_width=True)
        if st.form_submit_button("Calculer Gaz Parfaits"):
            for i, row in df_gaz.iterrows():
                changed = get_changed_cols(row, st.session_state['gaz_prev'].iloc[i]) if i < len(st.session_state['gaz_prev']) else list(row.index)
                target = get_target([f"Pression ({u_p5})", "Rho (kg/m3)", f"Température ({u_t5})"], changed, row)
                
                r = row["Constante r"]
                p = convert_to_SI(row[f"Pression ({u_p5})"], u_p5, "Pression")
                t = convert_to_SI(row[f"Température ({u_t5})"], u_t5, "Temperature")
                rho = row["Rho (kg/m3)"]
                
                if target == f"Pression ({u_p5})" and not pd.isna(rho) and not pd.isna(t):
                    df_gaz.at[i, target] = convert_from_SI(rho * r * t, u_p5, "Pression")
                elif target == "Rho (kg/m3)" and not pd.isna(p) and not pd.isna(t) and t > 0:
                    df_gaz.at[i, target] = p / (r * t)
                elif target == f"Température ({u_t5})" and not pd.isna(p) and not pd.isna(rho) and rho > 0:
                    df_gaz.at[i, target] = convert_from_SI(p / (rho * r), u_t5, "Temperature")
            st.session_state['gaz'] = df_gaz
            st.session_state['gaz_prev'] = df_gaz.copy()
            st.rerun()

    col5, col6 = st.columns(2)

    with col5:
        with st.expander("ℹ️ Loi de Sutherland"):
            st.markdown("**Cas d'usage:** Viscosité dynamique de l'air.\n\n- **Hypothèses:** Modèle empirique cinétique.\n- **Limites:** Valable uniquement entre ~100K et 1900K.")
            st.latex(r"\mu \approx \frac{B \sqrt{T}}{1 + A/T}")
            
        u_t6 = st.selectbox("Unité Température", ["K", "°C"], key="u_t6")
        cols = ["Const. A", "Const. B", f"Temp T ({u_t6})", "Mu (Pa.s)"]
        init_state('suth', pd.DataFrame([[114.0, 1.458e-6, np.nan, np.nan]], columns=cols))
        st.session_state['suth'].columns = cols

        with st.form("form_suth"):
            df_suth = st.data_editor(st.session_state['suth'], num_rows="dynamic", use_container_width=True)
            if st.form_submit_button("Calculer Sutherland"):
                for i, row in df_suth.iterrows():
                    changed = get_changed_cols(row, st.session_state['suth_prev'].iloc[i]) if i < len(st.session_state['suth_prev']) else list(row.index)
                    target = get_target([f"Temp T ({u_t6})", "Mu (Pa.s)"], changed, row)
                    
                    A, B = row["Const. A"], row["Const. B"]
                    t = convert_to_SI(row[f"Temp T ({u_t6})"], u_t6, "Temperature")
                    mu = row["Mu (Pa.s)"]
                    
                    if target == "Mu (Pa.s)" and not pd.isna(t) and t > 0:
                        df_suth.at[i, target] = (B * np.sqrt(t)) / (1 + A/t)
                    elif target == f"Temp T ({u_t6})" and not pd.isna(mu) and mu > 0:
                        def eq(T_guess): return ((B * np.sqrt(T_guess)) / (1 + A/T_guess)) - mu
                        t_sol = fsolve(eq, 288.15)[0]
                        df_suth.at[i, target] = convert_from_SI(t_sol, u_t6, "Temperature")
                st.session_state['suth'] = df_suth
                st.session_state['suth_prev'] = df_suth.copy()
                st.rerun()

    with col6:
        with st.expander("ℹ️ Atmosphère Standard (ISA)"):
            st.markdown("**Cas d'usage:** Modèle standard de Pression et Température.\n\n- **Hypothèses:** Atmosphère calme, air sec, gaz parfait.\n- **Limites:** Ne prend pas en compte la météo ni l'humidité.")
            st.latex(r"Z < 11 km : P = P_0 \left( 1 - \frac{s Z}{100 T_0} \right)^{\frac{100 g}{r s}} \quad|\quad Z > 11 : P = P_{11} \exp\left(-\frac{g(Z - 11000)}{r T_{11}}\right)")
            
        col_u7_z, col_u7_p = st.columns(2)
        with col_u7_z: u_z7 = st.selectbox("Unité Altitude", ["m", "ft", "km"], key="u_z7")
        with col_u7_p: u_p7 = st.selectbox("Unité Pression", ["Pa", "bar", "atm"], key="u_p7_isa")
        
        cols = ["P0 (Pa)", "T0 (K)", "s (K/100m)", f"Alt Z ({u_z7})", f"Pression ({u_p7})"]
        init_state('isa', pd.DataFrame([[101325.0, 288.15, 0.65, np.nan, np.nan]], columns=cols))
        st.session_state['isa'].columns = cols

        with st.form("form_isa"):
            df_isa = st.data_editor(st.session_state['isa'], num_rows="dynamic", use_container_width=True)
            if st.form_submit_button("Calculer ISA"):
                for i, row in df_isa.iterrows():
                    changed = get_changed_cols(row, st.session_state['isa_prev'].iloc[i]) if i < len(st.session_state['isa_prev']) else list(row.index)
                    target = get_target([f"Alt Z ({u_z7})", f"Pression ({u_p7})"], changed, row)
                    
                    p0, t0, s = row["P0 (Pa)"], row["T0 (K)"], row["s (K/100m)"]
                    z = convert_to_SI(row[f"Alt Z ({u_z7})"], u_z7, "Distance")
                    p = convert_to_SI(row[f"Pression ({u_p7})"], u_p7, "Pression")
                    g, r = 9.81, 287.05
                    p11 = p0 * (1 - (s*11000)/(100*t0))**((100*g)/(r*s))
                    t11 = t0 - (s*11000/100)
                    
                    if target == f"Pression ({u_p7})" and not pd.isna(z):
                        res = p0 * (1 - (s*z)/(100*t0))**((100*g)/(r*s)) if z <= 11000 else p11 * np.exp(-g*(z-11000)/(r*t11))
                        df_isa.at[i, target] = convert_from_SI(res, u_p7, "Pression")
                    elif target == f"Alt Z ({u_z7})" and not pd.isna(p):
                        res = (100*t0/s) * (1 - (p/p0)**((r*s)/(100*g))) if p >= p11 else 11000 - (np.log(p/p11) * r * t11 / g)
                        df_isa.at[i, target] = convert_from_SI(res, u_z7, "Distance")
                st.session_state['isa'] = df_isa
                st.session_state['isa_prev'] = df_isa.copy()
                st.rerun()

# =========================================================================
# ONGLET 2 : SIMILITUDE & COEFFICIENTS
# =========================================================================
with tab2:
    st.header("1. Nombres Adimensionnels")
    with st.expander("ℹ️ Reynolds, Froude, Mach, Strouhal"):
        st.markdown("**Cas d'usage:** Similitude entre modèle réduit et réel.\n\n- **Hypothèses:** Conservation des régimes d'écoulement.\n- **Limites:** Impossible d'égaler Re et Mach simultanément en soufflerie subsonique.")
        st.latex(r"Re = \frac{\rho L U}{\mu} \quad|\quad Fr = \frac{U}{\sqrt{g L}} \quad|\quad M = \frac{U}{\sqrt{\gamma r T}} \quad|\quad St = \frac{f L}{U}")
    
    cols = ["Rho", "Mu", "Gamma", "r", "T(K)", "g", "freq(Hz)", "L(m)", "U(m/s)", "Reynolds", "Froude", "Mach", "Strouhal"]
    init_state('adim', pd.DataFrame([[1.225, 1.81e-5, 1.4, 287.05, 288.15, 9.81, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]], columns=cols))
    
    with st.form("form_adim"):
        df_adim = st.data_editor(st.session_state['adim'], num_rows="dynamic", use_container_width=True)
        if st.form_submit_button("Calculer Similitude"):
            for i, row in df_adim.iterrows():
                changed = get_changed_cols(row, st.session_state['adim_prev'].iloc[i]) if i < len(st.session_state['adim_prev']) else list(row.index)
                rho, mu, y, r, T, g, freq = row["Rho"], row["Mu"], row["Gamma"], row["r"], row["T(K)"], row["g"], row["freq(Hz)"]
                L, U = row["L(m)"], row["U(m/s)"]
                
                # Reverse pour U
                if "Mach" in changed and not pd.isna(row["Mach"]): U = row["Mach"] * np.sqrt(y * r * T); df_adim.at[i, "U(m/s)"] = U
                elif "Reynolds" in changed and not pd.isna(row["Reynolds"]) and not pd.isna(L): U = (row["Reynolds"] * mu) / (rho * L); df_adim.at[i, "U(m/s)"] = U
                elif "Froude" in changed and not pd.isna(row["Froude"]) and not pd.isna(L): U = row["Froude"] * np.sqrt(g * L); df_adim.at[i, "U(m/s)"] = U
                
                # Forward
                if not pd.isna(U) and not pd.isna(L):
                    if "Reynolds" not in changed: df_adim.at[i, "Reynolds"] = (rho * L * U) / mu
                    if "Froude" not in changed: df_adim.at[i, "Froude"] = U / np.sqrt(g * L)
                    if "Mach" not in changed: df_adim.at[i, "Mach"] = U / np.sqrt(y * r * T)
                    if "Strouhal" not in changed and not pd.isna(freq): df_adim.at[i, "Strouhal"] = (freq * L) / U
            st.session_state['adim'] = df_adim
            st.session_state['adim_prev'] = df_adim.copy()
            st.rerun()

    st.divider()
    st.header("2. Coeffs. Aérodynamiques & Hélices")
    with st.expander("ℹ️ Forces et Rendement Propulsif"):
        st.markdown("**Cas d'usage:** Calcul des efforts aérodynamiques globaux et de l'efficacité d'une hélice.\n\n- **Hypothèses:** Les coefficients (Ct, Cp) sont valides pour un pas fixe.\n- **Limites:** Rendement s'effondre en transsonique.")
        st.latex(r"C_F = \frac{F}{\frac{1}{2}\rho U^2 S} \quad|\quad J = \frac{V}{n \cdot D} \quad|\quad \eta = \frac{C_t \cdot J}{C_{p\_helice}}")

    cols = ["Rho", "S(m2)", "D(m)", "V(m/s)", "n(tr/s)", "Force F(N)", "C_F", "Traction(N)", "Ct", "Puissance(W)", "Cp_helice", "Avancement J", "Rendement"]
    init_state('helice', pd.DataFrame([[1.225, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]], columns=cols))
    
    with st.form("form_helice"):
        df_helice = st.data_editor(st.session_state['helice'], num_rows="dynamic", use_container_width=True)
        if st.form_submit_button("Calculer Coeffs & Hélices"):
            for i, row in df_helice.iterrows():
                changed = get_changed_cols(row, st.session_state['helice_prev'].iloc[i]) if i < len(st.session_state['helice_prev']) else list(row.index)
                rho, S, D, V, n = row["Rho"], row["S(m2)"], row["D(m)"], row["V(m/s)"], row["n(tr/s)"]
                
                # CF et Force
                target_F = get_target(["Force F(N)", "C_F"], changed, row)
                if target_F == "C_F" and not pd.isna(row["Force F(N)"]) and not pd.isna(V) and not pd.isna(S):
                    df_helice.at[i, "C_F"] = row["Force F(N)"] / (0.5 * rho * V**2 * S)
                elif target_F == "Force F(N)" and not pd.isna(row["C_F"]) and not pd.isna(V) and not pd.isna(S):
                    df_helice.at[i, "Force F(N)"] = row["C_F"] * 0.5 * rho * V**2 * S
                
                # Helice : J, Ct, Cp
                if not pd.isna(n) and not pd.isna(D) and n > 0 and D > 0:
                    if not pd.isna(V) and "Avancement J" not in changed: df_helice.at[i, "Avancement J"] = V / (n * D)
                    
                    target_T = get_target(["Traction(N)", "Ct"], changed, row)
                    if target_T == "Ct" and not pd.isna(row["Traction(N)"]): df_helice.at[i, "Ct"] = row["Traction(N)"] / (rho * n**2 * D**4)
                    elif target_T == "Traction(N)" and not pd.isna(row["Ct"]): df_helice.at[i, "Traction(N)"] = row["Ct"] * rho * n**2 * D**4
                    
                    target_P = get_target(["Puissance(W)", "Cp_helice"], changed, row)
                    if target_P == "Cp_helice" and not pd.isna(row["Puissance(W)"]): df_helice.at[i, "Cp_helice"] = row["Puissance(W)"] / (rho * n**3 * D**5)
                    elif target_P == "Puissance(W)" and not pd.isna(row["Cp_helice"]): df_helice.at[i, "Puissance(W)"] = row["Cp_helice"] * rho * n**3 * D**5
                
                # Rendement
                if "Rendement" not in changed:
                    J, Ct, Cp_h = df_helice.at[i, "Avancement J"], df_helice.at[i, "Ct"], df_helice.at[i, "Cp_helice"]
                    if not pd.isna(J) and not pd.isna(Ct) and not pd.isna(Cp_h) and Cp_h != 0:
                        df_helice.at[i, "Rendement"] = (Ct * J) / Cp_h

            st.session_state['helice'] = df_helice
            st.session_state['helice_prev'] = df_helice.copy()
            st.rerun()

# =========================================================================
# ONGLET 3 : COMPRESSIBLE
# =========================================================================
with tab3:
    st.header("1. Relations Isentropiques (Stagnation)")
    with st.expander("ℹ️ Grandeurs génératrices"):
        st.markdown("**Cas d'usage:** Décélération théorique isentropique d'un gaz jusqu'à l'arrêt.\n\n- **Hypothèses:** Écoulement adiabatique, réversible (sans choc).\n- **Limites:** Non valide si présence d'ondes de chocs (dissipation).")
        st.latex(r"\frac{T_i}{T} = 1 + \frac{\gamma-1}{2}M^2 \quad|\quad \frac{P_i}{P} = \left(\frac{T_i}{T}\right)^{\frac{\gamma}{\gamma-1}}")
    
    cols = ["Gamma", "Mach M", "T_i / T", "P_i / P", "Rho_i / Rho"]
    init_state('isen', pd.DataFrame([[1.4, np.nan, np.nan, np.nan, np.nan]], columns=cols))
        
    with st.form("form_isen"):
        df_isen = st.data_editor(st.session_state['isen'], num_rows="dynamic", use_container_width=True)
        if st.form_submit_button("Calculer Ratios Isentropiques"):
            for i, row in df_isen.iterrows():
                changed = get_changed_cols(row, st.session_state['isen_prev'].iloc[i]) if i < len(st.session_state['isen_prev']) else list(row.index)
                y, M = row["Gamma"], row["Mach M"]
                
                if "T_i / T" in changed and not pd.isna(row["T_i / T"]): M = np.sqrt(max(0, (row["T_i / T"] - 1) * 2 / (y - 1)))
                elif "P_i / P" in changed and not pd.isna(row["P_i / P"]): M = np.sqrt(max(0, (row["P_i / P"]**((y-1)/y) - 1) * 2 / (y - 1)))
                elif "Rho_i / Rho" in changed and not pd.isna(row["Rho_i / Rho"]): M = np.sqrt(max(0, (row["Rho_i / Rho"]**(y-1) - 1) * 2 / (y - 1)))
                df_isen.at[i, "Mach M"] = M
                
                if not pd.isna(M):
                    if "T_i / T" not in changed: df_isen.at[i, "T_i / T"] = 1 + ((y-1)/2) * M**2
                    if "P_i / P" not in changed: df_isen.at[i, "P_i / P"] = (1 + ((y-1)/2) * M**2)**(y/(y-1))
                    if "Rho_i / Rho" not in changed: df_isen.at[i, "Rho_i / Rho"] = (1 + ((y-1)/2) * M**2)**(1/(y-1))
            st.session_state['isen'] = df_isen
            st.session_state['isen_prev'] = df_isen.copy()
            st.rerun()

    st.divider()
    st.header("2. Tuyères et Chocs Droits")
    with st.expander("ℹ️ Section Sonique & Relations de Rankine-Hugoniot"):
        st.markdown("**Cas d'usage:** Tuyères de Laval (A/A*) et sauts à travers un choc droit.\n\n- **Hypothèses:** Écoulement 1D quasi-stationnaire. Choc infiniment mince.\n- **Limites:** Non valable pour chocs obliques ou couches limites épaisses.")
        st.latex(r"\frac{A}{A^*} = \frac{1}{M} \left[ \frac{2}{\gamma+1} \left( 1 + \frac{\gamma-1}{2}M^2 \right) \right]^{\frac{\gamma+1}{2(\gamma-1)}} \quad|\quad M_2 = \sqrt{\frac{1 + \frac{\gamma-1}{2}M_1^2}{\gamma M_1^2 - \frac{\gamma-1}{2}}}")

    cols = ["Gamma", "Mach amont M1", "Rapport A / A*", "Régime inverse", "Mach aval M2", "Rapport P2/P1"]
    init_state('choc', pd.DataFrame([[1.4, np.nan, np.nan, "Subsonique", np.nan, np.nan]], columns=cols))
        
    with st.form("form_choc"):
        df_choc = st.data_editor(st.session_state['choc'], column_config={"Régime inverse": st.column_config.SelectboxColumn(options=["Subsonique", "Supersonique"])}, num_rows="dynamic", use_container_width=True)
        if st.form_submit_button("Calculer Chocs & Tuyères"):
            for i, row in df_choc.iterrows():
                changed = get_changed_cols(row, st.session_state['choc_prev'].iloc[i]) if i < len(st.session_state['choc_prev']) else list(row.index)
                y, M1, A_ratio, regime = row["Gamma"], row["Mach amont M1"], row["Rapport A / A*"], row["Régime inverse"]
                
                # Tuyères : Solveur A/A* -> M1
                if "Rapport A / A*" in changed and not pd.isna(A_ratio) and A_ratio >= 1.0:
                    def eq_A(M_guess):
                        if M_guess <= 0: return 9999
                        return (1/M_guess) * ((2/(y+1))*(1 + ((y-1)/2)*M_guess**2))**((y+1)/(2*(y-1))) - A_ratio
                    M1 = fsolve(eq_A, 0.5 if regime == "Subsonique" else 2.0)[0]
                    df_choc.at[i, "Mach amont M1"] = M1
                    
                # Tuyères : Forward M1 -> A/A*
                elif "Mach amont M1" in changed and not pd.isna(M1) and M1 > 0:
                    term = (2/(y+1))*(1 + ((y-1)/2)*M1**2)
                    df_choc.at[i, "Rapport A / A*"] = (1/M1) * term**((y+1)/(2*(y-1)))

                # Chocs Droits (seulement si M1 > 1)
                if not pd.isna(M1) and M1 > 1.0:
                    if "Mach aval M2" not in changed:
                        num = 1 + ((y-1)/2)*M1**2
                        den = y*M1**2 - (y-1)/2
                        df_choc.at[i, "Mach aval M2"] = np.sqrt(num / den) if den > 0 else np.nan
                    if "Rapport P2/P1" not in changed:
                        df_choc.at[i, "Rapport P2/P1"] = 1 + (2*y/(y+1))*(M1**2 - 1)
                elif not pd.isna(M1) and M1 <= 1.0:
                    df_choc.at[i, "Mach aval M2"] = np.nan
                    df_choc.at[i, "Rapport P2/P1"] = np.nan

            st.session_state['choc'] = df_choc
            st.session_state['choc_prev'] = df_choc.copy()
            st.rerun()
