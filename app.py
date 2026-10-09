import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import fsolve

st.set_page_config(page_title="Appli Aéro Turbo-Boostée", layout="wide", page_icon="🚀")

st.title("🚀 Appli Aéro Turbo-Boostée")
st.markdown("Outil de résolution aérodynamique et mécanique des fluides multi-régimes.")

# --- Dictionnaires de conversion ---
unit_factors = {
    "Pression": {"Pa": 1.0, "bar": 1e5, "atm": 101325.0},
    "Vitesse": {"m/s": 1.0, "km/h": 1/3.6, "kt": 1/1.94384},
    "Distance": {"m": 1.0, "ft": 0.3048, "km": 1000.0},
    "Temperature": {"K": (1.0, 0.0), "°C": (1.0, 273.15)}
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
            st.markdown("**Cas d'usage:** Mesure de faibles différences de pression.")
            st.latex(r"\Delta P = \rho_{eau} \cdot g \cdot \Delta h")
            st.markdown("- **Hypothèses:** Fluide incompressible, équilibre hydrostatique.\n- **Limites:** Inadapté aux hautes pressions (la colonne d'eau serait trop haute).")
            
        col_u1_p, col_u1_h = st.columns(2)
        with col_u1_p: u_p1 = st.selectbox("Unité Pression", ["Pa", "bar"], key="u_p1")
        with col_u1_h: u_h1 = st.selectbox("Unité Hauteur", ["m", "cm", "mm"], key="u_h1")
        factor_h1 = {"m":1.0, "cm":0.01, "mm":0.001}[u_h1]
        
        if 'eau_data' not in st.session_state:
            st.session_state.eau_data = pd.DataFrame({"Rho_eau (kg/m3)": [997.0], "g (m/s2)": [9.81], f"Delta P ({u_p1})": [np.nan], f"Delta h ({u_h1})": [np.nan]})
        st.session_state.eau_data.columns = ["Rho_eau (kg/m3)", "g (m/s2)", f"Delta P ({u_p1})", f"Delta h ({u_h1})"]

        with st.form("form_eau"):
            df_eau = st.data_editor(st.session_state.eau_data, num_rows="dynamic", use_container_width=True)
            if st.form_submit_button("Calculer"):
                for i, row in df_eau.iterrows():
                    rho, g = row["Rho_eau (kg/m3)"], row["g (m/s2)"]
                    p_val = convert_to_SI(row[f"Delta P ({u_p1})"], u_p1, "Pression")
                    h_val = row[f"Delta h ({u_h1})"] * factor_h1
                    
                    if pd.isna(p_val) and not pd.isna(h_val):
                        df_eau.at[i, f"Delta P ({u_p1})"] = convert_from_SI(rho * g * h_val, u_p1, "Pression")
                    elif pd.isna(h_val) and not pd.isna(p_val):
                        df_eau.at[i, f"Delta h ({u_h1})"] = (p_val / (rho * g)) / factor_h1
                st.session_state.eau_data = df_eau
                st.rerun()

    with col2:
        with st.expander("ℹ️ Manomètre à Mercure"):
            st.markdown("**Cas d'usage:** Mesure de pressions atmosphériques (Baromètre de Torricelli).")
            st.latex(r"\Delta P = \rho_{Hg} \cdot g \cdot \Delta h")
            st.markdown("- **Hypothèses:** Fluide incompressible, densité constante.\n- **Limites:** Utilisation restreinte due à la toxicité du mercure.")
            
        col_u2_p, col_u2_h = st.columns(2)
        with col_u2_p: u_p2 = st.selectbox("Unité Pression", ["Pa", "bar"], key="u_p2")
        with col_u2_h: u_h2 = st.selectbox("Unité Hauteur", ["m", "cm", "mm"], key="u_h2")
        factor_h2 = {"m":1.0, "cm":0.01, "mm":0.001}[u_h2]
        
        if 'hg_data' not in st.session_state:
            st.session_state.hg_data = pd.DataFrame({"Rho_Hg (kg/m3)": [13600.0], "g (m/s2)": [9.81], f"Delta P ({u_p2})": [np.nan], f"Delta h ({u_h2})": [np.nan]})
        st.session_state.hg_data.columns = ["Rho_Hg (kg/m3)", "g (m/s2)", f"Delta P ({u_p2})", f"Delta h ({u_h2})"]

        with st.form("form_hg"):
            df_hg = st.data_editor(st.session_state.hg_data, num_rows="dynamic", use_container_width=True)
            if st.form_submit_button("Calculer"):
                for i, row in df_hg.iterrows():
                    rho, g = row["Rho_Hg (kg/m3)"], row["g (m/s2)"]
                    p_val = convert_to_SI(row[f"Delta P ({u_p2})"], u_p2, "Pression")
                    h_val = row[f"Delta h ({u_h2})"] * factor_h2
                    
                    if pd.isna(p_val) and not pd.isna(h_val):
                        df_hg.at[i, f"Delta P ({u_p2})"] = convert_from_SI(rho * g * h_val, u_p2, "Pression")
                    elif pd.isna(h_val) and not pd.isna(p_val):
                        df_hg.at[i, f"Delta h ({u_h2})"] = (p_val / (rho * g)) / factor_h2
                st.session_state.hg_data = df_hg
                st.rerun()

    st.divider()
    st.header("2. Dynamique & Vitesse")
    col3, col4 = st.columns(2)

    with col3:
        with st.expander("ℹ️ Loi de Torricelli"):
            st.markdown("**Cas d'usage:** Vitesse d'écoulement d'un réservoir à l'air libre.")
            st.latex(r"U = \sqrt{2 \cdot g \cdot h}")
            st.markdown("- **Hypothèses:** Fluide parfait, charge constante, surface libre grande.\n- **Limites:** Néglige les pertes de charge et la contraction de la veine fluide.")
            
        col_u3_v, col_u3_h = st.columns(2)
        with col_u3_v: u_v3 = st.selectbox("Unité Vitesse", ["m/s", "km/h"], key="u_v3")
        with col_u3_h: u_h3 = st.selectbox("Unité Hauteur", ["m", "ft"], key="u_h3")
        
        if 'torri_data' not in st.session_state:
            st.session_state.torri_data = pd.DataFrame({"g (m/s2)": [9.81], f"Vitesse U ({u_v3})": [np.nan], f"Hauteur h ({u_h3})": [np.nan]})
        st.session_state.torri_data.columns = ["g (m/s2)", f"Vitesse U ({u_v3})", f"Hauteur h ({u_h3})"]

        with st.form("form_torri"):
            df_torri = st.data_editor(st.session_state.torri_data, num_rows="dynamic", use_container_width=True)
            if st.form_submit_button("Calculer"):
                for i, row in df_torri.iterrows():
                    g = row["g (m/s2)"]
                    u_val = convert_to_SI(row[f"Vitesse U ({u_v3})"], u_v3, "Vitesse")
                    h_val = convert_to_SI(row[f"Hauteur h ({u_h3})"], u_h3, "Distance")
                    
                    if pd.isna(u_val) and not pd.isna(h_val) and h_val >= 0:
                        df_torri.at[i, f"Vitesse U ({u_v3})"] = convert_from_SI(np.sqrt(2 * g * h_val), u_v3, "Vitesse")
                    elif pd.isna(h_val) and not pd.isna(u_val):
                        df_torri.at[i, f"Hauteur h ({u_h3})"] = convert_from_SI((u_val**2) / (2 * g), u_h3, "Distance")
                st.session_state.torri_data = df_torri
                st.rerun()

    with col4:
        with st.expander("ℹ️ Sonde Pitot (Incompressible)"):
            st.markdown("**Cas d'usage:** Calcul de la vitesse vraie ou indiquée d'un aéronef via la pression dynamique.")
            st.latex(r"U = \sqrt{\frac{2 \cdot \Delta P}{\rho}}")
            st.markdown("- **Hypothèses:** Écoulement incompressible (Mach < 0.3), isentropique.\n- **Limites:** Invalide si effets de compressibilité (utiliser les lois de Saint-Venant).")
            
        col_u4_v, col_u4_p = st.columns(2)
        with col_u4_v: u_v4 = st.selectbox("Unité Vitesse", ["m/s", "km/h", "kt"], key="u_v4")
        with col_u4_p: u_p4 = st.selectbox("Unité Pression", ["Pa", "bar"], key="u_p4")
        
        if 'pitot_data' not in st.session_state:
            st.session_state.pitot_data = pd.DataFrame({"Rho_air (kg/m3)": [1.225], f"Delta P ({u_p4})": [np.nan], f"Vitesse U ({u_v4})": [np.nan]})
        st.session_state.pitot_data.columns = ["Rho_air (kg/m3)", f"Delta P ({u_p4})", f"Vitesse U ({u_v4})"]

        with st.form("form_pitot"):
            df_pitot = st.data_editor(st.session_state.pitot_data, num_rows="dynamic", use_container_width=True)
            if st.form_submit_button("Calculer"):
                for i, row in df_pitot.iterrows():
                    rho = row["Rho_air (kg/m3)"]
                    p_val = convert_to_SI(row[f"Delta P ({u_p4})"], u_p4, "Pression")
                    u_val = convert_to_SI(row[f"Vitesse U ({u_v4})"], u_v4, "Vitesse")
                    
                    if pd.isna(u_val) and not pd.isna(p_val) and p_val >= 0:
                        df_pitot.at[i, f"Vitesse U ({u_v4})"] = convert_from_SI(np.sqrt(2 * p_val / rho), u_v4, "Vitesse")
                    elif pd.isna(p_val) and not pd.isna(u_val):
                        df_pitot.at[i, f"Delta P ({u_p4})"] = convert_from_SI(0.5 * rho * (u_val**2), u_p4, "Pression")
                st.session_state.pitot_data = df_pitot
                st.rerun()

    st.divider()
    st.header("3. Thermodynamique & Atmosphère")
    
    with st.expander("ℹ️ Loi des Gaz Parfaits"):
        st.markdown("**Cas d'usage:** Lier Pression, Température et Densité pour l'air.")
        st.latex(r"P = \rho \cdot r \cdot T")
        st.markdown("- **Hypothèses:** Molécules ponctuelles sans interactions (hors chocs).\n- **Limites:** Non applicable sous très fortes pressions ou très basses températures.")

    col_u5_p, col_u5_t = st.columns(2)
    with col_u5_p: u_p5 = st.selectbox("Unité Pression", ["Pa", "bar", "atm"], key="u_p5")
    with col_u5_t: u_t5 = st.selectbox("Unité Température", ["K", "°C"], key="u_t5")

    if 'gaz_data' not in st.session_state:
        st.session_state.gaz_data = pd.DataFrame({"Constante r (J/kg.K)": [287.05], f"Pression P ({u_p5})": [np.nan], "Densité Rho (kg/m3)": [np.nan], f"Température T ({u_t5})": [np.nan]})
    st.session_state.gaz_data.columns = ["Constante r (J/kg.K)", f"Pression P ({u_p5})", "Densité Rho (kg/m3)", f"Température T ({u_t5})"]

    with st.form("form_gaz"):
        df_gaz = st.data_editor(st.session_state.gaz_data, num_rows="dynamic", use_container_width=True)
        if st.form_submit_button("Calculer"):
            for i, row in df_gaz.iterrows():
                r = row["Constante r (J/kg.K)"]
                p = convert_to_SI(row[f"Pression P ({u_p5})"], u_p5, "Pression")
                rho = row["Densité Rho (kg/m3)"]
                t = convert_to_SI(row[f"Température T ({u_t5})"], u_t5, "Temperature")
                
                if pd.isna(p) and not pd.isna(rho) and not pd.isna(t):
                    df_gaz.at[i, f"Pression P ({u_p5})"] = convert_from_SI(rho * r * t, u_p5, "Pression")
                elif pd.isna(rho) and not pd.isna(p) and not pd.isna(t) and t > 0:
                    df_gaz.at[i, "Densité Rho (kg/m3)"] = p / (r * t)
                elif pd.isna(t) and not pd.isna(p) and not pd.isna(rho) and rho > 0:
                    df_gaz.at[i, f"Température T ({u_t5})"] = convert_from_SI(p / (rho * r), u_t5, "Temperature")
            st.session_state.gaz_data = df_gaz
            st.rerun()

    col5, col6 = st.columns(2)

    with col5:
        with st.expander("ℹ️ Loi de Sutherland"):
            st.markdown("**Cas d'usage:** Estimer la viscosité dynamique de l'air en fonction de sa température.")
            st.latex(r"\mu \approx \frac{B \sqrt{T}}{1 + A/T}")
            st.markdown("- **Hypothèses:** Gaz dilué, théorie cinétique des gaz.\n- **Limites:** Valable environ entre 100K et 1900K.")
            
        u_t6 = st.selectbox("Unité Température (Sutherland)", ["K", "°C"], key="u_t6")
        
        if 'suth_data' not in st.session_state:
            st.session_state.suth_data = pd.DataFrame({"Constante A": [114.0], "Constante B": [1.458e-6], f"Température T ({u_t6})": [np.nan], "Viscosité Mu (Pa.s)": [np.nan]})
        st.session_state.suth_data.columns = ["Constante A", "Constante B", f"Température T ({u_t6})", "Viscosité Mu (Pa.s)"]

        with st.form("form_suth"):
            df_suth = st.data_editor(st.session_state.suth_data, num_rows="dynamic", use_container_width=True)
            if st.form_submit_button("Calculer"):
                for i, row in df_suth.iterrows():
                    A, B = row["Constante A"], row["Constante B"]
                    t = convert_to_SI(row[f"Température T ({u_t6})"], u_t6, "Temperature")
                    mu = row["Viscosité Mu (Pa.s)"]
                    
                    if pd.isna(mu) and not pd.isna(t) and t > 0:
                        df_suth.at[i, "Viscosité Mu (Pa.s)"] = (B * np.sqrt(t)) / (1 + A/t)
                    elif pd.isna(t) and not pd.isna(mu) and mu > 0:
                        def eq(T_guess): return ((B * np.sqrt(T_guess)) / (1 + A/T_guess)) - mu
                        t_sol = fsolve(eq, 288.15)[0]
                        df_suth.at[i, f"Température T ({u_t6})"] = convert_from_SI(t_sol, u_t6, "Temperature")
                st.session_state.suth_data = df_suth
                st.rerun()

    with col6:
        with st.expander("ℹ️ Atmosphère Standard (ISA)"):
            st.markdown("**Cas d'usage:** Évolution de la Pression et Température atmosphérique.")
            st.latex(r"Z < 11 km : P = P_0 \left( 1 - \frac{s \cdot Z}{100 \cdot T_0} \right)^{\frac{100 \cdot g}{r \cdot s}}")
            st.latex(r"Z > 11 km : P = P_{11} \cdot \exp\left(-\frac{g \cdot (Z - 11000)}{r \cdot T_{11}}\right)")
            st.markdown("- **Hypothèses:** Atmosphère théorique standardisée, fluide statique, gaz parfait.\n- **Limites:** Ne reflète pas la météorologie locale ni l'humidité.")
            
        col_u7_z, col_u7_p = st.columns(2)
        with col_u7_z: u_z7 = st.selectbox("Unité Altitude", ["m", "ft", "km"], key="u_z7")
        with col_u7_p: u_p7 = st.selectbox("Unité Pression", ["Pa", "bar", "atm"], key="u_p7")
        
        if 'isa_data' not in st.session_state:
            st.session_state.isa_data = pd.DataFrame({"P0 (Pa)": [101325.0], "T0 (K)": [288.15], "s (K/100m)": [0.65], f"Altitude Z ({u_z7})": [np.nan], f"Pression P ({u_p7})": [np.nan]})
        st.session_state.isa_data.columns = ["P0 (Pa)", "T0 (K)", "s (K/100m)", f"Altitude Z ({u_z7})", f"Pression P ({u_p7})"]

        with st.form("form_isa"):
            df_isa = st.data_editor(st.session_state.isa_data, num_rows="dynamic", use_container_width=True)
            if st.form_submit_button("Calculer"):
                for i, row in df_isa.iterrows():
                    p0, t0, s = row["P0 (Pa)"], row["T0 (K)"], row["s (K/100m)"]
                    z = convert_to_SI(row[f"Altitude Z ({u_z7})"], u_z7, "Distance")
                    p = convert_to_SI(row[f"Pression P ({u_p7})"], u_p7, "Pression")
                    g, r = 9.81, 287.05
                    p11 = p0 * (1 - (s*11000)/(100*t0))**((100*g)/(r*s))
                    t11 = t0 - (s*11000/100)
                    
                    if pd.isna(p) and not pd.isna(z):
                        if z <= 11000: p = p0 * (1 - (s*z)/(100*t0))**((100*g)/(r*s))
                        else: p = p11 * np.exp(-g*(z-11000)/(r*t11))
                        df_isa.at[i, f"Pression P ({u_p7})"] = convert_from_SI(p, u_p7, "Pression")
                    elif pd.isna(z) and not pd.isna(p):
                        if p >= p11: z = (100*t0/s) * (1 - (p/p0)**((r*s)/(100*g)))
                        else: z = 11000 - (np.log(p/p11) * r * t11 / g)
                        df_isa.at[i, f"Altitude Z ({u_z7})"] = convert_from_SI(z, u_z7, "Distance")
                st.session_state.isa_data = df_isa
                st.rerun()

# =========================================================================
# ONGLET 2 : SIMILITUDE & COEFFICIENTS
# =========================================================================
with tab2:
    st.header("1. Nombres Adimensionnels")
    
    with st.expander("ℹ️ Nombres de Reynolds, Froude et Mach"):
        st.markdown("**Cas d'usage:** Similitude entre maquette en soufflerie et modèle réel.")
        st.latex(r"Re = \frac{\rho \cdot L \cdot U}{\mu} \quad|\quad Fr = \frac{U}{\sqrt{g \cdot L}} \quad|\quad M = \frac{U}{\sqrt{\gamma \cdot r \cdot T}}")
        st.markdown("- **Hypothèses:** Analogie dimensionnelle complète requise pour des résultats fidèles.\n- **Limites:** Impossible d'égaler Re et Mach simultanément en soufflerie classique.")
    
    if 'adim_data' not in st.session_state:
        st.session_state.adim_data = pd.DataFrame({"Rho": [1.225], "Mu": [1.81e-5], "Gamma": [1.4], "r": [287.05], "Temp T(K)": [288.15], "g": [9.81], "L(m)": [np.nan], "U(m/s)": [np.nan], "Reynolds": [np.nan], "Froude": [np.nan], "Mach": [np.nan]})
    
    with st.form("form_adim"):
        df_adim = st.data_editor(st.session_state.adim_data, num_rows="dynamic", use_container_width=True)
        if st.form_submit_button("Calculer Nombres Sans Dimension"):
            for i, row in df_adim.iterrows():
                rho, mu, g_c, r, T, g = row["Rho"], row["Mu"], row["Gamma"], row["r"], row["Temp T(K)"], row["g"]
                L, U = row["L(m)"], row["U(m/s)"]
                
                # Forward calculations if inputs exist
                if not pd.isna(L) and not pd.isna(U):
                    if pd.isna(row["Reynolds"]): df_adim.at[i, "Reynolds"] = (rho * L * U) / mu
                    if pd.isna(row["Froude"]): df_adim.at[i, "Froude"] = U / np.sqrt(g * L)
                    if pd.isna(row["Mach"]): df_adim.at[i, "Mach"] = U / np.sqrt(g_c * r * T)
                # Reverse for U if Mach is given
                elif pd.isna(U) and not pd.isna(row["Mach"]) and not pd.isna(T):
                    U_calc = row["Mach"] * np.sqrt(g_c * r * T)
                    df_adim.at[i, "U(m/s)"] = U_calc
                    if not pd.isna(L): 
                        df_adim.at[i, "Reynolds"] = (rho * L * U_calc) / mu
                        df_adim.at[i, "Froude"] = U_calc / np.sqrt(g * L)
            st.session_state.adim_data = df_adim
            st.rerun()

    st.divider()
    st.header("2. Aérodynamique & Hélices")
    
    with st.expander("ℹ️ Coefficients de force & Paramètres Hélice"):
        st.markdown("**Cas d'usage:** Calcul des forces (Portance, Traînée) et efficacité de l'hélice.")
        st.latex(r"C_F = \frac{F}{\frac{1}{2}\rho U^2 S} \quad|\quad J = \frac{V}{n \cdot D} \quad|\quad \eta = \frac{C_t \cdot J}{C_p}")
        st.markdown("- **Hypothèses:** Les coefficients (Ct, Cp) sont valables pour un pas d'hélice fixe.\n- **Limites:** Rendement s'effondre près des régimes transsoniques en bout de pale.")

    if 'helice_data' not in st.session_state:
        st.session_state.helice_data = pd.DataFrame({"Rho": [1.225], "S(m2)": [np.nan], "D(m)": [np.nan], "V(m/s)": [np.nan], "n(tr/s)": [np.nan], "Force F(N)": [np.nan], "Coeff CF": [np.nan], "Traction T(N)": [np.nan], "Ct": [np.nan], "Puissance(W)": [np.nan], "Cp": [np.nan], "Avancement J": [np.nan], "Rendement": [np.nan]})
    
    with st.form("form_helice"):
        df_helice = st.data_editor(st.session_state.helice_data, num_rows="dynamic", use_container_width=True)
        if st.form_submit_button("Calculer Aéro/Hélice"):
            for i, row in df_helice.iterrows():
                rho, S, D, V, n = row["Rho"], row["S(m2)"], row["D(m)"], row["V(m/s)"], row["n(tr/s)"]
                F, CF = row["Force F(N)"], row["Coeff CF"]
                T, Ct, P, Cp = row["Traction T(N)"], row["Ct"], row["Puissance(W)"], row["Cp"]
                
                # CF <-> Force
                if pd.isna(CF) and not pd.isna(F) and not pd.isna(V) and not pd.isna(S):
                    df_helice.at[i, "Coeff CF"] = F / (0.5 * rho * V**2 * S)
                elif pd.isna(F) and not pd.isna(CF) and not pd.isna(V) and not pd.isna(S):
                    df_helice.at[i, "Force F(N)"] = CF * 0.5 * rho * V**2 * S
                
                # Helice
                if not pd.isna(n) and not pd.isna(D) and n > 0 and D > 0:
                    if not pd.isna(V) and pd.isna(row["Avancement J"]): 
                        df_helice.at[i, "Avancement J"] = V / (n * D)
                    
                    if not pd.isna(T) and pd.isna(Ct): 
                        df_helice.at[i, "Ct"] = T / (rho * n**2 * D**4)
                    elif pd.isna(T) and not pd.isna(Ct): 
                        df_helice.at[i, "Traction T(N)"] = Ct * rho * n**2 * D**4
                        
                    if not pd.isna(P) and pd.isna(Cp): 
                        df_helice.at[i, "Cp"] = P / (rho * n**3 * D**5)
                    elif pd.isna(P) and not pd.isna(Cp): 
                        df_helice.at[i, "Puissance(W)"] = Cp * rho * n**3 * D**5
                        
                # Rendement
                J_val = df_helice.at[i, "Avancement J"]
                Ct_val = df_helice.at[i, "Ct"]
                Cp_val = df_helice.at[i, "Cp"]
                if not pd.isna(J_val) and not pd.isna(Ct_val) and not pd.isna(Cp_val) and Cp_val != 0:
                    df_helice.at[i, "Rendement"] = (Ct_val * J_val) / Cp_val
            st.session_state.helice_data = df_helice
            st.rerun()

# =========================================================================
# ONGLET 3 : COMPRESSIBLE
# =========================================================================
with tab3:
    st.header("1. Relations Isentropiques (Stagnation)")
    
    with st.expander("ℹ️ Grandeurs génératrices et Mach"):
        st.markdown("**Cas d'usage:** Décélération isentropique d'un gaz jusqu'à l'arrêt (points d'arrêt).")
        st.latex(r"\frac{T_i}{T} = 1 + \frac{\gamma-1}{2}M^2 \quad|\quad \frac{P_i}{P} = \left(1 + \frac{\gamma-1}{2}M^2\right)^{\frac{\gamma}{\gamma-1}}")
        st.markdown("- **Hypothèses:** Écoulement adiabatique, réversible (sans frottement ni choc).\n- **Limites:** Cesse d'être valide s'il y a des ondes de choc (dissipation de l'entropie).")

    if 'isen_data' not in st.session_state:
        st.session_state.isen_data = pd.DataFrame({"Gamma": [1.4], "Mach M": [np.nan], "Rapport T_i / T": [np.nan], "Rapport P_i / P": [np.nan], "Rapport Rho_i / Rho": [np.nan]})
        
    with st.form("form_isen"):
        df_isen = st.data_editor(st.session_state.isen_data, num_rows="dynamic", use_container_width=True)
        if st.form_submit_button("Calculer Ratios Isentropiques"):
            for i, row in df_isen.iterrows():
                y = row["Gamma"]
                M, rt, rp, rrho = row["Mach M"], row["Rapport T_i / T"], row["Rapport P_i / P"], row["Rapport Rho_i / Rho"]
                
                # Solve for M if missing
                if pd.isna(M):
                    if not pd.isna(rt): M = np.sqrt(max(0, (rt - 1) * 2 / (y - 1)))
                    elif not pd.isna(rp): M = np.sqrt(max(0, (rp**((y-1)/y) - 1) * 2 / (y - 1)))
                    elif not pd.isna(rrho): M = np.sqrt(max(0, (rrho**(y-1) - 1) * 2 / (y - 1)))
                    df_isen.at[i, "Mach M"] = M
                
                # Compute ratios
                if not pd.isna(M):
                    df_isen.at[i, "Rapport T_i / T"] = 1 + ((y-1)/2) * M**2
                    df_isen.at[i, "Rapport P_i / P"] = (1 + ((y-1)/2) * M**2)**(y/(y-1))
                    df_isen.at[i, "Rapport Rho_i / Rho"] = (1 + ((y-1)/2) * M**2)**(1/(y-1))
            st.session_state.isen_data = df_isen
            st.rerun()

    st.divider()
    st.header("2. Tuyères et Chocs (Relations de Rankine-Hugoniot)")

    with st.expander("ℹ️ Section Sonique (A/A*) & Choc Droit"):
        st.markdown("**Cas d'usage:** Design de tuyère de Laval (A/A*) et calcul des sauts à travers un choc droit.")
        st.latex(r"\frac{A}{A^*} = \frac{1}{M} \left[ \frac{2}{\gamma+1} \left( 1 + \frac{\gamma-1}{2}M^2 \right) \right]^{\frac{\gamma+1}{2(\gamma-1)}} \quad|\quad M_2 = \sqrt{\frac{1 + \frac{\gamma-1}{2}M_1^2}{\gamma M_1^2 - \frac{\gamma-1}{2}}}")
        st.markdown("- **Hypothèses:** Écoulement 1D quasi-stationnaire. Choc infiniment mince.\n- **Limites:** Non valable pour les chocs obliques ou les couches limites épaisses.")

    if 'choc_data' not in st.session_state:
        st.session_state.choc_data = pd.DataFrame({"Gamma": [1.4], "Mach amont M1": [np.nan], "Rapport A / A*": [np.nan], "Régime d'inversion": ["Subsonique"], "Mach aval M2 (Choc)": [np.nan], "Rapport P2/P1 (Choc)": [np.nan]})
        
    with st.form("form_choc"):
        df_choc = st.data_editor(
            st.session_state.choc_data, 
            column_config={"Régime d'inversion": st.column_config.SelectboxColumn(options=["Subsonique", "Supersonique"])},
            num_rows="dynamic", use_container_width=True
        )
        if st.form_submit_button("Calculer Écoulements Supersoniques"):
            for i, row in df_choc.iterrows():
                y = row["Gamma"]
                M1 = row["Mach amont M1"]
                A_ratio = row["Rapport A / A*"]
                regime = row["Régime d'inversion"]
                
                # Inverse A/A* -> M1
                if pd.isna(M1) and not pd.isna(A_ratio) and A_ratio >= 1.0:
                    def eq_A(M_guess):
                        # Avoid div by 0
                        if M_guess <= 0: return 9999
                        return (1/M_guess) * ((2/(y+1))*(1 + ((y-1)/2)*M_guess**2))**((y+1)/(2*(y-1))) - A_ratio
                    
                    guess = 0.5 if regime == "Subsonique" else 2.0
                    M1 = fsolve(eq_A, guess)[0]
                    df_choc.at[i, "Mach amont M1"] = M1
                    
                # Forward A/A* from M1
                elif not pd.isna(M1) and M1 > 0:
                    term = (2/(y+1))*(1 + ((y-1)/2)*M1**2)
                    df_choc.at[i, "Rapport A / A*"] = (1/M1) * term**((y+1)/(2*(y-1)))

                # Choc Droit (seulement si M1 > 1)
                if not pd.isna(M1) and M1 > 1.0:
                    num = 1 + ((y-1)/2)*M1**2
                    den = y*M1**2 - (y-1)/2
                    if den > 0:
                        df_choc.at[i, "Mach aval M2 (Choc)"] = np.sqrt(num / den)
                        df_choc.at[i, "Rapport P2/P1 (Choc)"] = 1 + (2*y/(y+1))*(M1**2 - 1)
                elif not pd.isna(M1) and M1 <= 1.0:
                    df_choc.at[i, "Mach aval M2 (Choc)"] = np.nan
                    df_choc.at[i, "Rapport P2/P1 (Choc)"] = np.nan

            st.session_state.choc_data = df_choc
            st.rerun()
