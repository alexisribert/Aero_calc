import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import fsolve

st.set_page_config(page_title="Turbo-Boosted Aero App", layout="wide", page_icon="🚀")

st.title("🚀 Turbo-Boosted Aero App")
st.markdown("Aerodynamics and fluid mechanics multi-regime resolution tool with bidirectional solver.")

# --- Conversion Dictionaries ---
unit_factors = {
    "Pressure": {"Pa": 1.0, "bar": 1e5, "atm": 101325.0},
    "Velocity": {"m/s": 1.0, "km/h": 1/3.6, "kt": 1/1.94384},
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

# --- Solver Engine: Change Detection ---
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

# --- Tabs Creation ---
tab1, tab2, tab3 = st.tabs(["📊 Experimental", "📐 Similitude & Propellers", "💨 Compressible"])

# =========================================================================
# TAB 1: EXPERIMENTAL
# =========================================================================
with tab1:
    st.header("1. Hydrostatics & Manometers")
    col1, col2 = st.columns(2)

    with col1:
        with st.expander("ℹ️ Water Manometer"):
            st.markdown("**Use case:** Measurement of small pressure differences.\n\n- **Assumptions:** Incompressible fluid, hydrostatic equilibrium.\n- **Limits:** Unsuitable for high pressures.")
            st.latex(r"\Delta P = \rho_{water} \cdot g \cdot \Delta h")
            
        col_u1_p, col_u1_h = st.columns(2)
        with col_u1_p: u_p1 = st.selectbox("Pressure Unit", ["Pa", "bar"], key="u_p1")
        with col_u1_h: u_h1 = st.selectbox("Height Unit", ["m", "cm", "mm"], key="u_h1")
        factor_h1 = {"m":1.0, "cm":0.01, "mm":0.001}[u_h1]
        
        cols = ["Rho_water (kg/m3)", "g (m/s2)", f"Delta P ({u_p1})", f"Delta h ({u_h1})"]
        init_state('eau', pd.DataFrame([[997.0, 9.81, np.nan, np.nan]], columns=cols))
        st.session_state['eau'].columns = cols

        with st.form("form_eau"):
            df_eau = st.data_editor(st.session_state['eau'], num_rows="dynamic", use_container_width=True)
            if st.form_submit_button("Calculate Water"):
                for i, row in df_eau.iterrows():
                    changed = get_changed_cols(row, st.session_state['eau_prev'].iloc[i]) if i < len(st.session_state['eau_prev']) else list(row.index)
                    target = get_target([f"Delta P ({u_p1})", f"Delta h ({u_h1})"], changed, row)
                    rho, g = row["Rho_water (kg/m3)"], row["g (m/s2)"]
                    
                    if target == f"Delta P ({u_p1})" and not pd.isna(row[f"Delta h ({u_h1})"]):
                        df_eau.at[i, target] = convert_from_SI(rho * g * (row[f"Delta h ({u_h1})"] * factor_h1), u_p1, "Pressure")
                    elif target == f"Delta h ({u_h1})" and not pd.isna(row[f"Delta P ({u_p1})"]):
                        df_eau.at[i, target] = (convert_to_SI(row[f"Delta P ({u_p1})"], u_p1, "Pressure") / (rho * g)) / factor_h1
                st.session_state['eau'] = df_eau
                st.session_state['eau_prev'] = df_eau.copy()
                st.rerun()

    with col2:
        with st.expander("ℹ️ Mercury Manometer"):
            st.markdown("**Use case:** Measurement of atmospheric pressures (Torricelli Barometer).\n\n- **Assumptions:** Incompressible, constant density.\n- **Limits:** Limited column height due to toxicity and weight.")
            st.latex(r"\Delta P = \rho_{Hg} \cdot g \cdot \Delta h")
            
        col_u2_p, col_u2_h = st.columns(2)
        with col_u2_p: u_p2 = st.selectbox("Pressure Unit", ["Pa", "bar"], key="u_p2")
        with col_u2_h: u_h2 = st.selectbox("Height Unit", ["m", "cm", "mm"], key="u_h2")
        factor_h2 = {"m":1.0, "cm":0.01, "mm":0.001}[u_h2]
        
        cols = ["Rho_Hg (kg/m3)", "g (m/s2)", f"Delta P ({u_p2})", f"Delta h ({u_h2})"]
        init_state('hg', pd.DataFrame([[13600.0, 9.81, np.nan, np.nan]], columns=cols))
        st.session_state['hg'].columns = cols

        with st.form("form_hg"):
            df_hg = st.data_editor(st.session_state['hg'], num_rows="dynamic", use_container_width=True)
            if st.form_submit_button("Calculate Mercury"):
                for i, row in df_hg.iterrows():
                    changed = get_changed_cols(row, st.session_state['hg_prev'].iloc[i]) if i < len(st.session_state['hg_prev']) else list(row.index)
                    target = get_target([f"Delta P ({u_p2})", f"Delta h ({u_h2})"], changed, row)
                    rho, g = row["Rho_Hg (kg/m3)"], row["g (m/s2)"]
                    
                    if target == f"Delta P ({u_p2})" and not pd.isna(row[f"Delta h ({u_h2})"]):
                        df_hg.at[i, target] = convert_from_SI(rho * g * (row[f"Delta h ({u_h2})"] * factor_h2), u_p2, "Pressure")
                    elif target == f"Delta h ({u_h2})" and not pd.isna(row[f"Delta P ({u_p2})"]):
                        df_hg.at[i, target] = (convert_to_SI(row[f"Delta P ({u_p2})"], u_p2, "Pressure") / (rho * g)) / factor_h2
                st.session_state['hg'] = df_hg
                st.session_state['hg_prev'] = df_hg.copy()
                st.rerun()

    st.divider()
    st.header("2. Dynamics & Velocity")
    col3, col4 = st.columns(2)

    with col3:
        with st.expander("ℹ️ Torricelli's Law"):
            st.markdown("**Use case:** Flow velocity from an open tank.\n\n- **Assumptions:** Perfect fluid, constant head.\n- **Limits:** Neglects head losses and vena contracta.")
            st.latex(r"U = \sqrt{2 \cdot g \cdot h}")
            
        col_u3_v, col_u3_h = st.columns(2)
        with col_u3_v: u_v3 = st.selectbox("Velocity Unit", ["m/s", "km/h"], key="u_v3")
        with col_u3_h: u_h3 = st.selectbox("Height Unit", ["m", "ft"], key="u_h3")
        
        cols = ["g (m/s2)", f"Velocity U ({u_v3})", f"Height h ({u_h3})"]
        init_state('torri', pd.DataFrame([[9.81, np.nan, np.nan]], columns=cols))
        st.session_state['torri'].columns = cols

        with st.form("form_torri"):
            df_torri = st.data_editor(st.session_state['torri'], num_rows="dynamic", use_container_width=True)
            if st.form_submit_button("Calculate Torricelli"):
                for i, row in df_torri.iterrows():
                    changed = get_changed_cols(row, st.session_state['torri_prev'].iloc[i]) if i < len(st.session_state['torri_prev']) else list(row.index)
                    target = get_target([f"Velocity U ({u_v3})", f"Height h ({u_h3})"], changed, row)
                    g = row["g (m/s2)"]
                    
                    if target == f"Velocity U ({u_v3})" and not pd.isna(row[f"Height h ({u_h3})"]):
                        h = convert_to_SI(row[f"Height h ({u_h3})"], u_h3, "Distance")
                        if h >= 0: df_torri.at[i, target] = convert_from_SI(np.sqrt(2 * g * h), u_v3, "Velocity")
                    elif target == f"Height h ({u_h3})" and not pd.isna(row[f"Velocity U ({u_v3})"]):
                        u = convert_to_SI(row[f"Velocity U ({u_v3})"], u_v3, "Velocity")
                        df_torri.at[i, target] = convert_from_SI((u**2) / (2 * g), u_h3, "Distance")
                st.session_state['torri'] = df_torri
                st.session_state['torri_prev'] = df_torri.copy()
                st.rerun()

    with col4:
        with st.expander("ℹ️ Pitot Tube (Incompressible)"):
            st.markdown("**Use case:** Velocity calculation via dynamic pressure.\n\n- **Assumptions:** Mach < 0.3, isentropic.\n- **Limits:** Invalid for compressible fluids (use Saint-Venant equations).")
            st.latex(r"U = \sqrt{\frac{2 \cdot \Delta P}{\rho}}")
            
        col_u4_v, col_u4_p = st.columns(2)
        with col_u4_v: u_v4 = st.selectbox("Velocity Unit", ["m/s", "km/h", "kt"], key="u_v4")
        with col_u4_p: u_p4 = st.selectbox("Pressure Unit", ["Pa", "bar"], key="u_p4")
        
        cols = ["Rho_air (kg/m3)", f"Delta P ({u_p4})", f"Velocity U ({u_v4})"]
        init_state('pitot', pd.DataFrame([[1.225, np.nan, np.nan]], columns=cols))
        st.session_state['pitot'].columns = cols

        with st.form("form_pitot"):
            df_pitot = st.data_editor(st.session_state['pitot'], num_rows="dynamic", use_container_width=True)
            if st.form_submit_button("Calculate Pitot"):
                for i, row in df_pitot.iterrows():
                    changed = get_changed_cols(row, st.session_state['pitot_prev'].iloc[i]) if i < len(st.session_state['pitot_prev']) else list(row.index)
                    target = get_target([f"Velocity U ({u_v4})", f"Delta P ({u_p4})"], changed, row)
                    rho = row["Rho_air (kg/m3)"]
                    
                    if target == f"Velocity U ({u_v4})" and not pd.isna(row[f"Delta P ({u_p4})"]):
                        p = convert_to_SI(row[f"Delta P ({u_p4})"], u_p4, "Pressure")
                        if p >= 0: df_pitot.at[i, target] = convert_from_SI(np.sqrt(2 * p / rho), u_v4, "Velocity")
                    elif target == f"Delta P ({u_p4})" and not pd.isna(row[f"Velocity U ({u_v4})"]):
                        u = convert_to_SI(row[f"Velocity U ({u_v4})"], u_v4, "Velocity")
                        df_pitot.at[i, target] = convert_from_SI(0.5 * rho * (u**2), u_p4, "Pressure")
                st.session_state['pitot'] = df_pitot
                st.session_state['pitot_prev'] = df_pitot.copy()
                st.rerun()

    st.divider()
    st.header("3. Thermodynamics & Atmosphere")
    
    with st.expander("ℹ️ Ideal Gas Law"):
        st.markdown("**Use case:** Equation of state.\n\n- **Assumptions:** Point particles, elastic collisions.\n- **Limits:** Inaccurate under very high pressures.")
        st.latex(r"P = \rho \cdot r \cdot T")

    col_u5_p, col_u5_t = st.columns(2)
    with col_u5_p: u_p5 = st.selectbox("Pressure Unit", ["Pa", "bar", "atm"], key="u_p5_2")
    with col_u5_t: u_t5 = st.selectbox("Temperature Unit", ["K", "°C"], key="u_t5_2")

    cols = ["Constant r", f"Pressure ({u_p5})", "Rho (kg/m3)", f"Temperature ({u_t5})"]
    init_state('gaz', pd.DataFrame([[287.05, np.nan, np.nan, np.nan]], columns=cols))
    st.session_state['gaz'].columns = cols

    with st.form("form_gaz"):
        df_gaz = st.data_editor(st.session_state['gaz'], num_rows="dynamic", use_container_width=True)
        if st.form_submit_button("Calculate Ideal Gas"):
            for i, row in df_gaz.iterrows():
                changed = get_changed_cols(row, st.session_state['gaz_prev'].iloc[i]) if i < len(st.session_state['gaz_prev']) else list(row.index)
                target = get_target([f"Pressure ({u_p5})", "Rho (kg/m3)", f"Temperature ({u_t5})"], changed, row)
                
                r = row["Constant r"]
                p = convert_to_SI(row[f"Pressure ({u_p5})"], u_p5, "Pressure")
                t = convert_to_SI(row[f"Temperature ({u_t5})"], u_t5, "Temperature")
                rho = row["Rho (kg/m3)"]
                
                if target == f"Pressure ({u_p5})" and not pd.isna(rho) and not pd.isna(t):
                    df_gaz.at[i, target] = convert_from_SI(rho * r * t, u_p5, "Pressure")
                elif target == "Rho (kg/m3)" and not pd.isna(p) and not pd.isna(t) and t > 0:
                    df_gaz.at[i, target] = p / (r * t)
                elif target == f"Temperature ({u_t5})" and not pd.isna(p) and not pd.isna(rho) and rho > 0:
                    df_gaz.at[i, target] = convert_from_SI(p / (rho * r), u_t5, "Temperature")
            st.session_state['gaz'] = df_gaz
            st.session_state['gaz_prev'] = df_gaz.copy()
            st.rerun()

    col5, col6 = st.columns(2)

    with col5:
        with st.expander("ℹ️ Sutherland's Law"):
            st.markdown("**Use case:** Dynamic viscosity of air.\n\n- **Assumptions:** Empirical kinetic model.\n- **Limits:** Valid only between ~100K and 1900K.")
            st.latex(r"\mu \approx \frac{B \sqrt{T}}{1 + A/T}")
            
        u_t6 = st.selectbox("Temperature Unit", ["K", "°C"], key="u_t6")
        cols = ["Const. A", "Const. B", f"Temp T ({u_t6})", "Mu (Pa.s)"]
        init_state('suth', pd.DataFrame([[114.0, 1.458e-6, np.nan, np.nan]], columns=cols))
        st.session_state['suth'].columns = cols

        with st.form("form_suth"):
            df_suth = st.data_editor(st.session_state['suth'], num_rows="dynamic", use_container_width=True)
            if st.form_submit_button("Calculate Sutherland"):
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
        with st.expander("ℹ️ Standard Atmosphere (ISA)"):
            st.markdown("**Use case:** Standard Pressure and Temperature model.\n\n- **Assumptions:** Calm atmosphere, dry air, ideal gas.\n- **Limits:** Does not account for local weather or humidity.")
            st.latex(r"Z < 11 km : P = P_0 \left( 1 - \frac{s Z}{100 T_0} \right)^{\frac{100 g}{r s}} \quad|\quad Z > 11 : P = P_{11} \exp\left(-\frac{g(Z - 11000)}{r T_{11}}\right)")
            
        col_u7_z, col_u7_p = st.columns(2)
        with col_u7_z: u_z7 = st.selectbox("Altitude Unit", ["m", "ft", "km"], key="u_z7")
        with col_u7_p: u_p7 = st.selectbox("Pressure Unit", ["Pa", "bar", "atm"], key="u_p7_isa")
        
        cols = ["P0 (Pa)", "T0 (K)", "s (K/100m)", f"Alt Z ({u_z7})", f"Pressure ({u_p7})"]
        init_state('isa', pd.DataFrame([[101325.0, 288.15, 0.65, np.nan, np.nan]], columns=cols))
        st.session_state['isa'].columns = cols

        with st.form("form_isa"):
            df_isa = st.data_editor(st.session_state['isa'], num_rows="dynamic", use_container_width=True)
            if st.form_submit_button("Calculate ISA"):
                for i, row in df_isa.iterrows():
                    changed = get_changed_cols(row, st.session_state['isa_prev'].iloc[i]) if i < len(st.session_state['isa_prev']) else list(row.index)
                    target = get_target([f"Alt Z ({u_z7})", f"Pressure ({u_p7})"], changed, row)
                    
                    p0, t0, s = row["P0 (Pa)"], row["T0 (K)"], row["s (K/100m)"]
                    z = convert_to_SI(row[f"Alt Z ({u_z7})"], u_z7, "Distance")
                    p = convert_to_SI(row[f"Pressure ({u_p7})"], u_p7, "Pressure")
                    g, r = 9.81, 287.05
                    p11 = p0 * (1 - (s*11000)/(100*t0))**((100*g)/(r*s))
                    t11 = t0 - (s*11000/100)
                    
                    if target == f"Pressure ({u_p7})" and not pd.isna(z):
                        res = p0 * (1 - (s*z)/(100*t0))**((100*g)/(r*s)) if z <= 11000 else p11 * np.exp(-g*(z-11000)/(r*t11))
                        df_isa.at[i, target] = convert_from_SI(res, u_p7, "Pressure")
                    elif target == f"Alt Z ({u_z7})" and not pd.isna(p):
                        res = (100*t0/s) * (1 - (p/p0)**((r*s)/(100*g))) if p >= p11 else 11000 - (np.log(p/p11) * r * t11 / g)
                        df_isa.at[i, target] = convert_from_SI(res, u_z7, "Distance")
                st.session_state['isa'] = df_isa
                st.session_state['isa_prev'] = df_isa.copy()
                st.rerun()

# =========================================================================
# TAB 2: SIMILITUDE & PROPELLERS
# =========================================================================
with tab2:
    st.header("1. Dimensionless Numbers")
    with st.expander("ℹ️ Reynolds, Froude, Mach, Strouhal"):
        st.markdown("**Use case:** Similarity between scale model and full-size model.\n\n- **Assumptions:** Complete dimensional analogy required.\n- **Limits:** Impossible to match Re and Mach simultaneously in a subsonic wind tunnel.")
        st.latex(r"Re = \frac{\rho L U}{\mu} \quad|\quad Fr = \frac{U}{\sqrt{g L}} \quad|\quad M = \frac{U}{\sqrt{\gamma r T}} \quad|\quad St = \frac{f L}{U}")
    
    cols = ["Rho", "Mu", "Gamma", "r", "T(K)", "g", "freq(Hz)", "L(m)", "U(m/s)", "Reynolds", "Froude", "Mach", "Strouhal"]
    init_state('adim', pd.DataFrame([[1.225, 1.81e-5, 1.4, 287.05, 288.15, 9.81, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]], columns=cols))
    
    with st.form("form_adim"):
        df_adim = st.data_editor(st.session_state['adim'], num_rows="dynamic", use_container_width=True)
        if st.form_submit_button("Calculate Similarity"):
            for i, row in df_adim.iterrows():
                changed = get_changed_cols(row, st.session_state['adim_prev'].iloc[i]) if i < len(st.session_state['adim_prev']) else list(row.index)
                rho, mu, y, r, T, g, freq = row["Rho"], row["Mu"], row["Gamma"], row["r"], row["T(K)"], row["g"], row["freq(Hz)"]
                L, U = row["L(m)"], row["U(m/s)"]
                
                # Reverse for U
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
    st.header("2. Aerodynamic Coeffs. & Propellers")
    with st.expander("ℹ️ Forces and Propulsive Efficiency"):
        st.markdown("**Use case:** Calculation of global aerodynamic loads and propeller efficiency.\n\n- **Assumptions:** Coefficients (Ct, Cp) are valid for a fixed pitch.\n- **Limits:** Efficiency drops in the transonic regime (shocks at blade tips).")
        st.latex(r"C_F = \frac{F}{\frac{1}{2}\rho U^2 S} \quad|\quad J = \frac{V}{n \cdot D} \quad|\quad \eta = \frac{C_t \cdot J}{C_{p\_prop}}")

    cols = ["Rho", "S(m2)", "D(m)", "V(m/s)", "n(rev/s)", "Force F(N)", "C_F", "Thrust(N)", "Ct", "Power(W)", "Cp_prop", "Advance J", "Efficiency"]
    init_state('helice', pd.DataFrame([[1.225, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]], columns=cols))
    
    with st.form("form_helice"):
        df_helice = st.data_editor(st.session_state['helice'], num_rows="dynamic", use_container_width=True)
        if st.form_submit_button("Calculate Coeffs & Propellers"):
            for i, row in df_helice.iterrows():
                changed = get_changed_cols(row, st.session_state['helice_prev'].iloc[i]) if i < len(st.session_state['helice_prev']) else list(row.index)
                rho, S, D, V, n = row["Rho"], row["S(m2)"], row["D(m)"], row["V(m/s)"], row["n(rev/s)"]
                
                # CF and Force
                target_F = get_target(["Force F(N)", "C_F"], changed, row)
                if target_F == "C_F" and not pd.isna(row["Force F(N)"]) and not pd.isna(V) and not pd.isna(S):
                    df_helice.at[i, "C_F"] = row["Force F(N)"] / (0.5 * rho * V**2 * S)
                elif target_F == "Force F(N)" and not pd.isna(row["C_F"]) and not pd.isna(V) and not pd.isna(S):
                    df_helice.at[i, "Force F(N)"] = row["C_F"] * 0.5 * rho * V**2 * S
                
                # Propeller : J, Ct, Cp
                if not pd.isna(n) and not pd.isna(D) and n > 0 and D > 0:
                    if not pd.isna(V) and "Advance J" not in changed: df_helice.at[i, "Advance J"] = V / (n * D)
                    
                    target_T = get_target(["Thrust(N)", "Ct"], changed, row)
                    if target_T == "Ct" and not pd.isna(row["Thrust(N)"]): df_helice.at[i, "Ct"] = row["Thrust(N)"] / (rho * n**2 * D**4)
                    elif target_T == "Thrust(N)" and not pd.isna(row["Ct"]): df_helice.at[i, "Thrust(N)"] = row["Ct"] * rho * n**2 * D**4
                    
                    target_P = get_target(["Power(W)", "Cp_prop"], changed, row)
                    if target_P == "Cp_prop" and not pd.isna(row["Power(W)"]): df_helice.at[i, "Cp_prop"] = row["Power(W)"] / (rho * n**3 * D**5)
                    elif target_P == "Power(W)" and not pd.isna(row["Cp_prop"]): df_helice.at[i, "Power(W)"] = row["Cp_prop"] * rho * n**3 * D**5
                
                # Efficiency
                if "Efficiency" not in changed:
                    J, Ct, Cp_h = df_helice.at[i, "Advance J"], df_helice.at[i, "Ct"], df_helice.at[i, "Cp_prop"]
                    if not pd.isna(J) and not pd.isna(Ct) and not pd.isna(Cp_h) and Cp_h != 0:
                        df_helice.at[i, "Efficiency"] = (Ct * J) / Cp_h

            st.session_state['helice'] = df_helice
            st.session_state['helice_prev'] = df_helice.copy()
            st.rerun()

# =========================================================================
# TAB 3: COMPRESSIBLE
# =========================================================================
with tab3:
    st.header("1. Laplace's Law (Isentropic Transformations)")
    with st.expander("ℹ️ Framework, Assumptions and Limits"):
        st.markdown("**Use case:** Thermodynamic evolution of an ideal gas during compression or expansion (without heat transfer).\n\n- **Assumptions:** Adiabatic and reversible flow (isentropic), constant heat capacity (Gamma).\n- **Limits:** Does not account for shock waves, viscous friction, or heat transfers.")
        st.latex(r"P_1 V_1^\gamma = P_2 V_2^\gamma \quad|\quad \frac{T_2}{T_1} = \left(\frac{P_2}{P_1}\right)^{\frac{\gamma-1}{\gamma}} \quad|\quad \frac{\rho_2}{\rho_1} = \left(\frac{P_2}{P_1}\right)^{\frac{1}{\gamma}}")
    
    cols = ["Gamma", "P2 / P1", "T2 / T1", "Rho2 / Rho1"]
    init_state('laplace', pd.DataFrame([[1.4, np.nan, np.nan, np.nan]], columns=cols))
    
    with st.form("form_laplace"):
        df_laplace = st.data_editor(st.session_state['laplace'], num_rows="dynamic", use_container_width=True)
        if st.form_submit_button("Calculate Laplace"):
            for i, row in df_laplace.iterrows():
                changed = get_changed_cols(row, st.session_state['laplace_prev'].iloc[i]) if i < len(st.session_state['laplace_prev']) else list(row.index)
                y = row["Gamma"]
                
                if "P2 / P1" in changed and not pd.isna(row["P2 / P1"]):
                    if "T2 / T1" not in changed: df_laplace.at[i, "T2 / T1"] = row["P2 / P1"]**((y-1)/y)
                    if "Rho2 / Rho1" not in changed: df_laplace.at[i, "Rho2 / Rho1"] = row["P2 / P1"]**(1/y)
                elif "T2 / T1" in changed and not pd.isna(row["T2 / T1"]):
                    if "P2 / P1" not in changed: df_laplace.at[i, "P2 / P1"] = row["T2 / T1"]**(y/(y-1))
                    if "Rho2 / Rho1" not in changed: df_laplace.at[i, "Rho2 / Rho1"] = row["T2 / T1"]**(1/(y-1))
                elif "Rho2 / Rho1" in changed and not pd.isna(row["Rho2 / Rho1"]):
                    if "P2 / P1" not in changed: df_laplace.at[i, "P2 / P1"] = row["Rho2 / Rho1"]**y
                    if "T2 / T1" not in changed: df_laplace.at[i, "T2 / T1"] = row["Rho2 / Rho1"]**(y-1)
                    
            st.session_state['laplace'] = df_laplace
            st.session_state['laplace_prev'] = df_laplace.copy()
            st.rerun()

    st.divider()
    st.header("2. Stagnation & Critical Properties")
    with st.expander("ℹ️ Definitions and Isentropic Ratios"):
        st.markdown("""
        - **Stagnation property ($X_i$):** Reference state of a fluid if it were brought to rest isentropically (**zero velocity**, $M = 0$).
        - **Critical property ($X^*$):** Reference state of a fluid if it were accelerated or decelerated isentropically to exactly **the speed of sound** ($M = 1$).
        
        - **Assumptions:** Adiabatic, reversible flow (no shocks or friction).
        - **Limits:** Stagnation properties are not conserved across a shock wave (entropy loss).
        """)
        st.latex(r"\frac{T_i}{T} = 1 + \frac{\gamma-1}{2}M^2 \quad|\quad \frac{P_i}{P} = \left(\frac{T_i}{T}\right)^{\frac{\gamma}{\gamma-1}} \quad|\quad \frac{\rho_i}{\rho} = \left(\frac{T_i}{T}\right)^{\frac{1}{\gamma-1}}")
        st.latex(r"\text{At Mach 1} : \frac{T^*}{T_i} = \frac{2}{\gamma+1} \quad|\quad \frac{P^*}{P_i} = \left(\frac{2}{\gamma+1}\right)^{\frac{\gamma}{\gamma-1}} \quad|\quad \frac{\rho^*}{\rho_i} = \left(\frac{2}{\gamma+1}\right)^{\frac{1}{\gamma-1}}")
    
    cols = ["Gamma", "Mach M", "T_i / T", "P_i / P", "Rho_i / Rho", "T* / T_i", "P* / P_i", "Rho* / Rho_i"]
    init_state('isen', pd.DataFrame([[1.4, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]], columns=cols))
        
    with st.form("form_isen"):
        df_isen = st.data_editor(st.session_state['isen'], num_rows="dynamic", use_container_width=True)
        if st.form_submit_button("Calculate Isentropic Ratios"):
            for i, row in df_isen.iterrows():
                changed = get_changed_cols(row, st.session_state['isen_prev'].iloc[i]) if i < len(st.session_state['isen_prev']) else list(row.index)
                y, M = row["Gamma"], row["Mach M"]
                
                # Inversion Mach
                if "T_i / T" in changed and not pd.isna(row["T_i / T"]): M = np.sqrt(max(0, (row["T_i / T"] - 1) * 2 / (y - 1)))
                elif "P_i / P" in changed and not pd.isna(row["P_i / P"]): M = np.sqrt(max(0, (row["P_i / P"]**((y-1)/y) - 1) * 2 / (y - 1)))
                elif "Rho_i / Rho" in changed and not pd.isna(row["Rho_i / Rho"]): M = np.sqrt(max(0, (row["Rho_i / Rho"]**(y-1) - 1) * 2 / (y - 1)))
                df_isen.at[i, "Mach M"] = M
                
                # Direct calculation
                if not pd.isna(M):
                    if "T_i / T" not in changed: df_isen.at[i, "T_i / T"] = 1 + ((y-1)/2) * M**2
                    if "P_i / P" not in changed: df_isen.at[i, "P_i / P"] = (1 + ((y-1)/2) * M**2)**(y/(y-1))
                    if "Rho_i / Rho" not in changed: df_isen.at[i, "Rho_i / Rho"] = (1 + ((y-1)/2) * M**2)**(1/(y-1))
                
                # Critical values fixed by Gamma
                df_isen.at[i, "T* / T_i"] = 2 / (y+1)
                df_isen.at[i, "P* / P_i"] = (2 / (y+1))**(y/(y-1))
                df_isen.at[i, "Rho* / Rho_i"] = (2 / (y+1))**(1/(y-1))
                
            st.session_state['isen'] = df_isen
            st.session_state['isen_prev'] = df_isen.copy()
            st.rerun()

    st.divider()
    st.header("3. Nozzles, Sonic Throat and Mass Flow")
    with st.expander("ℹ️ 1D Flow (A/A*) and Mass Flow"):
        st.markdown("**Use case:** Sizing of converging-diverging de Laval nozzles and mass flow calculation.\n\n- **Assumptions:** Quasi-steady 1D flow.\n- **Limits:** Choked flow phenomenon: mass flow cannot increase further if the throat is at $M=1$.")
        st.latex(r"\frac{A}{A^*} = \frac{1}{M} \left[ \frac{2}{\gamma+1} \left( 1 + \frac{\gamma-1}{2}M^2 \right) \right]^{\frac{\gamma+1}{2(\gamma-1)}}")
        st.latex(r"\dot{m} = \frac{A \cdot P_i}{\sqrt{r \cdot T_i}} \sqrt{\gamma} \cdot M \left(1 + \frac{\gamma-1}{2}M^2\right)^{-\frac{\gamma+1}{2(\gamma-1)}}")

    cols = ["Gamma", "r (J/kgK)", "P_i (Pa)", "T_i (K)", "Area A(m2)", "Mach M", "Ratio A / A*", "Mass Flow (kg/s)", "Inverse regime"]
    init_state('tuyere', pd.DataFrame([[1.4, 287.05, 101325.0, 288.15, np.nan, np.nan, np.nan, np.nan, "Subsonic"]], columns=cols))
        
    with st.form("form_tuyere"):
        df_tuyere = st.data_editor(st.session_state['tuyere'], column_config={"Inverse regime": st.column_config.SelectboxColumn(options=["Subsonic", "Supersonic"])}, num_rows="dynamic", use_container_width=True)
        if st.form_submit_button("Calculate Nozzles"):
            for i, row in df_tuyere.iterrows():
                changed = get_changed_cols(row, st.session_state['tuyere_prev'].iloc[i]) if i < len(st.session_state['tuyere_prev']) else list(row.index)
                y, r, Pi, Ti, A = row["Gamma"], row["r (J/kgK)"], row["P_i (Pa)"], row["T_i (K)"], row["Area A(m2)"]
                M, A_ratio, debit, regime = row["Mach M"], row["Ratio A / A*"], row["Mass Flow (kg/s)"], row["Inverse regime"]
                
                # Complex inversion via numerical solver
                if "Ratio A / A*" in changed and not pd.isna(A_ratio) and A_ratio >= 1.0:
                    def eq_A(M_guess): return (1/M_guess) * ((2/(y+1))*(1 + ((y-1)/2)*M_guess**2))**((y+1)/(2*(y-1))) - A_ratio
                    M = fsolve(eq_A, 0.5 if regime == "Subsonic" else 2.0)[0]
                    df_tuyere.at[i, "Mach M"] = M
                elif "Mass Flow (kg/s)" in changed and not pd.isna(debit) and not pd.isna(A) and not pd.isna(Pi) and not pd.isna(Ti):
                    def eq_m(M_guess): return ((A * Pi) / np.sqrt(r * Ti)) * np.sqrt(y) * M_guess * (1 + ((y-1)/2)*M_guess**2)**(-(y+1)/(2*(y-1))) - debit
                    M = fsolve(eq_m, 0.5 if regime == "Subsonic" else 2.0)[0]
                    df_tuyere.at[i, "Mach M"] = M
                    
                # Direct calculation
                if not pd.isna(M) and M > 0:
                    if "Ratio A / A*" not in changed:
                        df_tuyere.at[i, "Ratio A / A*"] = (1/M) * ((2/(y+1))*(1 + ((y-1)/2)*M**2))**((y+1)/(2*(y-1)))
                    if "Mass Flow (kg/s)" not in changed and not pd.isna(A) and not pd.isna(Pi) and not pd.isna(Ti):
                        df_tuyere.at[i, "Mass Flow (kg/s)"] = (A * Pi / np.sqrt(r * Ti)) * np.sqrt(y) * M * (1 + ((y-1)/2)*M**2)**(-(y+1)/(2*(y-1)))

            st.session_state['tuyere'] = df_tuyere
            st.session_state['tuyere_prev'] = df_tuyere.copy()
            st.rerun()

    st.divider()
    st.header("4. Normal Shock Waves")
    with st.expander("ℹ️ Rankine-Hugoniot Relations (Normal Shocks)"):
        st.markdown("**Use case:** Thermodynamic jumps and losses across a normal shock wave.\n\n- **Assumptions:** Infinitely thin shock, ideal gas.\n- **Limits:** Not applicable to oblique shocks. The upstream Mach number ($M_1$) must be strictly > 1.")
        st.latex(r"M_2 = \sqrt{\frac{1 + \frac{\gamma-1}{2}M_1^2}{\gamma M_1^2 - \frac{\gamma-1}{2}}} \quad|\quad \frac{P_2}{P_1} = 1 + \frac{2\gamma}{\gamma+1}(M_1^2 - 1)")
        st.latex(r"\frac{T_2}{T_1} = \frac{1 + \frac{\gamma-1}{2}M_1^2}{1 + \frac{\gamma-1}{2}M_2^2} \quad|\quad \frac{\rho_2}{\rho_1} = \frac{(\gamma+1)M_1^2}{2 + (\gamma-1)M_1^2}")

    cols = ["Gamma", "Upstream Mach M1", "Downstream Mach M2", "P2 / P1", "T2 / T1", "Rho2 / Rho1"]
    init_state('choc', pd.DataFrame([[1.4, np.nan, np.nan, np.nan, np.nan, np.nan]], columns=cols))
        
    with st.form("form_choc"):
        df_choc = st.data_editor(st.session_state['choc'], num_rows="dynamic", use_container_width=True)
        if st.form_submit_button("Calculate Normal Shocks"):
            for i, row in df_choc.iterrows():
                changed = get_changed_cols(row, st.session_state['choc_prev'].iloc[i]) if i < len(st.session_state['choc_prev']) else list(row.index)
                y = row["Gamma"]
                M1 = row["Upstream Mach M1"]
                
                # Inversion M1
                if "Downstream Mach M2" in changed and not pd.isna(row["Downstream Mach M2"]):
                    def eq_M2(M1_g): return np.sqrt((1 + ((y-1)/2)*M1_g**2) / (y*M1_g**2 - (y-1)/2)) - row["Downstream Mach M2"]
                    M1 = fsolve(eq_M2, 2.0)[0]
                elif "P2 / P1" in changed and not pd.isna(row["P2 / P1"]):
                    M1 = np.sqrt(((y+1)/(2*y)) * (row["P2 / P1"] - 1) + 1)
                elif "T2 / T1" in changed and not pd.isna(row["T2 / T1"]):
                    def eq_T2(M1_g): 
                        m2_sq = (1 + ((y-1)/2)*M1_g**2) / (y*M1_g**2 - (y-1)/2)
                        return ((1 + ((y-1)/2)*M1_g**2) / (1 + ((y-1)/2)*m2_sq)) - row["T2 / T1"]
                    M1 = fsolve(eq_T2, 2.0)[0]
                elif "Rho2 / Rho1" in changed and not pd.isna(row["Rho2 / Rho1"]):
                    def eq_Rho(M1_g): return (((y+1)*M1_g**2) / (2 + (y-1)*M1_g**2)) - row["Rho2 / Rho1"]
                    M1 = fsolve(eq_Rho, 2.0)[0]
                    
                df_choc.at[i, "Upstream Mach M1"] = M1
                
                # Direct calculation of jumps
                if not pd.isna(M1) and M1 > 1.0:
                    M2_sq = (1 + ((y-1)/2)*M1**2) / (y*M1**2 - (y-1)/2)
                    if "Downstream Mach M2" not in changed: df_choc.at[i, "Downstream Mach M2"] = np.sqrt(M2_sq)
                    if "P2 / P1" not in changed: df_choc.at[i, "P2 / P1"] = 1 + (2*y/(y+1))*(M1**2 - 1)
                    if "T2 / T1" not in changed: df_choc.at[i, "T2 / T1"] = (1 + ((y-1)/2)*M1**2) / (1 + ((y-1)/2)*M2_sq)
                    if "Rho2 / Rho1" not in changed: df_choc.at[i, "Rho2 / Rho1"] = ((y+1)*M1**2) / (2 + (y-1)*M1**2)
                elif not pd.isna(M1) and M1 <= 1.0:
                    df_choc.at[i, "Downstream Mach M2"] = np.nan
                    df_choc.at[i, "P2 / P1"] = np.nan
                    df_choc.at[i, "T2 / T1"] = np.nan
                    df_choc.at[i, "Rho2 / Rho1"] = np.nan

            st.session_state['choc'] = df_choc
            st.session_state['choc_prev'] = df_choc.copy()
            st.rerun()
