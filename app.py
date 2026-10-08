import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import root_scalar

# Configuration de la page
st.set_page_config(page_title="AER Turbo-Boosté", layout="wide", page_icon="🚀")

# ==========================================
# GESTION DES UNITÉS (Dictionnaires de conversion vers le SI)
# ==========================================
UNITS = {
    "Vitesse": {"m/s": 1.0, "km/h": 1/3.6, "kts": 0.51444},
    "Pression": {"Pa": 1.0, "hPa": 100.0, "bar": 1e5, "psi": 6894.76},
    "Température": {"K": "K", "°C": "C"},
    "Longueur": {"m": 1.0, "cm": 0.01, "ft": 0.3048},
    "Densité": {"kg/m³": 1.0, "g/cm³": 1000.0},
    "Force": {"N": 1.0, "kN": 1000.0}
}

def convert_temp_to_si(val, unit):
    if pd.isna(val): return np.nan
    return val + 273.15 if unit == "°C" else val

def convert_temp_from_si(val, unit):
    if pd.isna(val): return np.nan
    return val - 273.15 if unit == "°C" else val

# ==========================================
# FONCTIONS DE MISE EN PAGE DU FORMALISME
# ==========================================
def display_theory(nom, usage, latex_form, hypotheses, limites):
    with st.expander(f"📚 Théorie & Hypothèses : {nom}"):
        st.markdown(f"**Cas d'usage :** {usage}")
        st.latex(latex_form)
        st.markdown(f"**Hypothèses :** {hypotheses}")
        st.markdown(f"**Limitations :** {limites}")

# ==========================================
# BARRE DE NAVIGATION LATÉRALE
# ==========================================
st.sidebar.title("🚀 AER Turbo-Boosté")
page = st.sidebar.radio("Thématiques", ["1. Expérimental", "2. Similitude", "3. Compressible"])

st.sidebar.markdown("---")
st.sidebar.info("💡 **Astuce Réversibilité :** Laissez vide (ou effacez) la valeur que vous cherchez dans un tableau, le programme la calculera automatiquement à partir des autres !")

# ==========================================
# PAGE 1 : EXPÉRIMENTAL
# ==========================================
if page == "1. Expérimental":
    st.title("🔬 Aérodynamique Expérimentale")

    # --- BLOC 1 : TUBE DE PITOT (Incompressible) ---
    st.header("1. Mesure de Vitesse (Tube de Pitot)")
    display_theory(
        nom="Vitesse Incompressible (Bernoulli / Torricelli)",
        usage="Détermination de la vitesse, de la pression dynamique ou de la masse volumique.",
        latex_form=r"V = \sqrt{\frac{2 \cdot \Delta P}{\rho}} \quad \iff \quad \Delta P = \frac{1}{2} \rho V^2",
        hypotheses="Fluide incompressible ($\\rho = cte$), écoulement stationnaire, isentropique, fluide parfait.",
        limites="Non valide pour les hautes vitesses (Mach > 0.3) où les effets de compressibilité apparaissent."
    )
    
    col1, col2 = st.columns(2)
    unit_p = col1.selectbox("Unité de Pression ($\Delta P$)", list(UNITS["Pression"].keys()), key="pitot_p")
    unit_v = col2.selectbox("Unité de Vitesse ($V$)", list(UNITS["Vitesse"].keys()), key="pitot_v")

    df_pitot = pd.DataFrame([
        {"Delta P": 1500.0, "Rho (kg/m³)": 1.225, "Vitesse": None},
        {"Delta P": None, "Rho (kg/m³)": 1.225, "Vitesse": 50.0}
    ])

    edited_pitot = st.data_editor(
        df_pitot, num_rows="dynamic", use_container_width=True,
        column_config={
            "Delta P": f"ΔP Pression dyn. ({unit_p})",
            "Vitesse": f"Vitesse calculée ({unit_v})"
        }
    )
    
    if not edited_pitot.empty:
        def solve_pitot(row):
            dp_raw, rho, v_raw = row["Delta P"], row["Rho (kg/m³)"], row["Vitesse"]
            dp = dp_raw * UNITS["Pression"][unit_p] if pd.notna(dp_raw) else np.nan
            v = v_raw * UNITS["Vitesse"][unit_v] if pd.notna(v_raw) else np.nan

            if pd.isna(v) and pd.notna(dp) and pd.notna(rho) and rho > 0:
                v = np.sqrt(max(0, 2 * dp / rho))
            elif pd.isna(dp) and pd.notna(v) and pd.notna(rho):
                dp = 0.5 * rho * v**2
            elif pd.isna(rho) and pd.notna(dp) and pd.notna(v) and v > 0:
                rho = 2 * dp / v**2

            row["Delta P"] = dp / UNITS["Pression"][unit_p] if pd.notna(dp) else np.nan
            row["Rho (kg/m³)"] = rho
            row["Vitesse"] = v / UNITS["Vitesse"][unit_v] if pd.notna(v) else np.nan
            return row
            
        st.success("Résultats :")
        st.dataframe(edited_pitot.apply(solve_pitot, axis=1), use_container_width=True)

    st.markdown("---")

    # --- BLOC 2 : ATMOSPHÈRE STANDARD (ISA) ---
    st.header("2. Atmosphère Standard (Troposphère)")
    display_theory(
        nom="Modèle ISA (International Standard Atmosphere)",
        usage="Prédiction de la pression, température ou de l'altitude géopotentielle.",
        latex_form=r"T = T_0 - L \cdot h \quad \text{et} \quad P = P_0 \left(1 - \frac{L \cdot h}{T_0}\right)^{\frac{g}{R \cdot L}}",
        hypotheses="Gradient de température constant ($L = 0.0065$ K/m), gaz parfait, équilibre hydrostatique.",
        limites="Uniquement valable dans la Troposphère (Altitude $h < 11 000$ m). Ne reflète pas la météo réelle."
    )

    col1, col2, col3 = st.columns(3)
    unit_h = col1.selectbox("Unité d'Altitude", list(UNITS["Longueur"].keys()), key="isa_h")
    unit_t_isa = col2.selectbox("Unité Température", ["K", "°C"], key="isa_t")
    unit_p_isa = col3.selectbox("Unité Pression", list(UNITS["Pression"].keys()), key="isa_p")

    df_isa = pd.DataFrame([
        {"h": 5000.0, "T0": 15.0, "P0": 101325.0, "T": None, "P": None},
        {"h": None, "T0": 15.0, "P0": 101325.0, "T": None, "P": 50000.0}
    ])
    
    edited_isa = st.data_editor(
        df_isa, num_rows="dynamic", use_container_width=True,
        column_config={
            "h": f"Altitude ({unit_h})", "T0": f"T0 ({unit_t_isa})", "P0": f"P0 ({unit_p_isa})",
            "T": f"Température ({unit_t_isa})", "P": f"Pression ({unit_p_isa})"
        }
    )

    if not edited_isa.empty:
        def solve_isa(row):
            h_raw, t0_raw, p0_raw, t_raw, p_raw = row["h"], row["T0"], row["P0"], row["T"], row["P"]
            h = h_raw * UNITS["Longueur"][unit_h] if pd.notna(h_raw) else np.nan
            t0 = convert_temp_to_si(t0_raw, unit_t_isa) if pd.notna(t0_raw) else np.nan
            p0 = p0_raw * UNITS["Pression"][unit_p_isa] if pd.notna(p0_raw) else np.nan
            t = convert_temp_to_si(t_raw, unit_t_isa) if pd.notna(t_raw) else np.nan
            p = p_raw * UNITS["Pression"][unit_p_isa] if pd.notna(p_raw) else np.nan

            L, g, R = 0.0065, 9.80665, 287.05

            if pd.isna(h) and pd.notna(t0):
                if pd.notna(t): h = (t0 - t) / L
                elif pd.notna(p) and pd.notna(p0): h = (t0 / L) * (1 - (p / p0)**(R * L / g))

            if pd.notna(h) and pd.notna(t0) and pd.isna(t):
                t = t0 - L * h
            if pd.notna(h) and pd.notna(p0) and pd.notna(t0) and pd.isna(p):
                p = p0 * (1 - L * h / t0)**(g / (R * L))

            row["h"] = h / UNITS["Longueur"][unit_h] if pd.notna(h) else np.nan
            row["T"] = convert_temp_from_si(t, unit_t_isa) if pd.notna(t) else np.nan
            row["P"] = p / UNITS["Pression"][unit_p_isa] if pd.notna(p) else np.nan
            return row
            
        st.success("Résultats :")
        st.dataframe(edited_isa.apply(solve_isa, axis=1), use_container_width=True)

    st.markdown("---")

    # --- BLOC 3 : GAZ PARFAITS ---
    st.header("3. Loi des Gaz Parfaits")
    display_theory(
        nom="Équation d'état des gaz parfaits",
        usage="Relier la pression, la masse volumique et la température.",
        latex_form=r"P = \rho \cdot r \cdot T",
        hypotheses="Gaz parfait (pas d'interactions entre les molécules, volume propre nul).",
        limites="Dévie de la réalité à très haute pression ou très basse température."
    )

    df_gaz = pd.DataFrame([
        {"P": None, "Rho": 1.225, "r": 287.05, "T": 288.15},
        {"P": 101325.0, "Rho": None, "r": 287.05, "T": 288.15},
        {"P": 101325.0, "Rho": 1.225, "r": 287.05, "T": None},
    ])
    
    edited_gaz = st.data_editor(
        df_gaz, num_rows="dynamic", use_container_width=True,
        column_config={"P": "Pression (Pa)", "Rho": "Rho (kg/m³)", "r": "r (J/kg.K)", "T": "Température (K)"}
    )

    if not edited_gaz.empty:
        def solve_gas(row):
            p, rho, r, t = row["P"], row["Rho"], row["r"], row["T"]
            if pd.isna(p) and not pd.isna(rho) and not pd.isna(t) and not pd.isna(r): row["P"] = rho * r * t
            elif pd.isna(rho) and not pd.isna(p) and not pd.isna(t) and not pd.isna(r) and t != 0: row["Rho"] = p / (r * t)
            elif pd.isna(t) and not pd.isna(p) and not pd.isna(rho) and not pd.isna(r) and rho != 0: row["T"] = p / (rho * r)
            return row
            
        st.success("Résultats :")
        st.dataframe(edited_gaz.apply(solve_gas, axis=1), use_container_width=True)

# ==========================================
# PAGE 2 : SIMILITUDE
# ==========================================
elif page == "2. Similitude":
    st.title("⚖️ Similitude et Coefficients")

    # --- BLOC 1 : NOMBRE DE MACH ---
    st.header("1. Nombre de Mach")
    display_theory(
        nom="Vitesse du son et Nombre de Mach",
        usage="Caractérisation de la compressibilité d'un écoulement.",
        latex_form=r"a = \sqrt{\gamma R T} \quad \text{et} \quad M = \frac{V}{a}",
        hypotheses="Le fluide est considéré comme un gaz parfait. Propriétés locales.",
        limites="R et gamma dépendent du gaz."
    )

    col1, col2 = st.columns(2)
    unit_v = col1.selectbox("Unité de Vitesse", list(UNITS["Vitesse"].keys()), key="mach_v")
    unit_t = col2.selectbox("Unité de Température", ["K", "°C"], key="mach_t")

    df_mach = pd.DataFrame([
        {"V": 250.0, "T": 15.0, "Gamma": 1.4, "R": 287.05, "a": None, "M": None},
        {"V": None, "T": 15.0, "Gamma": 1.4, "R": 287.05, "a": None, "M": 0.8}
    ])
    
    edited_mach = st.data_editor(
        df_mach, num_rows="dynamic", use_container_width=True,
        column_config={
            "V": f"Vitesse V ({unit_v})", "T": f"Température T ({unit_t})",
            "a": f"Vitesse du son a ({unit_v})", "M": "Mach M"
        }
    )

    if not edited_mach.empty:
        def solve_mach(row):
            v_raw, t_raw, gamma, r, a_raw, m = row["V"], row["T"], row["Gamma"], row["R"], row["a"], row["M"]
            v = v_raw * UNITS["Vitesse"][unit_v] if pd.notna(v_raw) else np.nan
            t = convert_temp_to_si(t_raw, unit_t) if pd.notna(t_raw) else np.nan
            a = a_raw * UNITS["Vitesse"][unit_v] if pd.notna(a_raw) else np.nan

            if pd.isna(a) and pd.notna(t) and pd.notna(gamma) and pd.notna(r):
                a = np.sqrt(gamma * r * t)
            elif pd.isna(t) and pd.notna(a) and pd.notna(gamma) and pd.notna(r):
                t = a**2 / (gamma * r)

            if pd.isna(m) and pd.notna(v) and pd.notna(a) and a > 0:
                m = v / a
            elif pd.isna(v) and pd.notna(m) and pd.notna(a):
                v = m * a
            elif pd.isna(a) and pd.notna(m) and pd.notna(v) and m > 0:
                a = v / m
                if pd.isna(t) and pd.notna(gamma) and pd.notna(r):
                    t = a**2 / (gamma * r)

            row["V"] = v / UNITS["Vitesse"][unit_v] if pd.notna(v) else np.nan
            row["T"] = convert_temp_from_si(t, unit_t) if pd.notna(t) else np.nan
            row["a"] = a / UNITS["Vitesse"][unit_v] if pd.notna(a) else np.nan
            row["M"] = m
            return row

        st.success("Résultats :")
        st.dataframe(edited_mach.apply(solve_mach, axis=1), use_container_width=True)

    st.markdown("---")

    # --- BLOC 2 : COEFFICIENTS AERODYNAMIQUES ---
    st.header("2. Coefficients Aérodynamiques")
    display_theory(
        nom="Coefficients d'Efforts",
        usage="Passage des efforts physiques (Portance, Traînée) aux coefficients adimensionnels.",
        latex_form=r"q = \frac{1}{2} \rho V^2 \quad \text{et} \quad C_x = \frac{F}{q S}",
        hypotheses="Écoulement incompressible par défaut pour la pression dynamique.",
        limites="Ne tient pas compte des effets d'échelle (Reynolds) ni de compressibilité (Mach)."
    )

    col1, col2 = st.columns(2)
    unit_v2 = col1.selectbox("Unité de Vitesse ($V$)", list(UNITS["Vitesse"].keys()), key="coef_v")
    unit_f = col2.selectbox("Unité de Force ($F$)", list(UNITS["Force"].keys()), key="coef_f")
    
    df_coef = pd.DataFrame([
        {"F": 5000.0, "Rho": 1.225, "V": 50.0, "S": 15.0, "q": None, "C": None},
        {"F": None, "Rho": 1.225, "V": 50.0, "S": 15.0, "q": None, "C": 0.3}
    ])
    
    edited_coef = st.data_editor(
        df_coef, num_rows="dynamic", use_container_width=True,
        column_config={
            "F": f"Force ({unit_f})", "Rho": "Rho (kg/m³)", "V": f"Vitesse ({unit_v2})",
            "S": "Surface S (m²)", "q": "Pression Dyn q (Pa)", "C": "Coefficient Cx/Cz"
        }
    )

    if not edited_coef.empty:
        def solve_coef(row):
            f_raw, rho, v_raw, s, q, c = row["F"], row["Rho"], row["V"], row["S"], row["q"], row["C"]
            v = v_raw * UNITS["Vitesse"][unit_v2] if pd.notna(v_raw) else np.nan
            f = f_raw * UNITS["Force"][unit_f] if pd.notna(f_raw) else np.nan

            if pd.isna(q) and pd.notna(rho) and pd.notna(v):
                q = 0.5 * rho * v**2
            elif pd.isna(v) and pd.notna(q) and pd.notna(rho) and rho > 0:
                v = np.sqrt(2 * q / rho)
            elif pd.isna(rho) and pd.notna(q) and pd.notna(v) and v > 0:
                rho = 2 * q / v**2

            if pd.isna(c) and pd.notna(f) and pd.notna(q) and pd.notna(s) and q*s > 0:
                c = f / (q * s)
            elif pd.isna(f) and pd.notna(c) and pd.notna(q) and pd.notna(s):
                f = c * q * s
            elif pd.isna(s) and pd.notna(f) and pd.notna(c) and pd.notna(q) and c*q > 0:
                s = f / (c * q)
            elif pd.isna(q) and pd.notna(f) and pd.notna(c) and pd.notna(s) and c*s > 0:
                q = f / (c * s)
                if pd.isna(v) and pd.notna(rho) and rho > 0: v = np.sqrt(2 * q / rho)

            row["F"] = f / UNITS["Force"][unit_f] if pd.notna(f) else np.nan
            row["V"] = v / UNITS["Vitesse"][unit_v2] if pd.notna(v) else np.nan
            row["Rho"], row["S"], row["q"], row["C"] = rho, s, q, c
            return row
            
        st.success("Résultats :")
        st.dataframe(edited_coef.apply(solve_coef, axis=1), use_container_width=True)

# ==========================================
# PAGE 3 : COMPRESSIBLE
# ==========================================
elif page == "3. Compressible":
    st.title("💨 Aérodynamique Compressible")

    # --- BLOC 1 : RELATIONS ISENTROPIQUES ---
    st.header("1. Propriétés Génératrices (Relations Isentropiques)")
    display_theory(
        nom="Lois Isentropiques 1D",
        usage="Lien entre les propriétés locales (statiques) et génératrices (d'arrêt, totales).",
        latex_form=r"\frac{T_0}{T} = 1 + \frac{\gamma - 1}{2}M^2 \quad \text{et} \quad \frac{P_0}{P} = \left(1 + \frac{\gamma - 1}{2}M^2\right)^{\frac{\gamma}{\gamma - 1}}",
        hypotheses="Écoulement isentropique (adiabatique et réversible), gaz parfait.",
        limites="Faux s'il y a des chocs (non-isentropique), apports de chaleur ou frottements majeurs."
    )

    col1, col2 = st.columns(2)
    unit_p = col1.selectbox("Unité de Pression", list(UNITS["Pression"].keys()), key="comp_p")
    unit_t = col2.selectbox("Unité de Température", ["K", "°C"], key="comp_t")

    df_isen = pd.DataFrame([
        {"M": 0.8, "Ts": 15.0, "Ps": 80000.0, "Tt": None, "Pt": None, "g": 1.4},
        {"M": None, "Ts": 15.0, "Ps": None, "Tt": 45.0, "Pt": None, "g": 1.4}
    ])
    
    edited_isen = st.data_editor(
        df_isen, num_rows="dynamic", use_container_width=True,
        column_config={
            "M": "Mach", "g": "Gamma",
            "Ts": f"T statique ({unit_t})", "Ps": f"P statique ({unit_p})",
            "Tt": f"T totale/arrêt ({unit_t})", "Pt": f"P totale/arrêt ({unit_p})"
        }
    )

    if not edited_isen.empty:
        def solve_isen(row):
            m, ts_raw, ps_raw, tt_raw, pt_raw, g = row["M"], row["Ts"], row["Ps"], row["Tt"], row["Pt"], row["g"]
            ts = convert_temp_to_si(ts_raw, unit_t) if pd.notna(ts_raw) else np.nan
            tt = convert_temp_to_si(tt_raw, unit_t) if pd.notna(tt_raw) else np.nan
            ps = ps_raw * UNITS["Pression"][unit_p] if pd.notna(ps_raw) else np.nan
            pt = pt_raw * UNITS["Pression"][unit_p] if pd.notna(pt_raw) else np.nan

            if pd.isna(m) and pd.notna(g):
                if pd.notna(tt) and pd.notna(ts) and ts > 0:
                    val = (tt/ts - 1) * 2 / (g - 1)
                    if val >= 0: m = np.sqrt(val)
                elif pd.notna(pt) and pd.notna(ps) and ps > 0:
                    val = ((pt/ps)**((g-1)/g) - 1) * 2 / (g - 1)
                    if val >= 0: m = np.sqrt(val)

            if pd.notna(m) and pd.notna(g):
                t_ratio = 1 + ((g - 1) / 2) * m**2
                p_ratio = t_ratio ** (g / (g - 1))

                if pd.isna(tt) and pd.notna(ts): tt = ts * t_ratio
                if pd.isna(ts) and pd.notna(tt): ts = tt / t_ratio
                if pd.isna(pt) and pd.notna(ps): pt = ps * p_ratio
                if pd.isna(ps) and pd.notna(pt): ps = pt / p_ratio

            row["M"] = m
            row["Ts"] = convert_temp_from_si(ts, unit_t) if pd.notna(ts) else np.nan
            row["Tt"] = convert_temp_from_si(tt, unit_t) if pd.notna(tt) else np.nan
            row["Ps"] = ps / UNITS["Pression"][unit_p] if pd.notna(ps) else np.nan
            row["Pt"] = pt / UNITS["Pression"][unit_p] if pd.notna(pt) else np.nan
            return row
            
        st.success("Résultats :")
        st.dataframe(edited_isen.apply(solve_isen, axis=1), use_container_width=True)

    st.markdown("---")

    # --- BLOC 2 : RELATIONS DES AIRES ---
    st.header("2. Tuyères et Relation des Aires ($A/A^*$)")
    display_theory(
        nom="Fonction d'Aire (Théorème d'Hugoniot)",
        usage="Détermination de l'évolution de la section d'une tuyère en fonction du Mach (ou inversement).",
        latex_form=r"\frac{A}{A^*} = \frac{1}{M} \left[ \frac{2}{\gamma + 1} \left( 1 + \frac{\gamma - 1}{2} M^2 \right) \right]^{\frac{\gamma + 1}{2(\gamma - 1)}}",
        hypotheses="Écoulement quasi-1D, isentropique, gaz parfait. $A^*$ est la section critique (Mach = 1).",
        limites="Solveur numérique utilisé pour déduire Mach à partir du ratio d'aire (Préciser le régime !)."
    )

    df_area = pd.DataFrame([
        {"M": 2.0, "g": 1.4, "As": 0.05, "A": None, "R": None, "Regime": "Supersonique"},
        {"M": None, "g": 1.4, "As": 0.05, "A": None, "R": 2.5, "Regime": "Subsonique"}
    ])
    
    edited_area = st.data_editor(
        df_area, num_rows="dynamic", use_container_width=True,
        column_config={
            "M": "Mach", "g": "Gamma", "As": "Section A* (m²)", "A": "Section A (m²)", "R": "Ratio A/A*",
            "Regime": st.column_config.SelectboxColumn("Régime cible", options=["Subsonique", "Supersonique"], required=True)
        }
    )

    if not edited_area.empty:
        def area_ratio_func(M, gamma):
            if M <= 0: return np.nan
            term1 = 2 / (gamma + 1)
            term2 = 1 + ((gamma - 1) / 2) * M**2
            power = (gamma + 1) / (2 * (gamma - 1))
            return (1 / M) * ((term1 * term2) ** power)

        def solve_area(row):
            m, g, a_star, a, ratio, regime = row["M"], row["g"], row["As"], row["A"], row["R"], row["Regime"]

            if pd.isna(ratio) and pd.notna(a) and pd.notna(a_star) and a_star > 0:
                ratio = a / a_star

            if pd.isna(m) and pd.notna(ratio) and pd.notna(g) and ratio >= 1:
                def obj(M): return area_ratio_func(M, g) - ratio
                try:
                    if regime == "Subsonique":
                        sol = root_scalar(obj, bracket=[1e-4, 1.0])
                        m = sol.root
                    elif regime == "Supersonique":
                        sol = root_scalar(obj, bracket=[1.0, 20.0])
                        m = sol.root
                except:
                    pass

            if pd.notna(m) and pd.notna(g) and pd.isna(ratio):
                ratio = area_ratio_func(m, g)
                regime = "Supersonique" if m > 1 else "Subsonique"

            if pd.notna(ratio):
                if pd.isna(a) and pd.notna(a_star): a = ratio * a_star
                if pd.isna(a_star) and pd.notna(a): a_star = a / ratio

            row["M"], row["R"], row["As"], row["A"], row["Regime"] = m, ratio, a_star, a, regime
            return row

        st.success("Résultats :")
        st.dataframe(edited_area.apply(solve_area, axis=1), use_container_width=True)
