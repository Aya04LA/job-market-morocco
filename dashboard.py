# =============================================================================
# DASHBOARD — Job Market Intelligence Platform 🇲🇦
# =============================================================================
# Run with: streamlit run dashboard.py
#
# Pages:
#   1. 🏠 Vue d'ensemble   — KPIs + charts
#   2. 🔍 Explorer les offres — searchable jobs table
#   3. 📄 Analyse CV       — upload CV, get ATS score + job matches
#   4. 🤖 Performance ML   — MLflow metrics + classifier info
# =============================================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import re

# =============================================================================
# PAGE CONFIG — must be first Streamlit call
# =============================================================================
st.set_page_config(
    page_title="Emploi Maroc · Intelligence",
    page_icon="🇲🇦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# THEME SYSTEM
# Baby pink light mode + dark mode toggle
# =============================================================================

THEMES = {
    "🌙 Dark": {
        "bg":           "#0f0f14",
        "bg2":          "#1a1a24",
        "card":         "#1e1e2e",
        "border":       "#2e2e45",
        "text":         "#e8e8f0",
        "text2":        "#9090b0",
        "accent":       "#7c6fff",
        "accent2":      "#ff6b9d",
        "accent3":      "#00d4aa",
        "kpi_bg":       "#1e1e2e",
        "chart_theme":  "plotly_dark",
        "tag_bg":       "#2e2e45",
        "tag_text":     "#9090b0",
    },
    "🩷 Pink": {
        "bg":           "#fff0f5",
        "bg2":          "#fce4ec",
        "card":         "#ffffff",
        "border":       "#f8bbd0",
        "text":         "#2d1b2e",
        "text2":        "#8d5a6e",
        "accent":       "#e91e8c",
        "accent2":      "#ad1457",
        "accent3":      "#00897b",
        "kpi_bg":       "#fce4ec",
        "chart_theme":  "plotly_white",
        "tag_bg":       "#fce4ec",
        "tag_text":     "#ad1457",
    },
}

# Theme toggle in sidebar
if "theme" not in st.session_state:
    st.session_state.theme = "🌙 Dark"

T = THEMES[st.session_state.theme]

# =============================================================================
# GLOBAL CSS — injected based on current theme
# =============================================================================
def inject_css(T):
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

    /* Base */
    .stApp {{ background: {T['bg']}; color: {T['text']}; font-family: 'DM Sans', sans-serif; }}
    .stApp > header {{ background: transparent !important; }}
    [data-testid="stSidebar"] {{
        background: {T['bg2']} !important;
        border-right: 1px solid {T['border']};
    }}
    [data-testid="stSidebar"] * {{ color: {T['text']} !important; }}

    /* Remove default padding */
    .block-container {{ padding-top: 1.5rem !important; max-width: 1400px; }}

    /* KPI Cards */
    .kpi-card {{
        background: {T['kpi_bg']};
        border: 1px solid {T['border']};
        border-radius: 16px;
        padding: 20px 24px;
        text-align: center;
        transition: transform 0.2s;
    }}
    .kpi-card:hover {{ transform: translateY(-3px); }}
    .kpi-number {{
        font-family: 'DM Serif Display', serif;
        font-size: 2.4rem;
        color: {T['accent']};
        line-height: 1;
        margin-bottom: 4px;
    }}
    .kpi-label {{
        font-size: 0.78rem;
        color: {T['text2']};
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 500;
    }}

    /* Section titles */
    .section-title {{
        font-family: 'DM Serif Display', serif;
        font-size: 1.4rem;
        color: {T['text']};
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 8px;
        border-bottom: 2px solid {T['accent']};
        display: inline-block;
    }}

    /* Skill tags */
    .skill-tag {{
        display: inline-block;
        background: {T['tag_bg']};
        color: {T['tag_text']};
        border: 1px solid {T['border']};
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.75rem;
        margin: 2px;
        font-weight: 500;
    }}

    /* Score bar */
    .score-bar-wrap {{ background: {T['border']}; border-radius: 8px; height: 10px; overflow: hidden; margin: 6px 0; }}
    .score-bar-fill {{ height: 100%; border-radius: 8px; transition: width 0.6s ease; }}

    /* Match card */
    .match-card {{
        background: {T['card']};
        border: 1px solid {T['border']};
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
    }}
    .match-score {{
        font-family: 'DM Serif Display', serif;
        font-size: 1.8rem;
        color: {T['accent']};
    }}

    /* Page header */
    .page-header {{
        background: linear-gradient(135deg, {T['accent']}22, {T['accent2']}11);
        border: 1px solid {T['border']};
        border-radius: 20px;
        padding: 28px 36px;
        margin-bottom: 24px;
    }}
    .page-header h1 {{
        font-family: 'DM Serif Display', serif;
        font-size: 2rem;
        color: {T['text']};
        margin: 0;
    }}
    .page-header p {{
        color: {T['text2']};
        margin: 6px 0 0 0;
        font-size: 0.9rem;
    }}

    /* Streamlit overrides */
    .stSelectbox label, .stMultiSelect label, .stTextInput label, .stSlider label {{
        color: {T['text2']} !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    div[data-testid="metric-container"] {{
        background: {T['kpi_bg']};
        border: 1px solid {T['border']};
        border-radius: 12px;
        padding: 12px;
    }}
    .stDataFrame {{ border-radius: 12px; overflow: hidden; }}
    </style>
    """, unsafe_allow_html=True)

inject_css(T)

# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data
def load_data():
    """Load jobs_nlp.csv — cached so it only loads once per session."""
    path = Path("jobs_nlp.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    # Parse skills list from pipe-separated string
    df["skills_list"] = df["skills_str"].fillna("").apply(
        lambda x: [s.strip() for s in x.split("|") if s.strip()]
    )
    return df

@st.cache_data
def load_skill_freq():
    path = Path("skills_frequency.csv")
    if not path.exists():
        return None
    return pd.read_csv(path, encoding="utf-8-sig")

df       = load_data()
sf       = load_skill_freq()

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding: 12px 0 20px 0;'>
        <div style='font-family: DM Serif Display, serif; font-size: 1.4rem; color: {T["accent"]};'>
            🇲🇦 Emploi Maroc
        </div>
        <div style='font-size: 0.72rem; color: {T["text2"]}; letter-spacing: 0.1em; text-transform: uppercase;'>
            Intelligence du Marché
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Theme toggle
    theme_labels = {"🌙 Dark": "Dark mode", "🩷 Pink": "Pink mode"}
    new_theme = st.radio("Thème", list(THEMES.keys()), 
        format_func=lambda x: theme_labels[x],
        index=list(THEMES.keys()).index(st.session_state.theme))
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()

    st.markdown("---")

    page = st.radio("Navigation", [
        "🏠 Vue d'ensemble",
        "🔍 Explorer les offres",
        "📄 Analyse de CV",
        "🤖 Performance ML",
        "💬 Chatbot",
    ])

    st.markdown("---")
    if df is not None:
        st.markdown(f"<div style='font-size:0.75rem; color:{T['text2']};'>📊 {len(df)} offres analysées</div>", unsafe_allow_html=True)

# =============================================================================
# PAGE 1: VUE D'ENSEMBLE
# =============================================================================
if page == "🏠 Vue d'ensemble":

    st.markdown(f"""
    <div class='page-header'>
        <h1>Vue d'ensemble du marché 🇲🇦</h1>
        <p>Analyse en temps réel des offres d'emploi au Maroc — Rekrute & Emploi.ma</p>
    </div>
    """, unsafe_allow_html=True)

    if df is None:
        st.warning("⚠️ Fichier jobs_nlp.csv introuvable. Lancez d'abord le pipeline NLP.")
        st.stop()

    # ── KPI Row ─────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        (k1, len(df), "Offres analysées"),
        (k2, df["sector_final"].nunique(), "Secteurs"),
        (k3, df["location_clean"].nunique(), "Villes"),
        (k4, int(df["skills_count"].sum()), "Compétences extraites"),
        (k5, f"{(df['skills_count'] > 0).mean()*100:.0f}%", "Offres avec skills"),
    ]
    for col, val, label in kpis:
        col.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-number'>{val}</div>
            <div class='kpi-label'>{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Top Skills + Sector Distribution ──────────────────────────────
    c1, c2 = st.columns([3, 2])

    with c1:
        st.markdown("<div class='section-title'>Top compétences demandées</div>", unsafe_allow_html=True)
        if sf is not None:
            top_skills = sf.head(15).copy()
            fig = px.bar(
                top_skills, x="count", y="skill",
                orientation="h",
                color="category",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                labels={"count": "Nb d'offres", "skill": "", "category": "Catégorie"},
                template=T["chart_theme"],
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(autorange="reversed"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=0, r=0, t=30, b=0),
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("<div class='section-title'>Répartition par secteur</div>", unsafe_allow_html=True)
        sector_counts = df["sector_final"].value_counts().reset_index()
        sector_counts.columns = ["sector", "count"]
        fig2 = px.pie(
            sector_counts, names="sector", values="count",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Pastel,
            template=T["chart_theme"],
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="v", font=dict(size=11)),
            margin=dict(l=0, r=0, t=10, b=0),
            height=420,
        )
        fig2.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Row 2: Jobs by City + Contract Types ────────────────────────────────
    c3, c4 = st.columns([2, 1])

    with c3:
        st.markdown("<div class='section-title'>Offres par ville</div>", unsafe_allow_html=True)
        city_counts = df["location_clean"].value_counts().head(12).reset_index()
        city_counts.columns = ["ville", "count"]
        fig3 = px.bar(
            city_counts, x="ville", y="count",
            color="count",
            color_continuous_scale=["#f8bbd0", "#e91e8c"] if "Pink" in st.session_state.theme else ["#2e2e45", "#7c6fff"],
            template=T["chart_theme"],
            labels={"count": "Offres", "ville": ""},
        )
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=10, b=0),
            height=320,
        )
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.markdown("<div class='section-title'>Types de contrat</div>", unsafe_allow_html=True)
        contract_counts = df["contract_clean"].value_counts().reset_index()
        contract_counts.columns = ["contrat", "count"]
        fig4 = px.pie(
            contract_counts, names="contrat", values="count",
            hole=0.5,
            color_discrete_sequence=["#e91e8c","#f06292","#f48fb1","#fce4ec","#880e4f","#ad1457"]
            if "Pink" in st.session_state.theme
            else ["#7c6fff","#ff6b9d","#00d4aa","#ffd166","#06d6a0","#ef476f"],
            template=T["chart_theme"],
        )
        fig4.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
            height=320,
            showlegend=True,
            legend=dict(font=dict(size=10)),
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── Skill categories breakdown ───────────────────────────────────────────
    st.markdown("<div class='section-title'>Compétences par catégorie</div>", unsafe_allow_html=True)
    if sf is not None:
        cat_counts = sf.groupby("category")["count"].sum().reset_index().sort_values("count", ascending=False)
        fig5 = px.treemap(
            cat_counts, path=["category"], values="count",
            color="count",
            color_continuous_scale=["#fce4ec","#e91e8c"] if "Pink" in st.session_state.theme else ["#1e1e2e","#7c6fff"],
            template=T["chart_theme"],
        )
        fig5.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
            height=280,
        )
        st.plotly_chart(fig5, use_container_width=True)


# =============================================================================
# PAGE 2: EXPLORER LES OFFRES
# =============================================================================
elif page == "🔍 Explorer les offres":

    st.markdown(f"""
    <div class='page-header'>
        <h1>Explorer les offres 🔍</h1>
        <p>Filtrez et recherchez parmi toutes les offres analysées</p>
    </div>
    """, unsafe_allow_html=True)

    if df is None:
        st.warning("⚠️ Données introuvables.")
        st.stop()

    # ── Filters ─────────────────────────────────────────────────────────────
    f1, f2, f3, f4 = st.columns(4)

    with f1:
        sectors = ["Tous"] + sorted(df["sector_final"].dropna().unique().tolist())
        sector_filter = st.selectbox("Secteur", sectors)
    with f2:
        cities = ["Toutes"] + sorted(df["location_clean"].dropna().unique().tolist())
        city_filter = st.selectbox("Ville", cities)
    with f3:
        contracts = ["Tous"] + sorted(df["contract_clean"].dropna().unique().tolist())
        contract_filter = st.selectbox("Contrat", contracts)
    with f4:
        keyword = st.text_input("🔎 Recherche libre", placeholder="ex: Python, data, marketing...")

    # Apply filters
    filtered = df.copy()
    if sector_filter   != "Tous":    filtered = filtered[filtered["sector_final"]    == sector_filter]
    if city_filter     != "Toutes":  filtered = filtered[filtered["location_clean"]  == city_filter]
    if contract_filter != "Tous":    filtered = filtered[filtered["contract_clean"]  == contract_filter]
    if keyword:
        mask = (
            filtered["title"].str.contains(keyword, case=False, na=False) |
            filtered["description_clean"].str.contains(keyword, case=False, na=False) |
            filtered["skills_str"].str.contains(keyword, case=False, na=False)
        )
        filtered = filtered[mask]

    st.markdown(f"<div style='color:{T['text2']}; font-size:0.85rem; margin-bottom:12px;'>📋 {len(filtered)} offre(s) trouvée(s)</div>", unsafe_allow_html=True)

    # ── Results table ────────────────────────────────────────────────────────
    display_cols = {
        "title":          "Titre",
        "company":        "Entreprise",
        "location_clean": "Ville",
        "sector_final":   "Secteur",
        "contract_clean": "Contrat",
        "experience":     "Expérience",
        "skills_str":     "Compétences",
    }
    show_df = filtered[list(display_cols.keys())].rename(columns=display_cols)
    st.dataframe(show_df, use_container_width=True, height=500)

    # ── Job detail expander ──────────────────────────────────────────────────
    if len(filtered) > 0:
        st.markdown("<div class='section-title'>Détail d'une offre</div>", unsafe_allow_html=True)
        job_titles = filtered["title"].tolist()
        selected_title = st.selectbox("Sélectionner une offre", job_titles)
        job = filtered[filtered["title"] == selected_title].iloc[0]

        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.markdown(f"### {job['title']}")
            st.markdown(f"**{job.get('company','N/A')}** · {job.get('location_clean','N/A')} · {job.get('contract_clean','N/A')}")
            if job.get("url") and job["url"] != "N/A":
                st.markdown(f"[🔗 Voir l'offre originale]({job['url']})")
            st.markdown("**Description:**")
            desc = str(job.get("description_clean","")).strip()
            st.markdown(desc[:1500] + ("..." if len(desc) > 1500 else ""))

        with col_b:
            st.markdown("**Compétences détectées:**")
            skills = job.get("skills_list", [])
            if not isinstance(skills, list):
                skills = str(job.get("skills_str","")).split("|")
            tags = " ".join([f"<span class='skill-tag'>{s.strip()}</span>" for s in skills if s.strip()])
            st.markdown(tags or "*Aucune compétence détectée*", unsafe_allow_html=True)

            st.markdown("**Infos:**")
            info_items = {
                "📍 Ville":       job.get("location_clean", "N/A"),
                "🏢 Secteur":     job.get("sector_final",   "N/A"),
                "📝 Contrat":     job.get("contract_clean", "N/A"),
                "🎓 Formation":   job.get("education",      "N/A"),
                "⏱️ Expérience":  job.get("experience",     "N/A"),
                "💰 Salaire":     job.get("salary",         "N/A"),
            }
            for label, val in info_items.items():
                st.markdown(f"**{label}:** {val}")


# =============================================================================
# PAGE 3: ANALYSE DE CV
# =============================================================================
elif page == "📄 Analyse de CV":

    st.markdown(f"""
    <div class='page-header'>
        <h1>Analyse de CV & Matching 📄</h1>
        <p>Uploadez votre CV — obtenez votre score ATS, vos compétences manquantes et les offres qui vous correspondent</p>
    </div>
    """, unsafe_allow_html=True)

    if df is None:
        st.warning("⚠️ Données introuvables.")
        st.stop()

    uploaded = st.file_uploader("Déposez votre CV ici", type=["pdf", "txt"])

    def extract_text_from_pdf(file_bytes):
        """Extract text from PDF bytes using pdfminer."""
        try:
            import pdfminer.high_level
            import io
            return pdfminer.high_level.extract_text(io.BytesIO(file_bytes))
        except ImportError:
            try:
                import PyPDF2, io
                reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
                return " ".join(p.extract_text() or "" for p in reader.pages)
            except Exception:
                return ""

    def extract_cv_skills(text):
        """Reuse the same skill dictionary from nlp_pipeline to extract skills from CV."""
        from nlp_pipeline import SKILLS_BY_CATEGORY, SKILL_TO_CATEGORY
        text_lower = text.lower()
        found = {}
        for skill, category in SKILL_TO_CATEGORY.items():
            # Simple substring match for CV (faster than spaCy for single doc)
            if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
                found[skill] = category
        return found

    def compute_job_match(cv_skills_set, job_skills_list):
        """
        Compute match score between CV skills and job required skills.
        Score = Jaccard similarity = intersection / union
        WHY Jaccard? It's symmetric and handles different set sizes fairly.
        A CV with 20 skills matching 5/10 job skills scores higher than
        one with 5 skills matching 5/5 (which would be 100% but might miss
        important skills). Jaccard balances both sides.
        """
        if not job_skills_list or not cv_skills_set:
            return 0.0
        job_set = set(s.strip().lower() for s in job_skills_list if s.strip())
        if not job_set:
            return 0.0
        intersection = cv_skills_set & job_set
        union        = cv_skills_set | job_set
        return len(intersection) / len(union) if union else 0.0

    if uploaded:
        file_bytes = uploaded.read()

        # Extract text
        if uploaded.name.endswith(".pdf"):
            cv_text = extract_text_from_pdf(file_bytes)
        else:
            cv_text = file_bytes.decode("utf-8", errors="ignore")

        if not cv_text.strip():
            st.error("Impossible d'extraire le texte du CV. Essayez un fichier .txt.")
            st.stop()

        with st.spinner("Analyse en cours..."):
            cv_skills_dict = extract_cv_skills(cv_text)
            cv_skills_set  = set(cv_skills_dict.keys())

        # ── ATS Score ────────────────────────────────────────────────────────
        # ATS score = how many of the top 50 most-demanded skills you have
        if sf is not None:
            top50_skills  = set(sf.head(50)["skill"].tolist())
            ats_matches   = cv_skills_set & top50_skills
            ats_score     = min(int(len(ats_matches) / max(len(top50_skills), 1) * 100 * 3), 100)
        else:
            ats_score = min(len(cv_skills_set) * 4, 100)

        score_color = "#00897b" if ats_score >= 70 else "#e91e8c" if ats_score >= 40 else "#d32f2f"

        s1, s2, s3 = st.columns(3)
        s1.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-number' style='color:{score_color};'>{ats_score}%</div>
            <div class='kpi-label'>Score ATS</div>
        </div>""", unsafe_allow_html=True)
        s2.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-number'>{len(cv_skills_set)}</div>
            <div class='kpi-label'>Compétences détectées</div>
        </div>""", unsafe_allow_html=True)

        # Count CV words as proxy for CV length
        cv_words = len(cv_text.split())
        s3.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-number'>{cv_words}</div>
            <div class='kpi-label'>Mots dans le CV</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        cv_col, match_col = st.columns([1, 1])

        with cv_col:
            # ── Detected skills ──────────────────────────────────────────────
            st.markdown("<div class='section-title'>Compétences détectées</div>", unsafe_allow_html=True)
            by_cat = {}
            for skill, cat in cv_skills_dict.items():
                by_cat.setdefault(cat, []).append(skill)

            for cat, skills in sorted(by_cat.items()):
                st.markdown(f"**{cat}**")
                tags = " ".join([f"<span class='skill-tag'>{s}</span>" for s in sorted(skills)])
                st.markdown(tags, unsafe_allow_html=True)
                st.markdown("")

            # ── Missing skills ───────────────────────────────────────────────
            if sf is not None:
                missing = sf[~sf["skill"].isin(cv_skills_set)].head(20)
                
                tech_missing = missing[~missing["category"].isin(["Soft Skills", "Languages"])]
                soft_missing = missing[missing["category"].isin(["Soft Skills", "Languages"])]

                st.markdown("<div class='section-title'>Skills techniques manquants</div>", unsafe_allow_html=True)
                tags = " ".join([
                    f"<span class='skill-tag' style='border-color:#e91e8c; color:#e91e8c;'>+ {row['skill']}</span>"
                    for _, row in tech_missing.iterrows()
                ])
                st.markdown(tags or "*Aucun — bon profil technique !*", unsafe_allow_html=True)

                st.markdown("<div class='section-title'>Soft skills & langues manquants</div>", unsafe_allow_html=True)
                tags2 = " ".join([
                    f"<span class='skill-tag' style='border-color:#9090b0; color:#9090b0;'>+ {row['skill']}</span>"
                    for _, row in soft_missing.iterrows()
                ])
                st.markdown(tags2 or "*Aucun*", unsafe_allow_html=True)
        with match_col:
            # ── Job matching ─────────────────────────────────────────────────
            st.markdown("<div class='section-title'>Offres qui vous correspondent</div>", unsafe_allow_html=True)

            # Compute match for all jobs
            jobs_with_skills = df[df["skills_count"] > 0].copy()
            jobs_with_skills["match_score"] = jobs_with_skills["skills_list"].apply(
                lambda s: compute_job_match(cv_skills_set, s)
            )
            top_matches = jobs_with_skills.nlargest(8, "match_score")

            for _, row in top_matches.iterrows():
                pct = int(row["match_score"] * 100)
                bar_color = "#00897b" if pct >= 60 else "#e91e8c" if pct >= 30 else "#9e9e9e"
                matching_skills = cv_skills_set & set(row["skills_list"])

                st.markdown(f"""
                <div class='match-card'>
                    <div style='display:flex; justify-content:space-between; align-items:center;'>
                        <div>
                            <div style='font-weight:600; font-size:0.9rem; color:{T["text"]};'>{str(row["title"])[:60]}</div>
                            <div style='font-size:0.78rem; color:{T["text2"]};'>{row.get("company","N/A")} · {row.get("location_clean","N/A")}</div>
                        </div>
                        <div class='match-score' style='color:{bar_color};'>{pct}%</div>
                    </div>
                    <div class='score-bar-wrap'>
                        <div class='score-bar-fill' style='width:{pct}%; background:{bar_color};'></div>
                    </div>
                    <div style='font-size:0.72rem; color:{T["text2"]}; margin-top:4px;'>
                        Compétences communes: {", ".join(sorted(matching_skills)[:5]) or "—"}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if row.get("url") and str(row["url"]) not in ("N/A", "nan"):
                    st.markdown(f"[🔗 Voir l'offre]({row['url']})", unsafe_allow_html=True)


# =============================================================================
# PAGE 4: PERFORMANCE ML
# =============================================================================
elif page == "🤖 Performance ML":

    st.markdown(f"""
    <div class='page-header'>
        <h1>Performance du Modèle 🤖</h1>
        <p>Métriques du classifieur de secteurs et suivi des expériences MLflow</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Static metrics from last pipeline run ───────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    metrics_data = [
        (m1, "0.452", "CV F1-macro"),
        (m2, "TF-IDF + LogReg", "Modèle"),
        (m3, "364", "Échantillons d'entraînement"),
        (m4, "8", "Classes de secteurs"),
    ]
    for col, val, label in metrics_data:
        col.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-number' style='font-size:1.6rem;'>{val}</div>
            <div class='kpi-label'>{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Try to load MLflow runs ──────────────────────────────────────────────
    try:
        import mlflow
        mlflow.set_tracking_uri("./mlruns")
        client = mlflow.tracking.MlflowClient()
        exp    = client.get_experiment_by_name("job_market_morocco")

        if exp:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["start_time DESC"],
                max_results=20,
            )
            if runs:
                st.markdown("<div class='section-title'>Historique des runs MLflow</div>", unsafe_allow_html=True)
                rows = []
                for r in runs:
                    rows.append({
                        "Run":          r.info.run_name or r.info.run_id[:8],
                        "Date":         pd.to_datetime(r.info.start_time, unit="ms").strftime("%Y-%m-%d %H:%M"),
                        "CV F1":        round(r.data.metrics.get("cv_f1_macro", 0), 3),
                        "Jobs total":   int(r.data.metrics.get("n_jobs_total", 0)),
                        "Secteurs prédits": int(r.data.metrics.get("n_sectors_predicted", 0)),
                        "Logreg C":     r.data.params.get("logreg_C", "N/A"),
                        "Statut":       r.info.status,
                    })
                runs_df = pd.DataFrame(rows)
                st.dataframe(runs_df, use_container_width=True)

                # F1 over time chart
                if len(runs_df) > 1:
                    st.markdown("<div class='section-title'>Évolution du F1 au fil des runs</div>", unsafe_allow_html=True)
                    fig = px.line(
                        runs_df[::-1], x="Date", y="CV F1",
                        markers=True,
                        template=T["chart_theme"],
                        color_discrete_sequence=[T["accent"]],
                    )
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=300,
                        margin=dict(l=0, r=0, t=10, b=0),
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Aucun run MLflow trouvé. Lancez `python mlflow_tracking.py` d'abord.")
        else:
            st.info("Expérience MLflow 'job_market_morocco' non trouvée. Lancez le pipeline tracké.")

    except Exception as e:
        st.warning(f"MLflow non accessible : {e}")
        st.markdown("Lancez `mlflow ui` dans le terminal pour accéder à l'interface complète.")

    # ── Model explanation ─────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Choix du modèle — justification</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='match-card'>
        <p style='color:{T["text"]};'><strong>Pourquoi TF-IDF + Logistic Regression ?</strong></p>
        <ul style='color:{T["text2"]}; font-size:0.9rem; line-height:1.8;'>
            <li><strong>Dataset petit (364 exemples)</strong> — Les transformers (BERT, CamemBERT) nécessitent au minimum quelques milliers d'exemples pour un fine-tuning stable. Avec 364 exemples, TF-IDF + LogReg donne de meilleurs résultats.</li>
            <li><strong>Interprétabilité</strong> — On peut voir quels mots influencent chaque prédiction. Essentiel pour un système de décision en entreprise.</li>
            <li><strong>Rapidité</strong> — Entraînement en &lt;1s vs plusieurs minutes pour un transformer. Crucial pour un pipeline MLOps qui re-entraîne à chaque scraping.</li>
            <li><strong>F1-macro = 0.452</strong> — Correct pour 8 classes déséquilibrées (3 offres Logistique vs 185 IT). Le F1 pondéré par classe est bien plus élevé sur IT &amp; Finance.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    if df is not None:
        st.markdown("<div class='section-title'>Distribution finale des secteurs</div>", unsafe_allow_html=True)
        sec_df = df["sector_final"].value_counts().reset_index()
        sec_df.columns = ["Secteur", "Nombre d'offres"]
        pred_count = df.groupby("sector_final")["sector_predicted"].sum().reset_index()
        pred_count.columns = ["Secteur", "Prédits par ML"]
        sec_df = sec_df.merge(pred_count, on="Secteur")
        sec_df["% prédits"] = (sec_df["Prédits par ML"] / sec_df["Nombre d'offres"] * 100).round(1).astype(str) + "%"
        st.dataframe(sec_df, use_container_width=True)


# =============================================================================
# PAGE 5: Chatbot
# =============================================================================
elif page == "💬 Chatbot":
 
    import os
    from groq import Groq
 
    st.markdown(f"""
    <div class='page-header'>
        <h1>Assistant IA 💬</h1>
        <p>Posez vos questions sur le marché de l'emploi au Maroc ou demandez des conseils pour votre CV</p>
    </div>
    """, unsafe_allow_html=True)
 
    # ── Mode selector ────────────────────────────────────────────────────────
    mode = st.radio(
        "Mode",
        ["🧭 Marché de l'emploi", "📄 Conseils CV"],
        horizontal=True,
        label_visibility="collapsed",
    )
 
    # ── Build system prompt based on mode ────────────────────────────────────
    # WHY different system prompts?
    # The system prompt shapes the model's persona and focus.
    # A job market analyst and a CV coach have different expertise and tone.
    # Switching the system prompt is cheaper and faster than two separate models.
 
    if df is not None:
        # Build a compact market summary to inject as context
        # WHY inject data? The LLM has no knowledge of YOUR dataset.
        # We summarize the key stats and pass them in the system prompt
        # so the chatbot answers based on real scraped data, not generic knowledge.
        top_skills    = sf.head(10)["skill"].tolist() if sf is not None else []
        top_sectors   = df["sector_final"].value_counts().head(5).to_dict() if df is not None else {}
        top_cities    = df["location_clean"].value_counts().head(5).to_dict() if df is not None else {}
        n_jobs        = len(df) if df is not None else 0
 
        market_context = f"""
Tu as accès aux données réelles du marché de l'emploi au Maroc (scraped depuis Rekrute et Emploi.ma):
- {n_jobs} offres d'emploi analysées
- Top compétences demandées: {", ".join(top_skills)}
- Top secteurs: {", ".join([f"{k} ({v} offres)" for k, v in top_sectors.items()])}
- Top villes: {", ".join([f"{k} ({v} offres)" for k, v in top_cities.items()])}
"""
    else:
        market_context = "Tu analyses le marché de l'emploi au Maroc."
 
    SYSTEM_PROMPTS = {
        "🧭 Marché de l'emploi": f"""Tu es un expert analyste du marché de l'emploi au Maroc.
{market_context}
Réponds en français. Sois précis, data-driven, et cite les chiffres réels quand tu les connais.
Si on te pose une question hors sujet emploi/marché marocain, redirige poliment vers ton domaine.
Garde tes réponses concises (max 200 mots sauf si on demande plus de détails).""",
 
        "📄 Conseils CV": f"""Tu es un coach CV expert pour le marché marocain, spécialisé dans les secteurs tech, finance et conseil (IBM, Oracle, BCG Maroc).
{market_context}
Tu aides à:
- Améliorer la rédaction des expériences et projets
- Optimiser le CV pour les ATS (Applicant Tracking Systems)
- Identifier les compétences à mettre en avant selon le secteur visé
- Préparer les candidatures pour des grandes entreprises au Maroc
Réponds en français. Sois direct et actionnable. Max 200 mots sauf si on demande plus.""",
    }
 
    # ── Initialize chat history ───────────────────────────────────────────────
    # WHY session_state? Streamlit reruns the entire script on every interaction.
    # session_state persists data across reruns — it's how you keep chat history.
    # We use a different key per mode so switching modes clears the conversation.
 
    history_key = f"chat_history_{mode}"
    if history_key not in st.session_state:
        st.session_state[history_key] = []
 
    # ── Display chat history ──────────────────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        if not st.session_state[history_key]:
            # Welcome message
            welcome = {
                "🧭 Marché de l'emploi": "Bonjour ! Je suis votre analyste du marché de l'emploi marocain. Posez-moi vos questions sur les secteurs, les compétences demandées, les salaires ou les villes les plus actives. 📊",
                "📄 Conseils CV": "Bonjour ! Je suis votre coach CV pour le marché marocain. Partagez votre expérience ou vos questions — je vous aide à optimiser votre candidature pour IBM, Oracle, BCG et les grandes entreprises au Maroc. 🎯",
            }
            st.markdown(f"""
            <div style='background:{T["card"]}; border:1px solid {T["border"]};
                        border-radius:12px; padding:16px 20px; margin:8px 0;
                        border-left: 3px solid {T["accent"]};'>
                <div style='font-size:0.75rem; color:{T["text2"]}; margin-bottom:6px;'>🤖 Assistant</div>
                <div style='color:{T["text"]}; font-size:0.9rem;'>{welcome[mode]}</div>
            </div>
            """, unsafe_allow_html=True)
 
        for msg in st.session_state[history_key]:
            is_user = msg["role"] == "user"
            align   = "right" if is_user else "left"
            bg      = T["accent"] + "22" if is_user else T["card"]
            border  = T["accent"] if is_user else T["border"]
            label   = "Vous" if is_user else "🤖 Assistant"
 
            st.markdown(f"""
            <div style='background:{bg}; border:1px solid {border};
                        border-radius:12px; padding:14px 18px; margin:6px 0;
                        border-left: 3px solid {border};'>
                <div style='font-size:0.75rem; color:{T["text2"]}; margin-bottom:4px;'>{label}</div>
                <div style='color:{T["text"]}; font-size:0.9rem; white-space:pre-wrap;'>{msg["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
 
    # ── Input ─────────────────────────────────────────────────────────────────
    with st.form(key=f"chat_form_{mode}", clear_on_submit=True):
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            user_input = st.text_input(
                "Message",
                placeholder="Ex: Quelles sont les compétences les plus demandées à Casablanca ?" if "emploi" in mode else "Ex: Comment améliorer mon expérience en data science ?",
                label_visibility="collapsed",
            )
        with col_btn:
            submitted = st.form_submit_button("Envoyer →", use_container_width=True)
 
    # ── Call Groq API ─────────────────────────────────────────────────────────
    if submitted and user_input.strip():
        # Add user message to history
        st.session_state[history_key].append({
            "role": "user",
            "content": user_input.strip(),
        })
 
        # Get API key from environment (HuggingFace secret or local .env)
        api_key = os.environ.get("GROQ_API_KEY", "")
 
        if not api_key:
            st.error("⚠️ GROQ_API_KEY non trouvée. Ajoutez-la dans les secrets de votre Space HuggingFace.")
        else:
            try:
                with st.spinner("Réflexion en cours..."):
                    client = Groq(api_key=api_key)
 
                    # Build messages list: system prompt + full history
                    # WHY full history? The model has no memory between calls.
                    # We resend the entire conversation each time so it has context.
                    # This is called "context window management" — standard in LLM apps.
                    messages = [{"role": "system", "content": SYSTEM_PROMPTS[mode]}]
                    messages += st.session_state[history_key]
 
                    response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=messages,
                        max_tokens=400,      # keep responses concise
                        temperature=0.7,     # slight creativity, not too random
                    )
 
                    assistant_reply = response.choices[0].message.content.strip()
 
                # Add assistant reply to history
                st.session_state[history_key].append({
                    "role": "assistant",
                    "content": assistant_reply,
                })
 
                st.rerun()  # refresh to show new messages
 
            except Exception as e:
                st.error(f"Erreur API : {e}")
 
    # ── Clear conversation button ─────────────────────────────────────────────
    if st.session_state[history_key]:
        if st.button("🗑️ Effacer la conversation"):
            st.session_state[history_key] = []
            st.rerun()
 
    # ── Suggested questions ───────────────────────────────────────────────────
    if not st.session_state[history_key]:
        suggestions = {
            "🧭 Marché de l'emploi": [
                "Quelles compétences techniques sont les plus demandées au Maroc ?",
                "Quel secteur recrute le plus à Casablanca ?",
                "Quel est le profil type d'une offre CDI en IT au Maroc ?",
                "Quelles villes offrent le plus d'opportunités en finance ?",
            ],
            "📄 Conseils CV": [
                "Comment présenter un projet de Machine Learning sur mon CV ?",
                "Quelles compétences mettre en avant pour IBM Maroc ?",
                "Comment optimiser mon CV pour les ATS ?",
                "J'ai un stage de 2 mois, comment le valoriser ?",
            ],
        }
        st.markdown(f"<div style='color:{T['text2']}; font-size:0.8rem; margin-top:16px;'>💡 Suggestions :</div>", unsafe_allow_html=True)
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions[mode]):
            with cols[i % 2]:
                st.markdown(f"""
                <div style='background:{T["card"]}; border:1px solid {T["border"]};
                            border-radius:8px; padding:10px 14px; margin:4px 0;
                            font-size:0.8rem; color:{T["text2"]};'>
                    {suggestion}
                </div>
                """, unsafe_allow_html=True)