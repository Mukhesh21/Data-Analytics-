"""
Bird Migration Intelligence Platform v4
- No interactive play/reset buttons on charts — all render instantly
- Scroll-triggered fly-in animations (Figma-style intersection observer)
- Map: realistic coloured globe, click country → all individual bird names
- Click any bird name → full detail panel with image, success/failure analysis
- Bird images via Wikipedia / iNaturalist APIs
- Deployable with gunicorn
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import requests, os, re, json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# 1. DATA
# ══════════════════════════════════════════════════════════════
CSV_PATH = os.environ.get(
    "CSV_PATH",
   r"bird_migration.csv"
)
df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
df.columns = df.columns.str.strip()
df["Success"] = df["Migration_Success"] == "Successful"

MONTH_ORDER = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
df["Migration_Start_Month"] = pd.Categorical(
    df["Migration_Start_Month"], categories=MONTH_ORDER, ordered=True
)

ALL_SPECIES  = sorted(df["Species"].unique())
ALL_REGIONS  = sorted(df["Region"].unique())
ALL_BIRDS    = sorted(df["Bird_Name"].unique())
ALL_REASONS  = sorted(df["Interrupted_Reason"].dropna().unique())

# Birds grouped by species for the map panel
SPECIES_BIRDS = {sp: sorted(df[df["Species"] == sp]["Bird_Name"].unique())
                 for sp in ALL_SPECIES}

# ══════════════════════════════════════════════════════════════
# 2. GEO DATA
# ══════════════════════════════════════════════════════════════
CONTINENT_COUNTRIES = {
    "Africa":        ["Algeria","Angola","DR Congo","Egypt","Ethiopia","Ghana",
                      "Kenya","Morocco","Nigeria","South Africa","Sudan","Tanzania","Uganda","Zimbabwe"],
    "Asia":          ["Bangladesh","China","India","Indonesia","Iran","Japan","Malaysia",
                      "Myanmar","Pakistan","Philippines","South Korea","Sri Lanka","Thailand","Vietnam"],
    "Australia":     ["Australia","Fiji","New Zealand","Papua New Guinea","Solomon Islands","Vanuatu"],
    "Europe":        ["Belgium","Denmark","France","Germany","Greece","Italy","Netherlands",
                      "Norway","Poland","Portugal","Spain","Sweden","Switzerland","UK","Ukraine"],
    "North America": ["Canada","Costa Rica","Cuba","Guatemala","Honduras","Mexico","Panama","USA"],
    "South America": ["Argentina","Bolivia","Brazil","Chile","Colombia","Ecuador",
                      "Paraguay","Peru","Uruguay","Venezuela"],
}
ALL_COUNTRIES = {c: cont for cont, cl in CONTINENT_COUNTRIES.items() for c in cl}

COUNTRY_COORDS = {
    "Algeria":(28,1.6),"Angola":(-11,18),"DR Congo":(-4,22),"Egypt":(27,31),
    "Ethiopia":(9,40),"Ghana":(8,-1),"Kenya":(0,38),"Morocco":(32,-5),
    "Nigeria":(9,9),"South Africa":(-29,25),"Sudan":(16,33),"Tanzania":(-6,35),
    "Uganda":(1,32),"Zimbabwe":(-20,30),
    "Bangladesh":(24,90),"China":(36,104),"India":(21,79),"Indonesia":(-1,114),
    "Iran":(32,54),"Japan":(36,138),"Malaysia":(4,108),"Myanmar":(19,97),
    "Pakistan":(30,69),"Philippines":(13,123),"South Korea":(37,128),
    "Sri Lanka":(7.8,80.7),"Thailand":(16,101),"Vietnam":(14,108),
    "Australia":(-25,134),"Fiji":(-18,178),"New Zealand":(-41,175),
    "Papua New Guinea":(-6,144),"Solomon Islands":(-9,160),"Vanuatu":(-15,167),
    "Belgium":(50,4),"Denmark":(56,10),"France":(46,2),"Germany":(51,10),
    "Greece":(39,22),"Italy":(42,13),"Netherlands":(52,5),"Norway":(61,9),
    "Poland":(52,19),"Portugal":(40,-8),"Spain":(40,-4),"Sweden":(60,19),
    "Switzerland":(47,8),"UK":(55,-3),"Ukraine":(48,31),
    "Canada":(56,-106),"Costa Rica":(10,-84),"Cuba":(22,-80),
    "Guatemala":(16,-90),"Honduras":(15,-86),"Mexico":(24,-103),
    "Panama":(9,-81),"USA":(37,-96),
    "Argentina":(-38,-64),"Bolivia":(-17,-65),"Brazil":(-14,-52),
    "Chile":(-36,-72),"Colombia":(5,-74),"Ecuador":(-2,-78),
    "Paraguay":(-23,-58),"Peru":(-9,-75),"Uruguay":(-33,-56),"Venezuela":(6,-67),
}

REGION_CENTRE = {
    "Africa":(4,21),"Asia":(34,108),"Australia":(-25,134),
    "Europe":(50,10),"North America":(45,-93),"South America":(-15,-55),
}

ROUTE_WP = {
    ("Africa","Asia"):[(4,21),(20,45),(30,60),(34,108)],
    ("Africa","Europe"):[(4,21),(15,15),(35,10),(50,10)],
    ("Africa","North America"):[(4,21),(15,-15),(25,-60),(45,-93)],
    ("Africa","Australia"):[(4,21),(5,50),(-10,80),(-25,134)],
    ("Africa","South America"):[(4,21),(0,-10),(-10,-35),(-15,-55)],
    ("Asia","Africa"):[(34,108),(20,70),(10,45),(4,21)],
    ("Asia","Europe"):[(34,108),(50,90),(55,40),(50,10)],
    ("Asia","Australia"):[(34,108),(20,110),(5,120),(-25,134)],
    ("Asia","North America"):[(34,108),(50,140),(55,-170),(45,-93)],
    ("Asia","South America"):[(34,108),(10,110),(0,-80),(-15,-55)],
    ("Europe","Africa"):[(50,10),(40,10),(20,10),(4,21)],
    ("Europe","Asia"):[(50,10),(55,40),(50,80),(34,108)],
    ("Europe","North America"):[(50,10),(55,-20),(50,-50),(45,-93)],
    ("Europe","South America"):[(50,10),(30,-20),(10,-40),(-15,-55)],
    ("Europe","Australia"):[(50,10),(30,40),(10,80),(-25,134)],
    ("North America","South America"):[(45,-93),(20,-90),(0,-75),(-15,-55)],
    ("North America","Europe"):[(45,-93),(50,-50),(55,-20),(50,10)],
    ("North America","Asia"):[(45,-93),(55,-140),(55,170),(34,108)],
    ("North America","Africa"):[(45,-93),(25,-50),(10,-20),(4,21)],
    ("North America","Australia"):[(45,-93),(20,-130),(0,150),(-25,134)],
    ("South America","Africa"):[(-15,-55),(-5,-30),(0,10),(4,21)],
    ("South America","Europe"):[(-15,-55),(10,-50),(30,-20),(50,10)],
    ("South America","North America"):[(-15,-55),(0,-75),(20,-90),(45,-93)],
    ("South America","Asia"):[(-15,-55),(-5,-80),(10,120),(34,108)],
    ("South America","Australia"):[(-15,-55),(-30,-100),(-30,150),(-25,134)],
    ("Australia","Asia"):[(-25,134),(-5,120),(10,100),(34,108)],
    ("Australia","Africa"):[(-25,134),(-20,80),(-10,50),(4,21)],
    ("Australia","Europe"):[(-25,134),(10,100),(30,60),(50,10)],
    ("Australia","North America"):[(-25,134),(-10,170),(20,-170),(45,-93)],
    ("Australia","South America"):[(-25,134),(-30,180),(-30,-100),(-15,-55)],
}

def get_wps(o, d):
    k = (o, d)
    if k in ROUTE_WP: return ROUTE_WP[k]
    rk = (d, o)
    if rk in ROUTE_WP: return list(reversed(ROUTE_WP[rk]))
    oc, dc = REGION_CENTRE[o], REGION_CENTRE[d]
    return [oc, ((oc[0]+dc[0])/2, (oc[1]+dc[1])/2), dc]

# ══════════════════════════════════════════════════════════════
# 3. SPECIES INFO
# ══════════════════════════════════════════════════════════════
SPECIES_INFO = {
    "Crane":   {"desc": "Cranes undertake some of the longest migrations on Earth, crossing mountain ranges and vast wetlands with powerful, sustained flight.",
                "size": "100–176 cm wingspan", "diet": "Omnivore — grains, insects, vertebrates", "lifespan": "20–30 years"},
    "Eagle":   {"desc": "Eagles use thermal updrafts to glide thousands of kilometres, riding ridge lines and avoiding open-ocean crossings.",
                "size": "Up to 244 cm wingspan", "diet": "Carnivore — fish, mammals, birds", "lifespan": "20–40 years"},
    "Goose":   {"desc": "Geese migrate in iconic V-formations that cut aerodynamic drag, enabling non-stop flights across continents.",
                "size": "130–185 cm wingspan", "diet": "Herbivore — grasses, grains, aquatic plants", "lifespan": "10–25 years"},
    "Hawk":    {"desc": "Hawks exploit thermals and ridge lift with precision, often forming spectacular kettles at migration bottlenecks.",
                "size": "55–141 cm wingspan", "diet": "Carnivore — rodents, lizards, small birds", "lifespan": "12–20 years"},
    "Stork":   {"desc": "Storks are master soaring migrants that funnel through narrow land bridges like Gibraltar and the Bosphorus.",
                "size": "155–215 cm wingspan", "diet": "Carnivore — fish, frogs, insects, small mammals", "lifespan": "20–35 years"},
    "Swallow": {"desc": "Swallows are aerial insectivores capable of extraordinary sustained flight, covering up to 10,000 km each migration.",
                "size": "28–48 cm wingspan", "diet": "Insectivore — flying insects caught mid-air", "lifespan": "4–8 years"},
    "Warbler": {"desc": "Warblers make some of the most impressive transoceanic flights — Blackpoll Warblers cross the Atlantic non-stop for 88 hours.",
                "size": "16–26 cm wingspan", "diet": "Insectivore — insects, berries during migration", "lifespan": "5–10 years"},
}
SPECIES_LABEL = {
    "Crane":"Crane", "Eagle":"Eagle", "Goose":"Goose",
    "Hawk":"Hawk", "Stork":"Stork", "Swallow":"Swallow", "Warbler":"Warbler"
}

# ══════════════════════════════════════════════════════════════
# 4. COLOUR SYSTEM
# ══════════════════════════════════════════════════════════════
C = {
    "bg":      "#f0f4f8",
    "card":    "#ffffff",
    "card2":   "#f8fafc",
    "card3":   "#eef2f7",
    "accent":  "#0369a1",
    "accent2": "#6d28d9",
    "accent3": "#d97706",
    "success": "#059669",
    "danger":  "#dc2626",
    "warn":    "#ea580c",
    "text":    "#1e293b",
    "muted":   "#64748b",
    "border":  "#cbd5e1",
}

PALETTE = ["#10b981","#00d4ff","#7c3aed","#f59e0b","#ef4444",
           "#06b6d4","#a78bfa","#34d399","#fbbf24","#f87171","#fb923c","#e879f9"]

BL = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=C["text"], family="Inter,sans-serif", size=12),
    margin=dict(t=48, b=36, l=40, r=24),
    colorway=PALETTE,
    xaxis=dict(gridcolor="#e2e8f0", zerolinecolor="#e2e8f0"),
    yaxis=dict(gridcolor="#e2e8f0", zerolinecolor="#e2e8f0"),
)

# ══════════════════════════════════════════════════════════════
# 5. CHART HELPERS — no play/reset, all instant
# ══════════════════════════════════════════════════════════════
def bar(cats, vals, colors=None, title="", orient="v",
        xtitle="", ytitle="", height=320, texts=None):
    n = len(cats)
    if not colors: colors = [C["accent"]] * n
    if not texts:
        texts = [f"{v:.1f}%" if isinstance(v, float) else str(v) for v in vals]
    if orient == "h":
        tr = go.Bar(x=vals, y=cats, orientation="h",
                    marker_color=colors, text=texts,
                    textposition="outside", textfont=dict(color=C["text"]))
    else:
        tr = go.Bar(x=cats, y=vals, marker_color=colors,
                    text=texts, textposition="outside",
                    textfont=dict(color=C["text"]))
    fig = go.Figure(data=[tr])
    fig.update_layout(**BL, title=title, height=height,
                      xaxis_title=xtitle, yaxis_title=ytitle)
    if orient == "h":
        fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig


def pie(labels, vals, title="", height=300, hole=0.55):
    fig = go.Figure(go.Pie(
        labels=labels, values=vals, hole=hole,
        marker=dict(colors=PALETTE[:len(labels)]),
        textfont=dict(color="#fff"),
        textinfo="percent+label",
        hoverinfo="label+percent",
    ))
    fig.update_layout(**BL, title=title, height=height,
                      legend=dict(font=dict(color=C["text"])))
    return fig


def graph(figure, cls="fly-in", **kw):
    """Wrap a Graph in a div with fly-in class for scroll animation."""
    return html.Div(
        dcc.Graph(figure=figure, config={"displayModeBar": False}, **kw),
        className=cls
    )

# ══════════════════════════════════════════════════════════════
# 6. UI HELPERS
# ══════════════════════════════════════════════════════════════
def kpi(label, value, sub="", color=C["accent"]):
    return dbc.Card(dbc.CardBody([
        html.P(label, style={"color": C["muted"], "fontSize": "0.68rem",
                             "textTransform": "uppercase", "letterSpacing": "0.1em",
                             "marginBottom": "4px"}),
        html.H3(str(value), style={"color": color, "fontWeight": "800",
                                    "margin": "0", "fontSize": "1.45rem"}),
        html.P(sub, style={"color": C["muted"], "fontSize": "0.68rem", "marginBottom": "0"}),
    ]), className="fly-in",
       style={"background": C["card2"], "border": f"1px solid {C['border']}",
              "borderRadius": "12px", "height": "100%"})


def sec(icon, title):
    return html.Div(
        [html.Span(title, style={"fontWeight": "700"})],
        style={"color": C["accent"], "fontSize": "1rem",
               "borderBottom": f"1px solid {C['border']}",
               "paddingBottom": "8px", "marginBottom": "18px"}
    )


def badge(text, color=C["accent2"]):
    return html.Span(text, style={
        "background": color + "28", "color": color,
        "border": f"1px solid {color}44", "borderRadius": "20px",
        "padding": "3px 10px", "fontSize": "0.72rem", "fontWeight": "600",
        "marginRight": "6px", "marginBottom": "4px", "display": "inline-block"
    })


def irow(label, val):
    return html.Tr([
        html.Td(label, style={"color": C["muted"], "fontSize": "0.77rem",
                               "padding": "5px 12px", "whiteSpace": "nowrap",
                               "fontWeight": "600"}),
        html.Td(str(val), style={"color": C["text"], "fontSize": "0.82rem",
                                  "padding": "5px 12px"}),
    ])


def icard(icon, title, value, sub, color):
    return html.Div([
        html.Div([
            html.P(title, style={"color": C["muted"], "fontSize": "0.68rem", "margin": "0",
                                  "textTransform": "uppercase", "letterSpacing": "0.08em"}),
            html.P(value, style={"color": color, "fontWeight": "800",
                                  "fontSize": "1.05rem", "margin": "0"}),
            html.P(sub, style={"color": C["muted"], "fontSize": "0.68rem", "margin": "0"}),
        ]),
    ], className="fly-in",
       style={"display": "flex", "alignItems": "center",
              "background": color + "12", "border": f"1px solid {color}33",
              "borderRadius": "10px", "padding": "12px 14px"})


def dds():
    return {"background": C["card2"], "color": C["text"], "fontSize": "0.82rem"}

# ══════════════════════════════════════════════════════════════
# 7. BIRD IMAGE FETCHER
# ══════════════════════════════════════════════════════════════
def fetch_img(bird_name: str, species: str) -> str | None:
    for name in [bird_name, f"{bird_name} bird", species + " bird"]:
        try:
            r = requests.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{name.replace(' ', '_')}",
                timeout=6, headers={"User-Agent": "BirdMigrationApp/4.0"}
            )
            if r.status_code == 200:
                d = r.json()
                if "thumbnail" in d:
                    return re.sub(r"/\d+px-", "/500px-", d["thumbnail"]["source"])
        except Exception:
            pass
    try:
        r = requests.get(
            "https://api.inaturalist.org/v1/taxa",
            params={"q": bird_name, "rank": "species", "per_page": 1}, timeout=6
        )
        if r.status_code == 200:
            res = r.json().get("results", [])
            if res:
                ph = res[0].get("default_photo") or {}
                u = ph.get("medium_url") or ph.get("url", "")
                if u:
                    return u
    except Exception:
        pass
    return None


def bird_img_el(bird_name: str, species: str):
    url = fetch_img(bird_name, species)
    label = SPECIES_LABEL.get(species, species)
    if url:
        return html.Div([
            html.Img(src=url, style={
                "width": "100%", "maxHeight": "220px", "objectFit": "cover",
                "borderRadius": "10px", "boxShadow": "0 4px 20px rgba(0,0,0,0.15)"
            }),
            html.P(bird_name, style={"color": C["muted"], "fontSize": "0.7rem",
                                      "textAlign": "center", "marginTop": "4px"}),
        ])
    return html.Div([
        html.P(label, style={"fontSize": "1.5rem", "fontWeight": "700", "color": C["accent"]}),
        html.P(bird_name, style={"color": C["muted"], "fontSize": "0.7rem", "marginTop": "4px"}),
    ], style={"textAlign": "center", "background": C["card3"], "borderRadius": "10px",
              "padding": "20px", "display": "flex", "flexDirection": "column",
              "alignItems": "center"})

# ══════════════════════════════════════════════════════════════
# 8. BIRD DETAIL PANEL
# ══════════════════════════════════════════════════════════════
def bird_detail_panel(bird_name: str):
    bdf = df[df["Bird_Name"] == bird_name]
    if bdf.empty:
        return dbc.Alert(f"No data found for {bird_name}", color="warning")

    species  = bdf["Species"].iloc[0]
    total    = len(bdf)
    succ_n   = int(bdf["Success"].sum())
    fail_n   = total - succ_n
    succ_pct = succ_n / total * 100
    fail_pct = 100 - succ_pct

    sdf = bdf[bdf["Success"]]
    fdf = bdf[~bdf["Success"]]

    top_reason   = sdf["Migration_Reason"].mode().iloc[0]  if len(sdf) else "N/A"
    top_weath_s  = sdf["Weather_Condition"].mode().iloc[0] if len(sdf) else "N/A"
    avg_speed    = sdf["Average_Speed_kmph"].mean()        if len(sdf) else 0
    avg_alt      = sdf["Max_Altitude_m"].mean()            if len(sdf) else 0
    flock_pct    = (sdf["Migrated_in_Flock"] == "Yes").mean() * 100 if len(sdf) else 0
    top_food     = sdf["Food_Supply_Level"].mode().iloc[0] if len(sdf) else "N/A"
    fail_reasons = fdf["Interrupted_Reason"].value_counts() if len(fdf) else pd.Series(dtype=int)
    primary_fail = fail_reasons.index[0]                   if len(fail_reasons) else "N/A"
    top_weath_f  = fdf["Weather_Condition"].value_counts().index[0] if len(fdf) else "N/A"
    regions      = sorted(bdf["Region"].dropna().unique())
    habitats     = sorted(bdf["Habitat"].dropna().unique())
    sp_info      = SPECIES_INFO.get(species, {})

    fig_out = pie(["Successful", "Failed"], [succ_pct, fail_pct],
                  title="Migration Outcome", height=200, hole=0.62)
    fig_out.update_layout(
        margin=dict(t=30, b=10, l=10, r=10),
        annotations=[dict(text=f"{succ_pct:.0f}%", x=0.5, y=0.5,
                          font=dict(size=18, color=C["success"]), showarrow=False)]
    )

    fail_badges = (
        [badge(f"{r}: {c / fail_n * 100:.0f}% of failures", C["danger"])
         for r, c in fail_reasons.items()]
        if fail_n else [html.Span("No failures recorded", style={"color": C["success"]})]
    )

    return dbc.Card([
        dbc.CardBody([
            # Header row: info + image
            dbc.Row([
                dbc.Col([
                    html.H5(bird_name,
                            style={"color": C["accent"], "fontWeight": "800", "marginBottom": "4px"}),
                    html.Div([
                        badge(f"Species: {species}", C["accent2"]),
                        *[badge(h, C["accent3"]) for h in habitats],
                    ], style={"marginBottom": "8px"}),
                    html.P(sp_info.get("desc", ""),
                           style={"color": C["muted"], "fontSize": "0.8rem",
                                  "lineHeight": "1.55", "marginBottom": "8px"}),
                    html.Table([
                        irow("Size",     sp_info.get("size",     "N/A")),
                        irow("Diet",     sp_info.get("diet",     "N/A")),
                        irow("Lifespan", sp_info.get("lifespan", "N/A")),
                        irow("Regions",  ", ".join(regions)),
                        irow("Habitats", ", ".join(habitats)),
                    ], style={"background": C["card3"], "borderRadius": "8px",
                              "width": "100%", "marginBottom": "10px"}),
                ], md=7),
                dbc.Col(bird_img_el(bird_name, species), md=5,
                        style={"paddingLeft": "12px"}),
            ], className="mb-3"),

            # KPIs
            dbc.Row([
                dbc.Col(kpi("Success Rate", f"{succ_pct:.1f}%", "", C["success"]), md=4),
                dbc.Col(kpi("Failure Rate", f"{fail_pct:.1f}%", "", C["danger"]),  md=4),
                dbc.Col(dcc.Graph(figure=fig_out, config={"displayModeBar": False},
                                  style={"height": "200px"}), md=4),
            ], className="mb-3 g-2"),

            # Success / Failure blocks
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H6("Why Migrations Succeed",
                                style={"color": C["success"], "fontWeight": "700",
                                       "marginBottom": "8px"}),
                        html.Table([
                            irow("Primary Reason",    top_reason),
                            irow("Best Weather",      top_weath_s),
                            irow("Avg Speed",         f"{avg_speed:.1f} km/h"),
                            irow("Avg Max Altitude",  f"{avg_alt:.0f} m"),
                            irow("Migrate in Flock",  f"{flock_pct:.0f}%"),
                            irow("Food Supply",       top_food),
                        ], style={"background": C["card3"], "borderRadius": "8px", "width": "100%"}),
                    ], style={"background": C["success"] + "0e",
                              "border": f"1px solid {C['success']}33",
                              "borderRadius": "10px", "padding": "12px"}),
                ], md=6),
                dbc.Col([
                    html.Div([
                        html.H6("Why Migrations Fail",
                                style={"color": C["danger"], "fontWeight": "700",
                                       "marginBottom": "8px"}),
                        html.Table([
                            irow("Primary Cause",   primary_fail),
                            irow("Common Weather",  top_weath_f),
                        ], style={"background": C["card3"], "borderRadius": "8px",
                                  "width": "100%", "marginBottom": "8px"}),
                        html.P("Cause breakdown:",
                               style={"color": C["muted"], "fontSize": "0.7rem",
                                      "marginBottom": "4px"}),
                        html.Div(fail_badges),
                    ], style={"background": C["danger"] + "0e",
                              "border": f"1px solid {C['danger']}33",
                              "borderRadius": "10px", "padding": "12px"}),
                ], md=6),
            ], className="g-3"),
        ])
    ], style={"background": C["card2"], "border": f"1px solid {C['border']}",
              "borderRadius": "14px", "marginTop": "16px",
              "boxShadow": "0 4px 30px rgba(0,0,0,0.4)"})

# ══════════════════════════════════════════════════════════════
# 9. GLOBE BUILDER (realistic colours) — FIXED VERSION
# ══════════════════════════════════════════════════════════════
def build_globe(continents=None, country=None, species_sel=None, status="all"):
    """
    FIXED: Now properly handles the country parameter to filter by specific country
    """
    fdf = df.copy()
    if continents:    fdf = fdf[fdf["Region"].isin(continents if isinstance(continents, list) else [continents])]
    if country:       fdf = fdf[df["Country"] == country]  # FIX: Filter by country when specified
    if species_sel:   fdf = fdf[fdf["Species"].isin(species_sel)]
    if status == "success": fdf = fdf[fdf["Success"]]
    elif status == "failed": fdf = fdf[~fdf["Success"]]

    fig = go.Figure()

    # Route arcs (land-based)
    shown = set()
    for o in ALL_REGIONS:
        for d in ALL_REGIONS:
            if o == d: continue
            k = (o, d)
            if k in shown: continue
            shown.add(k)
            wps = get_wps(o, d)
            sr = fdf[fdf["Region"] == o]["Success"].mean() if len(fdf[fdf["Region"] == o]) else 0.5
            col = "#10b981" if sr >= 0.5 else "#ef4444"
            fig.add_trace(go.Scattergeo(
                lat=[p[0] for p in wps], lon=[p[1] for p in wps],
                mode="lines", line=dict(width=1.4, color=col),
                opacity=0.35, showlegend=False, hoverinfo="skip"
            ))

    # Country markers — FIXED: Only show countries in the filtered data's regions
    for country_name, coords in COUNTRY_COORDS.items():
        cont = ALL_COUNTRIES.get(country_name, "")
        
        # Filter by continent if specified
        if continents:
            cl = continents if isinstance(continents, list) else [continents]
            if cont not in cl:
                continue
        
        # FIX: Filter by country if specified — only show the selected country
        if country:
            if country_name != country:
                continue
            # For selected country, use filtered data (fdf)
            sub = fdf[fdf["Region"] == cont]
        else:
            # For all countries, use full filtered data
            sub = fdf[fdf["Region"] == cont]
        
        sr  = sub["Success"].mean() * 100 if len(sub) else 50
        col = "#10b981" if sr >= 50 else "#ef4444"
        n_birds = df[df["Region"] == cont]["Bird_Name"].nunique()
        
        fig.add_trace(go.Scattergeo(
            lat=[coords[0]], lon=[coords[1]],
            mode="markers",
            marker=dict(size=9, color=col, opacity=0.9,
                        line=dict(width=1, color="white")),
            name=country_name, showlegend=False,
            customdata=[[country_name, cont]],
            hovertemplate=(
                f"<b>{country_name}</b><br>"
                f"Continent: {cont}<br>"
                f"Migration Success: {sr:.1f}%<br>"
                f"Individual Birds: {n_birds}<extra></extra>"
            )
        ))

    # Legend traces
    fig.add_trace(go.Scattergeo(lat=[None], lon=[None], mode="markers",
                                 name="≥50% Success", marker=dict(color="#10b981", size=10)))
    fig.add_trace(go.Scattergeo(lat=[None], lon=[None], mode="markers",
                                 name="<50% Success", marker=dict(color="#ef4444", size=10)))

    fig.update_geos(
        projection_type="natural earth",
        showland=True,       landcolor="#c8dda7",
        showocean=True,      oceancolor="#aad4f5",
        showlakes=True,      lakecolor="#aad4f5",
        showrivers=True,     rivercolor="#7ec8e3",
        showcountries=True,  countrycolor="#8aaa6a",
        showcoastlines=True, coastlinecolor="#5a8a40",
        showframe=False,
        bgcolor="rgba(0,0,0,0)",
        lataxis_showgrid=True, lonaxis_showgrid=True,
        lataxis_gridcolor="rgba(0,0,0,0.06)",
        lonaxis_gridcolor="rgba(0,0,0,0.06)",
        showsubunits=True, subunitcolor="#9ab87a",
    )
    fig.update_layout(
        **BL, height=580,
        title="Click any country marker to explore birds in that region",
        geo=dict(bgcolor="rgba(0,0,0,0)"),
        legend=dict(orientation="h", y=0.01, x=0.5, xanchor="center",
                    bgcolor="rgba(255,255,255,0.8)", font=dict(color=C["text"]))
    )
    return fig

# ══════════════════════════════════════════════════════════════
# 10. TAB CONTENT BUILDERS
# ══════════════════════════════════════════════════════════════

# ── DASHBOARD ──────────────────────────────────────────────
def build_dashboard():
    total = len(df)
    succ  = int(df["Success"].sum())
    pct   = succ / total * 100

    # Overall pie
    fig_pie = pie(["Successful", "Failed"], [pct, 100 - pct],
                  title="Overall Migration Rate", height=280, hole=0.62)
    fig_pie.update_layout(
        annotations=[dict(text=f"{pct:.1f}%", x=0.5, y=0.5,
                          font=dict(size=22, color=C["success"], family="Inter"),
                          showarrow=False)]
    )

    # Species × Region grouped bar
    g = df.groupby(["Region", "Species"])["Success"].agg(["sum", "count"]).reset_index()
    g["pct"] = g["sum"] / g["count"] * 100
    fig_t5 = go.Figure()
    for i, sp in enumerate(ALL_SPECIES):
        sub = g[g["Species"] == sp]
        fig_t5.add_trace(go.Bar(
            name=sp, x=sub["Region"], y=sub["pct"].round(1),
            marker_color=PALETTE[i % len(PALETTE)],
            text=[f"{v:.1f}%" for v in sub["pct"]],
            textposition="outside", textfont=dict(color=C["text"], size=10),
        ))
    fig_t5.update_layout(
        **BL, barmode="group", height=400,
        title="Migration Success Rate by Species & Region (%)",
        yaxis_title="Success Rate (%)", yaxis_range=[0, 75],
        legend=dict(font=dict(color=C["text"]))
    )

    # Habitat pie
    hab = df.groupby("Habitat")["Success"].agg(["sum", "count"]).reset_index()
    hab["pct"] = hab["sum"] / hab["count"] * 100
    fig_hab = pie(hab["Habitat"].tolist(), hab["pct"].tolist(),
                  title="Success Rate by Habitat (%)", height=280)

    # Migration reason pie
    mr = df.groupby("Migration_Reason")["Success"].agg(["sum", "count"]).reset_index()
    mr["pct"] = mr["sum"] / mr["count"] * 100
    fig_mr = pie(mr["Migration_Reason"].tolist(), mr["pct"].tolist(),
                 title="Success Rate by Migration Reason (%)", height=280)

    best_region = df.groupby("Region")["Success"].mean().idxmax()
    best_sp     = df.groupby("Species")["Success"].mean().idxmax()
    best_sp_pct = df.groupby("Species")["Success"].mean().max() * 100

    return dbc.Container([
        dbc.Row([
            dbc.Col(kpi("Overall Success",  f"{pct:.1f}%",       "of all tracked migrations",   C["success"]), md=3),
            dbc.Col(kpi("Failure Rate",     f"{100 - pct:.1f}%", "of all tracked migrations",   C["danger"]),  md=3),
            dbc.Col(kpi("Species Tracked",  str(len(ALL_SPECIES)), "unique species",              C["accent2"]), md=3),
            dbc.Col(kpi("Regions Covered",  str(len(ALL_REGIONS)), "global continents",           C["accent"]),  md=3),
        ], className="mb-4 g-3"),
        dbc.Row([
            dbc.Col(icard("", "Best Performing Species", best_sp,
                          f"{best_sp_pct:.1f}% success rate",    C["success"]), md=4),
            dbc.Col(icard("", "Highest-Success Region",  best_region,
                          "most successful migrations",           C["accent"]),  md=4),
            dbc.Col(icard("", "Best Habitat",            "Wetland",
                          "52.3% success rate",                   C["accent3"]), md=4),
        ], className="mb-4 g-3"),
        dbc.Row([
            dbc.Col(graph(fig_t5),  md=8),
            dbc.Col(graph(fig_pie), md=4),
        ], className="mb-3 g-3"),
        dbc.Row([
            dbc.Col(graph(fig_hab), md=6),
            dbc.Col(graph(fig_mr),  md=6),
        ], className="g-3"),
    ], fluid=True, className="py-3")


# ── MAP TAB ────────────────────────────────────────────────
def build_map_tab():
    return dbc.Container([
        sec("", "Interactive Map — Click a Country to Explore Birds"),
        dbc.Row([
            dbc.Col([
                html.Label("Continent", style={"color": C["muted"], "fontSize": "0.75rem"}),
                dcc.Dropdown(id="map-continent",
                             options=[{"label": r, "value": r} for r in ALL_REGIONS],
                             multi=True, placeholder="All continents", style=dds()),
            ], md=3),
            dbc.Col([
                html.Label("Country", style={"color": C["muted"], "fontSize": "0.75rem"}),
                dcc.Dropdown(id="map-country", options=[],
                             placeholder="Select country…", style=dds()),
            ], md=3),
            dbc.Col([
                html.Label("Species filter", style={"color": C["muted"], "fontSize": "0.75rem"}),
                dcc.Dropdown(id="map-species",
                             options=[{"label": s, "value": s} for s in ALL_SPECIES],
                             multi=True, placeholder="All species", style=dds()),
            ], md=3),
            dbc.Col([
                html.Label("Status", style={"color": C["muted"], "fontSize": "0.75rem"}),
                dcc.RadioItems(id="map-status",
                               options=[{"label": " All",     "value": "all"},
                                        {"label": " Success", "value": "success"},
                                        {"label": " Failed",  "value": "failed"}],
                               value="all", inline=True,
                               style={"color": C["text"], "marginTop": "8px"},
                               inputStyle={"marginRight": "4px", "marginLeft": "10px"}),
            ], md=3),
        ], className="mb-3 g-2"),

        dbc.Row([
            # Globe
            dbc.Col([
                dcc.Graph(id="globe-fig", style={"height": "580px"},
                          config={"displayModeBar": True}),
            ], md=8),
            # Side panel
            dbc.Col([
                html.Div(id="map-side-panel",
                         children=html.Div([
                             html.P("Click a country marker on the map to see all "
                                    "individual birds tracked in that region.",
                                    style={"color": C["muted"], "marginTop": "12px",
                                           "fontSize": "0.85rem", "textAlign": "center"}),
                         ], style={
                             "textAlign": "center", "padding": "40px 20px",
                             "background": C["card2"], "borderRadius": "12px",
                             "border": f"1px solid {C['border']}", "height": "580px",
                             "display": "flex", "flexDirection": "column",
                             "alignItems": "center", "justifyContent": "center"
                         })),
            ], md=4),
        ], className="mb-3 g-3"),

        # Bird detail panel — shown below the map after clicking a bird
        html.Div(id="map-bird-detail"),

        html.P("Green = ≥50% success  |  Red = <50%  |  Click any country dot for details",
               style={"color": C["muted"], "fontSize": "0.72rem",
                      "textAlign": "center", "marginTop": "4px"}),
    ], fluid=True, className="py-3")


# ── MONTHS TAB ─────────────────────────────────────────────
def build_months():
    avail = [m for m in MONTH_ORDER if m in df["Migration_Start_Month"].values]
    return dbc.Container([
        sec("", "Monthly Migration Patterns"),
        dbc.Row([
            dbc.Col([
                html.Label("Month(s)", style={"color": C["muted"], "fontSize": "0.75rem"}),
                dcc.Dropdown(id="month-sel",
                             options=[{"label": m, "value": m} for m in avail],
                             multi=True, placeholder="All months", style=dds()),
            ], md=3),
            dbc.Col([
                html.Label("Continent", style={"color": C["muted"], "fontSize": "0.75rem"}),
                dcc.Dropdown(id="month-continent",
                             options=[{"label": r, "value": r} for r in ALL_REGIONS],
                             multi=True, placeholder="All continents", style=dds()),
            ], md=3),
            dbc.Col([
                html.Label("Country", style={"color": C["muted"], "fontSize": "0.75rem"}),
                dcc.Dropdown(id="month-country", options=[],
                             placeholder="Select country…", style=dds()),
            ], md=3),
            dbc.Col([
                html.Label("Status", style={"color": C["muted"], "fontSize": "0.75rem"}),
                dcc.RadioItems(id="month-status",
                               options=[{"label": " All",     "value": "all"},
                                        {"label": " Success", "value": "success"},
                                        {"label": " Failed",  "value": "failed"}],
                               value="all", inline=True,
                               style={"color": C["text"], "marginTop": "8px"},
                               inputStyle={"marginRight": "4px", "marginLeft": "10px"}),
            ], md=3),
        ], className="mb-3 g-2"),
        html.Div(id="month-results"),
    ], fluid=True, className="py-3")


# ── WEATHER TAB ────────────────────────────────────────────
def build_weather():
    wdf = df[df["Migration_Success"] == "Failed"]
    wc  = wdf["Weather_Condition"].value_counts().reset_index()
    wc.columns = ["Condition", "Count"]
    wc["pct"]  = wc["Count"] / wc["Count"].sum() * 100
    WCOL = {"Foggy": C["muted"], "Stormy": C["danger"], "Windy": C["accent3"],
             "Clear": C["success"], "Rainy": "#3b82f6"}

    fig1 = bar(wc["Condition"].tolist(), wc["pct"].round(1).tolist(),
               colors=[WCOL.get(c, C["accent"]) for c in wc["Condition"]],
               title="Share of Weather Conditions in Failures (%)",
               ytitle="Share (%)", height=300)

    sp_fail = (df[df["Migration_Success"] == "Failed"].groupby("Species").size() /
               df.groupby("Species").size() * 100).reset_index(name="pct")
    fig2 = pie(sp_fail["Species"].tolist(), sp_fail["pct"].round(1).tolist(),
               title="Failure Rate by Species (%)", height=300)

    fig3 = bar(wc["Condition"].tolist(), wc["pct"].round(1).tolist(),
               colors=[WCOL.get(c, C["accent"]) for c in wc["Condition"]],
               orient="h", title="Weather Failures by Condition (%)",
               xtitle="Share (%)", height=280)

    worst_cond = wc.iloc[0]["Condition"]
    best_cond  = wc.iloc[-1]["Condition"]

    return dbc.Container([
        sec("", "Weather & Failure Analysis"),
        dbc.Row([
            dbc.Col(icard("", "Most Dangerous Weather", worst_cond,
                          f"{wc.iloc[0]['pct']:.1f}% of failures",  C["danger"]),  md=4),
            dbc.Col(icard("", "Least Dangerous",        best_cond,
                          f"{wc.iloc[-1]['pct']:.1f}% of failures", C["success"]), md=4),
            dbc.Col(icard("", "Overall Failure Rate",
                          f"{100 - df['Success'].mean() * 100:.1f}%",
                          "across all migrations", C["accent"]), md=4),
        ], className="mb-4 g-3"),
        dbc.Row([
            dbc.Col(graph(fig1), md=6),
            dbc.Col(graph(fig2), md=6),
        ], className="mb-3 g-3"),
        dbc.Row([
            dbc.Col(graph(fig3), md=8),
        ], className="g-3"),
    ], fluid=True, className="py-3")


# ── INTERRUPTIONS TAB ──────────────────────────────────────
def build_interruptions():
    return dbc.Container([
        sec("", "Interruptions & Recovery"),
        dbc.Row([
            dbc.Col([
                html.Label("Select Interruption Reason",
                           style={"color": C["muted"], "fontSize": "0.78rem"}),
                dcc.Dropdown(id="int-reason",
                             options=[{"label": r, "value": r} for r in ALL_REASONS],
                             placeholder="Choose a reason to explore…", style=dds()),
            ], md=4),
            dbc.Col([
                html.Label("Continent", style={"color": C["muted"], "fontSize": "0.75rem"}),
                dcc.Dropdown(id="int-continent",
                             options=[{"label": r, "value": r} for r in ALL_REGIONS],
                             multi=True, placeholder="All continents", style=dds()),
            ], md=4),
            dbc.Col([
                html.Label("Country", style={"color": C["muted"], "fontSize": "0.75rem"}),
                dcc.Dropdown(id="int-country", options=[], placeholder="Select country…",
                             style=dds()),
            ], md=4),
        ], className="mb-4 g-2"),
        html.Div(id="int-results", children=html.Div([
            html.P("Select an interruption reason above to explore affected birds and "
                   "average recovery days by region.",
                   style={"color": C["muted"], "marginTop": "10px", "fontSize": "0.85rem"}),
        ], style={"textAlign": "center", "padding": "40px", "background": C["card2"],
                  "borderRadius": "12px", "border": f"1px solid {C['border']}"})),
    ], fluid=True, className="py-3")


# ── MIGRATION STATUS TAB ───────────────────────────────────
def build_migration_status():
    return dbc.Container([
        sec("", "Migration Status — Route Analysis by Species"),
        html.P("Select an origin and destination (continent → country optional) to see "
               "the most and least successful routes per species.",
               style={"color": C["muted"], "marginBottom": "20px", "fontSize": "0.85rem"}),
        dbc.Row([
            dbc.Col([
                html.Label("Origin Continent", style={"color": C["muted"], "fontSize": "0.75rem"}),
                dcc.Dropdown(id="ms-orig-cont",
                             options=[{"label": r, "value": r} for r in ALL_REGIONS],
                             placeholder="Origin continent…", style=dds()),
            ], md=3),
            dbc.Col([
                html.Label("Origin Country", style={"color": C["muted"], "fontSize": "0.75rem"}),
                dcc.Dropdown(id="ms-orig-country", options=[],
                             placeholder="Origin country…", style=dds()),
            ], md=3),
            dbc.Col([
                html.Label("Destination Continent", style={"color": C["muted"], "fontSize": "0.75rem"}),
                dcc.Dropdown(id="ms-dest-cont",
                             options=[{"label": r, "value": r} for r in ALL_REGIONS],
                             placeholder="Dest continent…", style=dds()),
            ], md=3),
            dbc.Col([
                html.Label("Destination Country", style={"color": C["muted"], "fontSize": "0.75rem"}),
                dcc.Dropdown(id="ms-dest-country", options=[],
                             placeholder="Dest country…", style=dds()),
            ], md=3),
        ], className="mb-3 g-2"),
        dbc.Row([
            dbc.Col([
                html.Label("Species (optional)", style={"color": C["muted"], "fontSize": "0.75rem"}),
                dcc.Dropdown(id="ms-species",
                             options=[{"label": s, "value": s} for s in ALL_SPECIES],
                             multi=True, placeholder="All species", style=dds()),
            ], md=5),
            dbc.Col([
                dbc.Button("Analyse Migration Status", id="ms-btn", n_clicks=0,
                           style={"background": C["accent"], "border": "none",
                                  "color": "#000", "fontWeight": "700",
                                  "borderRadius": "8px", "padding": "10px 24px",
                                  "marginTop": "22px"}),
            ], md=4),
        ], className="mb-4 g-2"),
        html.Div(id="ms-results"),
    ], fluid=True, className="py-3")

# ══════════════════════════════════════════════════════════════
# 11. APP
# ══════════════════════════════════════════════════════════════
app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap",
    ],
    suppress_callback_exceptions=True,
    title="Bird Migration Intelligence",
)
server = app.server  # gunicorn entry point

TABS = [
    ("dashboard",  "Dashboard"),
    ("map",        "Map & Explore"),
    ("months",     "Monthly Patterns"),
    ("weather",    "Weather & Failures"),
    ("interrupt",  "Interruptions"),
    ("migration",  "Migration Status"),
]

def nav_btn(key, label):
    return html.Button(
        label, id=f"tab-{key}",
        style={"background": "none", "border": "none",
               "borderBottom": "2px solid transparent",
               "color": "#94a3b8", "padding": "10px 16px",
               "cursor": "pointer", "fontSize": "0.8rem",
               "fontWeight": "600", "transition": "all 0.18s"})

app.layout = html.Div([
    html.Header(
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Bird Migration Intelligence",
                            style={"color": C["text"], "fontSize": "1.7rem",
                                   "fontWeight": "800", "margin": "0"}),
                    html.P("Migration Tracking & Analysis Platform",
                           style={"color": C["muted"], "fontSize": "0.75rem",
                                  "margin": "0", "marginTop": "2px",
                                  "letterSpacing": "0.05em"}),
                ], md=6),
                dbc.Col([
                    html.P(f"Live  Updated {datetime.now().strftime('%d %b %Y')}",
                           style={"color": C["success"], "fontSize": "0.72rem",
                                  "fontWeight": "600", "textAlign": "right",
                                  "marginBottom": "0"}),
                ], md=6, style={"display": "flex", "justifyContent": "flex-end",
                               "alignItems": "center"}),
            ], style={"paddingTop": "16px", "paddingBottom": "16px"}),
        ], fluid=True),
        style={"background": C["card"], "borderBottom": f"1px solid {C['border']}",
               "marginBottom": "24px"}
    ),
    
    # Navigation tabs
    html.Div(
        [nav_btn(key, label) for key, label in TABS],
        style={"display": "flex", "borderBottom": f"1px solid {C['border']}",
               "paddingLeft": "16px", "marginBottom": "24px"}
    ),

    # Page content
    html.Div(id="page-content", style={"minHeight": "600px"}),

    # Stores for page switching
    dcc.Store(id="active-tab", data="dashboard"),
])

# ══════════════════════════════════════════════════════════════
# 12. CALLBACKS
# ══════════════════════════════════════════════════════════════

# Tab routing
@app.callback(
    Output("page-content", "children"),
    Output("active-tab", "data"),
    *[Input(f"tab-{key}", "n_clicks") for key, _ in TABS],
    State("active-tab", "data"),
    prevent_initial_call=False,
)
def route(*args):
    ctx = callback_context
    state = args[-1]
    if not ctx.triggered:
        return build_dashboard(), "dashboard"
    trig_id = ctx.triggered[0]["prop_id"].split(".")[0]
    for key, _ in TABS:
        if trig_id == f"tab-{key}":
            if key == "dashboard":  return build_dashboard(), key
            elif key == "map":      return build_map_tab(), key
            elif key == "months":   return build_months(), key
            elif key == "weather":  return build_weather(), key
            elif key == "interrupt":return build_interruptions(), key
            elif key == "migration":return build_migration_status(), key
    return build_dashboard(), "dashboard"


# Globe — FIXED CALLBACK
@app.callback(
    Output("globe-fig", "figure"),
    Input("map-continent", "value"),
    Input("map-country",   "value"),
    Input("map-species",   "value"),
    Input("map-status",    "value"),
)
def update_globe(continents, country, species_sel, status):
    """FIXED: Now passes the country parameter to build_globe"""
    return build_globe(continents, country, species_sel, status)


# Continent → Country cascade (map)
@app.callback(
    Output("map-country", "options"),
    Input("map-continent", "value"),
    prevent_initial_call=False,
)
def map_countries(cont):
    if not cont:
        return [{"label": c, "value": c} for c in sorted(ALL_COUNTRIES.keys())]
    cl = cont if isinstance(cont, list) else [cont]
    cs = []
    for c in cl:
        cs.extend(CONTINENT_COUNTRIES.get(c, []))
    return [{"label": c, "value": c} for c in sorted(set(cs))]


# MAP CLICK → show all individual birds in side panel
@app.callback(
    Output("map-side-panel", "children"),
    Input("globe-fig", "clickData"),
    prevent_initial_call=True,
)
def map_click(click_data):
    if not click_data:
        return html.Div()
    pt = click_data["points"][0]
    cd = pt.get("customdata")
    if not cd:
        return html.Div("Click a country marker.", style={"color": C["muted"]})

    country_name, continent = cd[0], cd[1]
    region_df  = df[df["Region"] == continent]

    # All individual birds in this region with their species and success rates
    bird_stats = (
        region_df.groupby(["Bird_Name", "Species"])["Success"]
        .agg(["sum", "count"]).reset_index()
    )
    bird_stats.columns = ["Bird_Name", "Species", "Succ", "Total"]
    bird_stats["Rate"] = (bird_stats["Succ"] / bird_stats["Total"] * 100).round(1)
    bird_stats = bird_stats.sort_values("Rate", ascending=False)

    # Species success overview bars
    sp_rates = region_df.groupby("Species")["Success"].mean() * 100

    bird_items = []
    current_species = None
    for _, row in bird_stats.iterrows():
        # Species header
        if row["Species"] != current_species:
            current_species = row["Species"]
            bird_items.append(
                html.Div([
                    html.Span(f"{current_species}",
                              style={"color": C["accent2"], "fontWeight": "700",
                                     "fontSize": "0.8rem"}),
                    html.Span(f" — {sp_rates.get(current_species, 0):.1f}% success",
                              style={"color": C["muted"], "fontSize": "0.7rem"}),
                ], style={"background": C["card3"], "borderRadius": "6px",
                          "padding": "5px 8px", "marginTop": "8px",
                          "marginBottom": "3px",
                          "borderLeft": f"3px solid {C['accent2']}"})
            )
        # Individual bird button
        col = C["success"] if row["Rate"] >= 50 else C["danger"]
        bird_items.append(
            html.Div([
                html.Div([
                    html.Span(row["Bird_Name"],
                              style={"color": C["text"], "fontSize": "0.78rem",
                                     "fontWeight": "500"}),
                    html.Span(f" {row['Rate']:.0f}%",
                              style={"color": col, "fontSize": "0.72rem",
                                     "fontWeight": "700", "marginLeft": "auto"}),
                ], style={"display": "flex", "justifyContent": "space-between",
                          "alignItems": "center"}),
                html.Div(style={"width": f"{min(row['Rate'], 100):.0f}%",
                                "height": "3px", "background": col,
                                "borderRadius": "2px", "marginTop": "3px"}),
            ], id={"type": "bird-btn", "index": row["Bird_Name"]},
               style={"padding": "6px 8px", "cursor": "pointer",
                      "borderRadius": "6px", "marginBottom": "2px",
                      "background": C["card3"],
                      "border": f"1px solid {C['border']}",
                      "transition": "background 0.15s"},
               n_clicks=0)
        )

    return html.Div([
        html.H6(f"{country_name}",
                style={"color": C["accent"], "fontWeight": "800", "marginBottom": "2px"}),
        html.P(f"Continent: {continent}  ·  {len(bird_stats)} birds tracked",
               style={"color": C["muted"], "fontSize": "0.72rem", "marginBottom": "10px"}),
        html.P("Click any bird for full migration details:",
               style={"color": C["muted"], "fontSize": "0.7rem", "marginBottom": "6px",
                      "textTransform": "uppercase", "letterSpacing": "0.08em"}),
        html.Div(bird_items,
                 style={"overflowY": "auto", "maxHeight": "460px",
                        "paddingRight": "4px"}),
    ], style={"background": C["card2"], "borderRadius": "12px",
              "padding": "14px", "border": f"1px solid {C['border']}",
              "height": "580px", "display": "flex", "flexDirection": "column"})


# Individual bird click → detail panel below map
# bird-btn callback defined later with pattern matching


# Month analysis
@app.callback(
    Output("month-country", "options"),
    Input("month-continent", "value"),
    prevent_initial_call=False,
)
def month_countries(cont):
    if not cont:
        return [{"label": c, "value": c} for c in sorted(ALL_COUNTRIES.keys())]
    cl = cont if isinstance(cont, list) else [cont]
    cs = []
    for c in cl: cs.extend(CONTINENT_COUNTRIES.get(c, []))
    return [{"label": c, "value": c} for c in sorted(set(cs))]


@app.callback(
    Output("month-results", "children"),
    Input("month-sel",       "value"),
    Input("month-continent", "value"),
    Input("month-country",   "value"),
    Input("month-status",    "value"),
)
def update_months(months, continents, country, status):
    fdf = df.copy()
    if months:     fdf = fdf[fdf["Migration_Start_Month"].isin(months)]
    if continents:
        cl = continents if isinstance(continents, list) else [continents]
        fdf = fdf[fdf["Region"].isin(cl)]
    if status == "success": fdf = fdf[fdf["Success"]]
    elif status == "failed": fdf = fdf[~fdf["Success"]]
    if len(fdf) == 0:
        return dbc.Alert("No data for selected filters.", color="warning")

    sp_pct  = fdf.groupby("Species")["Success"].mean() * 100
    fig_sp  = pie(sp_pct.index.tolist(), sp_pct.values.round(1).tolist(),
                  title="Success Rate by Species (%)", height=300)

    hab = fdf.groupby("Habitat")["Success"].mean() * 100
    hab_colors = [C["success"] if v >= 50 else C["danger"] for v in hab.values]
    fig_hab = bar(hab.index.tolist(), hab.values.round(1).tolist(),
                  colors=hab_colors,
                  texts=[f"{v:.1f}%" for v in hab.values],
                  title="Success Rate by Habitat (%)", ytitle="Success Rate (%)", height=280)
    fig_hab.add_hline(y=50, line_dash="dash", line_color=C["muted"], opacity=0.5)

    reg_pct = fdf.groupby("Region")["Success"].mean() * 100
    fig_reg = pie(reg_pct.index.tolist(), reg_pct.values.round(1).tolist(),
                  title="Success Rate by Region (%)", height=300)

    monthly = (fdf.groupby("Migration_Start_Month")["Success"].mean() * 100
               ).reset_index()
    monthly.columns = ["Month", "Rate"]
    monthly = monthly.sort_values("Month")
    month_cols = [C["success"] if v >= 50 else C["danger"] for v in monthly["Rate"]]
    fig_month = bar(monthly["Month"].astype(str).tolist(), monthly["Rate"].round(1).tolist(),
                    colors=month_cols, texts=[f"{v:.1f}%" for v in monthly["Rate"]],
                    title="Monthly Migration Success Rate (%)",
                    ytitle="Success Rate (%)", height=280)
    fig_month.add_hline(y=50, line_dash="dash", line_color=C["muted"], opacity=0.5)

    mr = (fdf.groupby(["Migration_Reason", "Species"])["Success"].mean().reset_index())
    mr["pct"] = mr["Success"] * 100
    fig_sun = px.sunburst(mr, path=["Migration_Reason", "Species"], values="pct",
                           color="pct", color_continuous_scale="RdYlGn",
                           title="Success Rate: Migration Reason → Species (%)")
    fig_sun.update_layout(**BL, height=360)
    fig_sun.update_coloraxes(showscale=False)

    return html.Div([
        dbc.Row([
            dbc.Col(graph(fig_sp),  md=4),
            dbc.Col(graph(fig_reg), md=4),
            dbc.Col(graph(fig_hab), md=4),
        ], className="mb-3 g-3"),
        dbc.Row([
            dbc.Col(graph(fig_month), md=6),
            dbc.Col(graph(fig_sun),   md=6),
        ], className="g-3"),
    ])


# Interruptions
@app.callback(
    Output("int-country", "options"),
    Input("int-continent", "value"),
    prevent_initial_call=False,
)
def int_countries(cont):
    if not cont:
        return [{"label": c, "value": c} for c in sorted(ALL_COUNTRIES.keys())]
    cl = cont if isinstance(cont, list) else [cont]
    cs = []
    for c in cl: cs.extend(CONTINENT_COUNTRIES.get(c, []))
    return [{"label": c, "value": c} for c in sorted(set(cs))]


@app.callback(
    Output("int-results",   "children"),
    Input("int-reason",     "value"),
    Input("int-continent",  "value"),
    Input("int-country",    "value"),
)
def update_interruptions(reason, continents, country):
    if not reason:
        return html.Div([
            html.P("Select an interruption reason to explore.",
                   style={"color": C["muted"], "marginTop": "10px"}),
        ], style={"textAlign": "center", "padding": "40px",
                  "background": C["card2"], "borderRadius": "12px",
                  "border": f"1px solid {C['border']}"})

    fdf = df[df["Interrupted_Reason"] == reason].copy()
    if continents:
        cl = continents if isinstance(continents, list) else [continents]
        fdf = fdf[fdf["Region"].isin(cl)]
    if len(fdf) == 0:
        return dbc.Alert("No data for this selection.", color="warning")

    sp_c   = fdf.groupby("Species").size()
    sp_pct = (sp_c / sp_c.sum() * 100).round(1)
    fig_sp = pie(sp_pct.index.tolist(), sp_pct.values.tolist(),
                 title=f"Birds Affected by {reason} (%)", height=300)

    rec_sp = fdf.groupby("Species")["Recovery_Time_days"].mean().round(1)
    rec_colors = [C["danger"] if v > 65 else C["accent3"] if v > 55 else C["success"]
                  for v in rec_sp.values]
    fig_rec = bar(rec_sp.index.tolist(), rec_sp.values.tolist(), colors=rec_colors,
                  texts=[f"{v:.1f} days" for v in rec_sp.values],
                  title=f"Avg Recovery Days by Species after {reason}",
                  ytitle="Recovery Days", height=300)

    rec_reg = fdf.groupby("Region")["Recovery_Time_days"].mean().round(1).reset_index()
    rec_reg.columns = ["Region", "Days"]
    rec_reg = rec_reg.sort_values("Days", ascending=False)
    reg_colors = [C["danger"] if v > 62 else C["accent3"] if v > 58 else C["success"]
                  for v in rec_reg["Days"]]
    fig_reg = bar(rec_reg["Region"].tolist(), rec_reg["Days"].tolist(),
                  orient="h", colors=reg_colors,
                  texts=[f"{v:.1f} days" for v in rec_reg["Days"]],
                  title=f"Avg Recovery Days by Region after {reason}",
                  xtitle="Recovery Days", height=300)

    top_birds = (fdf.groupby("Bird_Name")["Recovery_Time_days"].mean()
                 .sort_values(ascending=False).head(10))
    fig_birds = bar(top_birds.index.tolist(), top_birds.values.round(1).tolist(),
                    orient="h", colors=[C["danger"]] * len(top_birds),
                    texts=[f"{v:.1f} days" for v in top_birds.values],
                    title="Top Affected Birds (Longest Recovery)",
                    xtitle="Avg Recovery Days", height=320)

    sr_sp = fdf.groupby("Species")["Success"].mean() * 100
    sr_colors = [C["success"] if v >= 50 else C["danger"] for v in sr_sp.values]
    fig_sr = bar(sr_sp.index.tolist(), sr_sp.values.round(1).tolist(),
                 colors=sr_colors, texts=[f"{v:.1f}%" for v in sr_sp.values],
                 title=f"Migration Success Rate Despite {reason} (%)",
                 ytitle="Success Rate (%)", height=280)
    fig_sr.add_hline(y=50, line_dash="dash", line_color=C["muted"], opacity=0.5)

    avg_rec       = fdf["Recovery_Time_days"].mean()
    most_affected = sp_c.idxmax()

    return html.Div([
        dbc.Row([
            dbc.Col(icard("", "Most Affected Species", most_affected,
                          f"{sp_pct[most_affected]:.1f}% of interruptions", C["danger"]), md=4),
            dbc.Col(icard("", "Avg Recovery Time", f"{avg_rec:.1f} days",
                          "across affected birds", C["accent3"]), md=4),
            dbc.Col(icard("", "Regions Affected", str(fdf["Region"].nunique()),
                          "continents with this interruption", C["accent"]), md=4),
        ], className="mb-4 g-3"),
        dbc.Row([
            dbc.Col(graph(fig_sp), md=5),
            dbc.Col(graph(fig_sr), md=7),
        ], className="mb-3 g-3"),
        dbc.Row([
            dbc.Col(graph(fig_rec), md=6),
            dbc.Col(graph(fig_reg), md=6),
        ], className="mb-3 g-3"),
        dbc.Row([
            dbc.Col(graph(fig_birds), md=8),
        ], className="g-3"),
    ])


# Migration status — continent→country cascades
@app.callback(Output("ms-orig-country", "options"), Input("ms-orig-cont", "value"))
def ms_orig_c(cont):
    if not cont: return [{"label": c, "value": c} for c in sorted(ALL_COUNTRIES.keys())]
    return [{"label": c, "value": c} for c in sorted(CONTINENT_COUNTRIES.get(cont, []))]

@app.callback(Output("ms-dest-country", "options"), Input("ms-dest-cont", "value"))
def ms_dest_c(cont):
    if not cont: return [{"label": c, "value": c} for c in sorted(ALL_COUNTRIES.keys())]
    return [{"label": c, "value": c} for c in sorted(CONTINENT_COUNTRIES.get(cont, []))]


@app.callback(
    Output("ms-results", "children"),
    Input("ms-btn",          "n_clicks"),
    State("ms-orig-cont",    "value"),
    State("ms-orig-country", "value"),
    State("ms-dest-cont",    "value"),
    State("ms-dest-country", "value"),
    State("ms-species",      "value"),
    prevent_initial_call=True,
)
def ms_analyse(_, orig_cont, orig_country, dest_cont, dest_country, sp_filter):
    if not orig_cont:
        return dbc.Alert("Please select at least an origin continent.", color="warning",
                         style={"background": C["accent3"] + "22", "color": C["accent3"],
                                "border": f"1px solid {C['accent3']}"})
    fdf = df[df["Region"] == orig_cont].copy()
    if sp_filter:
        fdf = fdf[fdf["Species"].isin(sp_filter)]
    if len(fdf) == 0:
        return dbc.Alert("No data for this selection.", color="warning")

    g = fdf.groupby("Species")["Success"].agg(["sum", "count"]).reset_index()
    g.columns = ["Species", "Succ", "Total"]
    g["Fail"] = g["Total"] - g["Succ"]
    g["Rate"] = (g["Succ"] / g["Total"] * 100).round(1)
    g = g.sort_values("Rate", ascending=False)
    best_sp  = g.iloc[0]
    worst_sp = g.iloc[-1]

    bar_cols = (
        [C["success"]] + [C["accent"]] * max(0, len(g) - 2) + [C["danger"]]
        if len(g) > 1 else [C["success"]]
    )
    title_str = (
        f"Migration Success Rate from {orig_cont}"
        + (f" ({orig_country})" if orig_country else "")
        + (f" → {dest_cont}" if dest_cont else "")
    )
    fig_rate = bar(g["Species"].tolist(), g["Rate"].tolist(),
                   colors=bar_cols, texts=[f"{v:.1f}%" for v in g["Rate"]],
                   title=title_str, ytitle="Success Rate (%)", height=320)
    fig_rate.add_hline(y=50, line_dash="dash", line_color=C["muted"], opacity=0.5)

    ovr_succ = g["Succ"].sum() / g["Total"].sum() * 100
    fig_pie  = pie(["Successful", "Failed"], [ovr_succ, 100 - ovr_succ],
                   title="Overall Success vs Failure (%)", height=280)

    dest_r = dest_cont or "Europe"
    
    # FIX: Use country-specific coordinates if selected, otherwise use region center
    if orig_country and orig_country in COUNTRY_COORDS:
        oc = COUNTRY_COORDS[orig_country]
    else:
        oc = REGION_CENTRE[orig_cont]
    
    if dest_country and dest_country in COUNTRY_COORDS:
        dc = COUNTRY_COORDS[dest_country]
    else:
        dc = REGION_CENTRE[dest_r]
    
    # FIX: Generate waypoints from actual country/region coordinates, not fixed regions
    # Create a direct route from origin to destination via midpoint
    wps = [oc, ((oc[0]+dc[0])/2, (oc[1]+dc[1])/2), dc]
    lats = [p[0] for p in wps]
    lons = [p[1] for p in wps]

    fig_globe = go.Figure()
    fig_globe.add_trace(go.Scattergeo(
        lat=lats, lon=lons, mode="lines",
        line=dict(width=5, color=C["success"]),
        name=f"Best: {best_sp['Species']} ({best_sp['Rate']:.1f}%)"
    ))
    fig_globe.add_trace(go.Scattergeo(
        lat=lats, lon=lons, mode="lines",
        line=dict(width=5, color=C["danger"]),
        name=f"Worst: {worst_sp['Species']} ({worst_sp['Rate']:.1f}%)"
    ))
    
    # FIX: Show origin marker at correct coordinates
    origin_label = orig_country if orig_country else orig_cont
    fig_globe.add_trace(go.Scattergeo(lat=[oc[0]], lon=[oc[1]], mode="markers",
                                       marker=dict(size=12, color=C["accent"],
                                                   symbol="star"),
                                       name=f"Origin: {origin_label}"))
    
    # FIX: Show destination marker at correct coordinates
    dest_label = dest_country if dest_country else dest_r
    fig_globe.add_trace(go.Scattergeo(lat=[dc[0]], lon=[dc[1]], mode="markers",
                                       marker=dict(size=12, color=C["accent3"],
                                                   symbol="star"),
                                       name=f"Destination: {dest_label}"))
    
    fig_globe.update_geos(
        projection_type="natural earth",
        showland=True, landcolor="#c8dda7",
        showocean=True, oceancolor="#aad4f5",
        showcoastlines=True, coastlinecolor="#5a8a40"
    )
    fig_globe.update_layout(
        **BL, height=380,
        title=f"Route: {orig_cont} → {dest_r}",
        legend=dict(font=dict(color=C["text"]))
    )

    return html.Div([
        dbc.Row([
            dbc.Col(icard("", "Best Species Route", best_sp["Species"],
                          f"{best_sp['Rate']:.1f}% success",  C["success"]), md=4),
            dbc.Col(icard("", "Riskiest Species Route", worst_sp["Species"],
                          f"{worst_sp['Rate']:.1f}% success", C["danger"]),  md=4),
            dbc.Col(icard("", "Overall Route Success",  f"{ovr_succ:.1f}%",
                          f"from {orig_cont}", C["accent"]), md=4),
        ], className="mb-4 g-3"),
        graph(fig_globe, cls="fly-in", style={"marginBottom": "20px"}),
        dbc.Row([
            dbc.Col(graph(fig_rate), md=8),
            dbc.Col(graph(fig_pie),  md=4),
        ], className="mb-4 g-3"),
        html.H6("Per-Species Analysis",
                style={"color": C["accent"], "fontWeight": "700", "marginBottom": "14px"}),
        dbc.Row([
            dbc.Col(
                html.Div([
                    html.P(f"{sp}", style={"color": C["accent2"], "fontWeight": "700",
                                           "fontSize": "0.9rem", "marginBottom": "6px"}),
                    html.P(f"Success Rate: {rate:.1f}%",
                           style={"color": C["text"], "fontSize": "0.8rem",
                                  "marginBottom": "2px"}),
                    html.P(f"Sample: {row['Total']} migrations",
                           style={"color": C["muted"], "fontSize": "0.75rem",
                                  "marginBottom": "0"}),
                ], style={"background": C["card2"], "border": f"1px solid {C['border']}",
                          "borderRadius": "10px", "padding": "12px"}),
                md=4
            )
            for sp, rate, row in zip(g["Species"], g["Rate"], g.to_dict("records"))
        ], className="g-2"),
    ])

# ══════════════════════════════════════════════════════════════
# 13. CSS + SCROLL-FLY-IN ANIMATION (Intersection Observer)
# ══════════════════════════════════════════════════════════════
app.index_string = f"""<!DOCTYPE html>
<html>
<head>
{{%metas%}}
<title>Bird Migration Intelligence</title>
{{%favicon%}}
{{%css%}}
<style>
/* ── Base ── */
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: {C['bg']} !important; font-family: 'Inter', sans-serif; }}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: {C['card']}; }}
::-webkit-scrollbar-thumb {{ background: {C['border']}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {C['accent']}44; }}

/* ── Nav tabs ── */
button[id^="tab-"] {{ transition: color .18s, border-bottom-color .18s !important; }}
button[id^="tab-"]:hover {{
  color: {C['accent']} !important;
  border-bottom-color: {C['accent']} !important;
}}

/* ── Cards ── */
.card {{ transition: transform .2s, box-shadow .2s !important; }}
.card:hover {{
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 32px rgba(0,212,255,0.09) !important;
}}

/* ── Dropdowns ── */
.Select-control {{ background: {C['card2']} !important; border-color: {C['border']} !important; }}
.Select-menu-outer, .Select-option {{ background: {C['card2']} !important; color: {C['text']} !important; }}
.Select-option:hover, .Select-option.is-focused {{ background: {C['accent']}22 !important; }}
.Select-value-label, .Select-placeholder {{ color: {C['muted']} !important; }}
.VirtualizedSelectOption {{ background: {C['card2']} !important; color: {C['text']} !important; }}

/* ── Page transition ── */
@keyframes pageIn {{
  from {{ opacity: 0; transform: translateY(14px); }}
  to   {{ opacity: 1; transform: translateY(0); }}
}}
#page-content > div {{ animation: pageIn .3s ease; }}

/* ── Scroll fly-in ── */
@keyframes flyIn {{
  from {{
    opacity: 0;
    transform: translateY(40px) scale(0.97);
  }}
  to {{
    opacity: 1;
    transform: translateY(0) scale(1);
  }}
}}

.fly-in {{
  opacity: 0;
  transform: translateY(40px) scale(0.97);
  transition: opacity 0.55s cubic-bezier(.22,.68,0,1.2),
              transform 0.55s cubic-bezier(.22,.68,0,1.2);
}}
.fly-in.visible {{
  opacity: 1;
  transform: translateY(0) scale(1);
}}

/* Bird btn hover */
[id*="bird-btn"]:hover {{
  background: {C['card2']} !important;
  border-color: {C['accent']}55 !important;
}}

hr {{ border-color: {C['border']} !important; margin: 14px 0; }}
</style>
</head>
<body>
{{%app_entry%}}
<footer>
{{%config%}}
{{%scripts%}}
{{%renderer%}}
</footer>

<script>
/* Intersection Observer — triggers fly-in when elements enter viewport */
(function() {{
  function initObserver() {{
    const targets = document.querySelectorAll('.fly-in');
    if (!targets.length) {{ return; }}

    const observer = new IntersectionObserver(function(entries) {{
      entries.forEach(function(entry) {{
        if (entry.isIntersecting) {{
          // Stagger siblings slightly
          const siblings = Array.from(entry.target.parentElement.children)
            .filter(el => el.classList.contains('fly-in'));
          const idx = siblings.indexOf(entry.target);
          entry.target.style.transitionDelay = (idx * 0.07) + 's';
          entry.target.classList.add('visible');
          observer.unobserve(entry.target);
        }}
      }});
    }}, {{ threshold: 0.1, rootMargin: '0px 0px -40px 0px' }});

    targets.forEach(function(el) {{ observer.observe(el); }});
  }}

  // Run on load and re-run whenever Dash updates the DOM
  const mo = new MutationObserver(function() {{
    initObserver();
  }});
  mo.observe(document.body, {{ childList: true, subtree: true }});
  window.addEventListener('load', initObserver);
}})();
</script>
</body>
</html>"""

# ══════════════════════════════════════════════════════════════
# 14. PATTERN-MATCH CALLBACK — bird button clicks
# ══════════════════════════════════════════════════════════════
# We need to import ALL for pattern-matching
from dash import ALL as _ALL

@app.callback(
    Output("map-bird-detail", "children"),
    Input({"type": "bird-btn", "index": _ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def bird_btn_click(n_clicks_list):
    ctx = callback_context
    if not ctx.triggered:
        return html.Div()
    # Find which button was actually clicked
    for trig in ctx.triggered:
        if trig["value"]:
            try:
                prop_id = trig["prop_id"].replace(".n_clicks", "")
                bird_name = json.loads(prop_id)["index"]
                return bird_detail_panel(bird_name)
            except Exception:
                pass
    return html.Div()

# ══════════════════════════════════════════════════════════════
# 15. RUN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 8050))
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    print(f"\n  Bird Migration Intelligence  →  http://localhost:{port}")
    print(f"  Gunicorn: gunicorn bird_migration_app:server -b 0.0.0.0:{port} -w 2")
    print(f"  Render/Railway: set CSV_PATH env var to your uploaded file path\n")
    app.run(debug=debug, host="0.0.0.0", port=port)