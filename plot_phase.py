"""
3D BSA-YCl3 phase diagram scatter with image display.
Click a point to see its photo. Cross-section slider. Sample strip viewer.
Run: python plot_phase.py
Then open http://127.0.0.1:8050 in your browser.
"""

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, ctx, ALL
import os
import base64

CSV_FILE = "samples.csv"
IMG_FOLDER = "images"

# --- Load data ---
df = pd.read_csv(CSV_FILE)
df = df.dropna(subset=["T", "state"])
df["T"] = pd.to_numeric(df["T"], errors="coerce")
df = df.dropna(subset=["T"])
df = df.reset_index(drop=True)

# --- Color map ---
color_map = {
    "turbid":      "#e63946",
    "clear":       "#457b9d",
    "unconfirmed": "#f4a261",
    "denatured":   "#adb5bd",
}

# --- Sample grid map ---
SAMPLE_GRID = {
    1:  (25, 1),   2:  (25, 3),   3:  (25, 6),   4:  (25, 9),
    5:  (50, 1),   6:  (50, 3),   7:  (50, 6),   8:  (50, 9),
    9:  (100, 1),  10: (100, 3),  11: (100, 6),  12: (100, 9),
    13: (150, 1),  14: (150, 3),  15: (150, 6),  16: (150, 9),
    17: (175, 1),  18: (175, 3),  19: (175, 6),  20: (175, 9),
}

# reverse lookup: (cp, cs) -> sample number
REVERSE_GRID = {v: k for k, v in SAMPLE_GRID.items()}

# which sample numbers are actually in the data
present_samples = sorted([
    REVERSE_GRID[(r["cp"], r["cs"])]
    for _, r in df.drop_duplicates(subset=["cp", "cs"]).iterrows()
    if (r["cp"], r["cs"]) in REVERSE_GRID
])

# --- Build 3D scatter figure ---
def build_3d_figure():
    f = go.Figure()

    for state, color in color_map.items():
        sub = df[df["state"].str.lower().str.strip() == state].copy()
        if sub.empty:
            continue

        hover_texts = [
            (f"cp = {row['cp']} mg/mL<br>"
             f"cs = {row['cs']} mM<br>"
             f"T = {row['T']} C<br>"
             f"state: {row['state']}<br>"
             f"click to see photo")
            for _, row in sub.iterrows()
        ]

        f.add_trace(go.Scatter3d(
            x=sub["cp"],
            y=sub["cs"],
            z=sub["T"],
            mode="markers",
            name=state,
            marker=dict(
                size=6,
                color=color,
                opacity=0.9,
                line=dict(width=1, color="black")
            ),
            customdata=sub.index.tolist(),
            hovertemplate="%{text}<extra></extra>",
            text=hover_texts,
        ))

    # --- Wireframe lines between turbid points ---
    turbid = df[df["state"].str.lower().str.strip() == "turbid"].copy()
    turbid = turbid.sort_values(["cp", "cs", "T"])

    # vertical threads (same composition, across T)
    for (cp_val, cs_val), group in turbid.groupby(["cp", "cs"]):
        group = group.sort_values("T")
        if len(group) < 2:
            continue
        f.add_trace(go.Scatter3d(
            x=group["cp"], y=group["cs"], z=group["T"],
            mode="lines",
            line=dict(color="rgba(230, 57, 70, 0.75)", width=2),
            showlegend=False, hoverinfo="skip",
        ))

    # horizontal mesh (same T, across compositions)
    for t_val, group in turbid.groupby("T"):
        group = group.sort_values(["cp", "cs"])
        if len(group) < 2:
            continue
        f.add_trace(go.Scatter3d(
            x=group["cp"], y=group["cs"], z=group["T"],
            mode="lines",
            line=dict(color="rgba(230, 57, 70, 0.75)", width=2),
            showlegend=False, hoverinfo="skip",
        ))

    # in-plane diagonals (same T, different cp AND cs)
    for t_val, group in turbid.groupby("T"):
        group = group.sort_values(["cp", "cs"])
        if len(group) < 2:
            continue
        rows_list = list(group.iterrows())
        for idx_i, (_, row_i) in enumerate(rows_list):
            for _, row_j in rows_list[idx_i + 1:]:
                if row_i["cp"] == row_j["cp"] or row_i["cs"] == row_j["cs"]:
                    continue
                f.add_trace(go.Scatter3d(
                    x=[row_i["cp"], row_j["cp"]],
                    y=[row_i["cs"], row_j["cs"]],
                    z=[row_i["T"],  row_j["T"]],
                    mode="lines",
                    line=dict(color="rgba(230, 57, 70, 0.75)", width=2),
                    showlegend=False, hoverinfo="skip",
                ))

    # 3D diagonals (adjacent points across T AND composition)
    turbid_list = turbid.reset_index(drop=True)
    rows_all = list(turbid_list.iterrows())
    for idx_i, (_, row_i) in enumerate(rows_all):
        for _, row_j in rows_all[idx_i + 1:]:
            cp_diff = abs(row_i["cp"] - row_j["cp"])
            cs_diff = abs(row_i["cs"] - row_j["cs"])
            t_diff  = abs(row_i["T"]  - row_j["T"])
            if t_diff != 5:
                continue
            if cp_diff > 25:
                continue
            if cs_diff > 3:
                continue
            if cp_diff == 0 and cs_diff == 0:
                continue
            f.add_trace(go.Scatter3d(
                x=[row_i["cp"], row_j["cp"]],
                y=[row_i["cs"], row_j["cs"]],
                z=[row_i["T"],  row_j["T"]],
                mode="lines",
                line=dict(color="rgba(230, 57, 70, 0.75)", width=2),
                showlegend=False, hoverinfo="skip",
            ))

    f.update_layout(
        title="BSA-YCl3 Phase Observations",
        scene=dict(
            xaxis=dict(title="Protein conc, cp (mg/mL)", range=[0, 200], dtick=25),
            yaxis=dict(title="Salt conc, cs (mM)", range=[0, 10], dtick=1),
            zaxis=dict(title="Temperature, T (C)", range=[0, 50], dtick=10),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=-0.1),
                eye=dict(x=-1.5, y=-1.8, z=1.2)
            ),
            aspectmode="manual",
            aspectratio=dict(x=2, y=1, z=1.2),
        ),
        legend_title="Observed state",
        margin=dict(l=0, r=0, t=40, b=0),
        uirevision="constant",
    )
    return f


fig = build_3d_figure()

# --- Dash app ---
app = Dash(__name__)

app.layout = html.Div([

    html.H2("BSA-YCl3 Phase Diagram",
            style={"textAlign": "center", "fontFamily": "Arial"}),

    # --- Cross-section controls ---
    html.Div([
        html.Label("Cross-section temperature (C):",
                   style={"fontFamily": "Arial", "fontWeight": "bold"}),
        dcc.Slider(
            id="temp-slider",
            min=7, max=50, step=None,
            marks={int(t): str(int(t)) for t in sorted(df["T"].unique())},
            value=None, included=False,
        ),
        html.Div([
            html.Button("Show cross-section", id="cross-btn", n_clicks=0,
                        style={"marginRight": "10px", "padding": "6px 14px",
                               "fontFamily": "Arial", "cursor": "pointer"}),
            html.Button("Show full 3D", id="full3d-btn", n_clicks=0,
                        style={"padding": "6px 14px", "fontFamily": "Arial",
                               "cursor": "pointer"}),
        ], style={"marginTop": "10px"}),
        html.Div(id="cross-section-label",
                 style={"fontFamily": "Arial", "marginTop": "6px",
                        "color": "#555", "fontStyle": "italic"}),
    ], style={"padding": "10px 20px", "backgroundColor": "#f0f0f0",
              "borderRadius": "8px", "margin": "0 20px 10px 20px"}),

    # --- Sample strip controls ---
    html.Div([
        html.Label("View all temperatures for a sample:",
                   style={"fontFamily": "Arial", "fontWeight": "bold"}),
        html.Div([
            html.Button(
                f"S{i}",
                id={"type": "sample-btn", "index": i},
                n_clicks=0,
                style={"marginRight": "5px", "marginBottom": "5px",
                       "padding": "5px 10px", "fontFamily": "Arial",
                       "cursor": "pointer", "fontSize": "12px"}
            )
            for i in present_samples
        ], style={"display": "flex", "flexWrap": "wrap", "marginTop": "8px"}),
        html.Div(id="strip-label",
                 style={"fontFamily": "Arial", "fontStyle": "italic",
                        "color": "#555", "marginTop": "6px"}),
        html.Div(id="image-strip",
                 style={"display": "flex", "flexWrap": "wrap",
                        "marginTop": "10px", "gap": "10px",
                        "overflowX": "auto"}),
    ], style={"padding": "10px 20px", "backgroundColor": "#f0f0f0",
              "borderRadius": "8px", "margin": "0 20px 10px 20px"}),

    # --- Main plot + click panel ---
    html.Div([
        html.Div([
            dcc.Graph(
                id="scatter3d",
                figure=fig,
                style={"height": "80vh"},
                config={"scrollZoom": True}
            )
        ], style={"width": "65%", "display": "inline-block",
                  "verticalAlign": "top"}),

        html.Div([
            html.H4("Sample photo",
                    style={"fontFamily": "Arial", "marginBottom": "8px"}),
            html.Div(id="info-box",
                     style={"fontFamily": "Arial", "fontSize": "14px",
                            "marginBottom": "12px", "whiteSpace": "pre-line"}),
            html.Img(id="sample-img", src="",
                     style={"maxWidth": "100%", "maxHeight": "55vh",
                            "border": "1px solid #ccc", "borderRadius": "6px",
                            "display": "none"}),
            html.P("(click a point to load photo)", id="placeholder-text",
                   style={"color": "#888", "fontFamily": "Arial",
                          "fontStyle": "italic"}),
        ], style={"width": "33%", "display": "inline-block",
                  "verticalAlign": "top", "paddingLeft": "20px",
                  "paddingTop": "40px"}),
    ]),

], style={"backgroundColor": "#f8f9fa", "padding": "20px"})


# --- Callback: click point -> show image ---
@app.callback(
    Output("sample-img", "src"),
    Output("sample-img", "style"),
    Output("info-box", "children"),
    Output("placeholder-text", "style"),
    Input("scatter3d", "clickData"),
)
def show_image(clickData):
    img_hidden  = {"maxWidth": "100%", "maxHeight": "55vh",
                   "border": "1px solid #ccc", "borderRadius": "6px",
                   "display": "none"}
    img_visible = {"maxWidth": "100%", "maxHeight": "55vh",
                   "border": "1px solid #ccc", "borderRadius": "6px",
                   "display": "block"}
    ph_visible = {"color": "#888", "fontFamily": "Arial", "fontStyle": "italic"}
    ph_hidden  = {"display": "none"}

    if clickData is None:
        return "", img_hidden, "", ph_visible

    point = clickData["points"][0]
    row_idx = point.get("customdata")
    if row_idx is None:
        return "", img_hidden, "Could not identify point.", ph_visible

    row = df.iloc[int(row_idx)]
    info = (f"cp = {row['cp']} mg/mL\n"
            f"cs = {row['cs']} mM\n"
            f"T = {row['T']} C\n"
            f"state: {row['state']}\n"
            f"file: {row['image']}")

    img_path = os.path.join(IMG_FOLDER, str(row.get("image", "")).strip())
    if os.path.exists(img_path):
        with open(img_path, "rb") as f_img:
            encoded = base64.b64encode(f_img.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}", img_visible, info, ph_hidden
    else:
        return "", img_hidden, info + "\n(no image file)", ph_visible


# --- Callback: cross-section / full 3D toggle ---
@app.callback(
    Output("scatter3d", "figure"),
    Output("cross-section-label", "children"),
    Input("cross-btn", "n_clicks"),
    Input("full3d-btn", "n_clicks"),
    Input("temp-slider", "value"),
    prevent_initial_call=True,
)
def update_view(cross_clicks, full3d_clicks, selected_T):
    triggered = ctx.triggered_id

    if triggered == "full3d-btn":
        return fig, ""

    if triggered in ("cross-btn", "temp-slider") and selected_T is not None:
        subset = df[df["T"] == selected_T].copy()

        if subset.empty:
            return fig, f"No data at T = {selected_T} C."

        cross_fig = go.Figure()

        for state, color in color_map.items():
            sub = subset[subset["state"].str.lower().str.strip() == state].copy()
            if sub.empty:
                continue

            hover_texts = [
                (f"cp = {row['cp']} mg/mL<br>"
                 f"cs = {row['cs']} mM<br>"
                 f"T = {row['T']} C<br>"
                 f"state: {row['state']}<br>"
                 f"click to see photo")
                for _, row in sub.iterrows()
            ]

            cross_fig.add_trace(go.Scatter(
                x=sub["cp"],
                y=sub["cs"],
                mode="markers",
                name=state,
                marker=dict(
                    size=12,
                    color=color,
                    opacity=0.9,
                    line=dict(width=1, color="black")
                ),
                customdata=sub.index.tolist(),
                hovertemplate="%{text}<extra></extra>",
                text=hover_texts,
            ))

        cross_fig.update_layout(
            title=f"Cross-section at T = {int(selected_T)} C",
            xaxis=dict(
                title="Protein conc, cp (mg/mL)",
                range=[0, 200],
                dtick=25,
            ),
            yaxis=dict(
                title="Salt conc, cs (mM)",
                range=[0, 10],
                dtick=1,
            ),
            legend_title="Observed state",
            plot_bgcolor="white",
            paper_bgcolor="#f8f9fa",
            font=dict(family="Arial"),
        )

        label = (f"Showing cross-section at T = {int(selected_T)} C  |  "
                 f"{len(subset)} points")
        return cross_fig, label

    return fig, ""


# --- Callback: sample strip ---
@app.callback(
    Output("image-strip", "children"),
    Output("strip-label", "children"),
    Input({"type": "sample-btn", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def show_sample_strip(n_clicks_list):
    triggered = ctx.triggered_id
    if triggered is None:
        return [], ""

    sample_num = triggered["index"]
    if sample_num not in SAMPLE_GRID:
        return [], f"Unknown sample {sample_num}."

    cp_val, cs_val = SAMPLE_GRID[sample_num]
    rows = df[(df["cp"] == cp_val) & (df["cs"] == cs_val)].sort_values("T")

    if rows.empty:
        return [], (f"No data found for Sample {sample_num} "
                    f"(cp={cp_val}, cs={cs_val}).")

    label = (f"Sample {sample_num}  |  "
             f"cp = {cp_val} mg/mL, cs = {cs_val} mM  |  "
             f"{len(rows)} temperature points")

    items = []
    for _, row in rows.iterrows():
        T_val    = int(row["T"])
        state    = row["state"]
        img_filename = str(row.get("image", "")).strip()
        img_path = os.path.join(IMG_FOLDER, img_filename)
        badge_color = color_map.get(state.lower().strip(), "#adb5bd")

        if os.path.exists(img_path):
            with open(img_path, "rb") as f_img:
                encoded = base64.b64encode(f_img.read()).decode("utf-8")
            img_src = f"data:image/jpeg;base64,{encoded}"
            img_el = html.Img(
                src=img_src,
                style={"width": "160px", "height": "200px",
                       "objectFit": "contain",
                       "backgroundColor": "#f8f9fa",
                       "border": "2px solid " + badge_color,
                       "borderRadius": "6px",
                       "padding": "4px"}
            )
        else:
            img_el = html.Div(
                "no image",
                style={"width": "140px", "height": "180px",
                       "display": "flex", "alignItems": "center",
                       "justifyContent": "center",
                       "border": "2px solid #ccc",
                       "borderRadius": "6px",
                       "color": "#aaa", "fontFamily": "Arial",
                       "fontSize": "12px"}
            )

        card = html.Div([
            img_el,
            html.Div(f"T = {T_val} C",
                     style={"textAlign": "center", "fontFamily": "Arial",
                            "fontSize": "13px", "fontWeight": "bold",
                            "marginTop": "4px"}),
            html.Div(state,
                     style={"textAlign": "center", "fontFamily": "Arial",
                            "fontSize": "11px", "color": badge_color,
                            "fontWeight": "bold"}),
        ], style={"display": "flex", "flexDirection": "column",
                  "alignItems": "center", "width": "170px"})
        items.append(card)

    return items, label


if __name__ == "__main__":
    app.run(debug=False)