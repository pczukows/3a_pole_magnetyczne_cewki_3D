
import numpy as np
import plotly.graph_objects as go
import streamlit as st

MU_0 = 4.0 * np.pi * 1e-7

st.set_page_config(page_title="Pole magnetyczne cewki 3D", page_icon="🧲", layout="wide")
st.title("🧲 Linie pola magnetycznego wokół cewki")
st.markdown("#### Model 3D oparty na faktycznych obliczeniach pola według praw fizyki")


@st.cache_data(show_spinner=False)
def coil_geometry(radius, length, turns, segments_per_turn=28):
    count = max(int(turns * segments_per_turn), 100)
    phi = np.linspace(0.0, 2.0 * np.pi * turns, count + 1)
    points = np.column_stack((
        radius * np.cos(phi),
        radius * np.sin(phi),
        np.linspace(-length / 2.0, length / 2.0, count + 1),
    ))
    dl = np.diff(points, axis=0)
    mids = 0.5 * (points[:-1] + points[1:])
    return points, mids, dl


def field_at(points, mids, dl):
    """Pole dla prądu 1 A. points ma kształt (n, 3)."""
    points = np.atleast_2d(points).astype(float)
    result = np.zeros_like(points)
    for start in range(0, len(mids), 220):
        m = mids[start:start + 220]
        d = dl[start:start + 220]
        r = points[:, None, :] - m[None, :, :]
        distance = np.maximum(np.linalg.norm(r, axis=2), 1e-6)
        result += np.sum(np.cross(d[None, :, :], r) / distance[..., None] ** 3, axis=1)
    return MU_0 * result / (4.0 * np.pi)


def normalized_direction(point, mids, dl, sign):
    b = field_at(np.asarray(point)[None, :], mids, dl)[0]
    norm = np.linalg.norm(b)
    if norm < 1e-15:
        return np.zeros(3)
    return sign * b / norm


def trace_half(seed, direction, mids, dl, step, max_steps, bounds, wire_clearance):
    """RK4 po znormalizowanym polu; daje jedną połowę linii pola."""
    points = [np.asarray(seed, dtype=float)]
    p = points[0].copy()
    for _ in range(max_steps):
        k1 = normalized_direction(p, mids, dl, direction)
        if not np.any(k1):
            break
        k2 = normalized_direction(p + 0.5 * step * k1, mids, dl, direction)
        k3 = normalized_direction(p + 0.5 * step * k2, mids, dl, direction)
        k4 = normalized_direction(p + step * k3, mids, dl, direction)
        p_new = p + step * (k1 + 2*k2 + 2*k3 + k4) / 6.0

        radial = np.hypot(p_new[0], p_new[1])
        if radial > bounds[0] or abs(p_new[2]) > bounds[1]:
            break
        # Ochrona przed osobliwością idealnie cienkiego przewodnika.
        if np.min(np.linalg.norm(mids - p_new, axis=1)) < wire_clearance:
            break
        if len(points) > 25 and np.linalg.norm(p_new - seed) < 0.7 * step:
            points.append(p_new)
            break
        points.append(p_new)
        p = p_new
    return np.asarray(points)


@st.cache_data(show_spinner=False)
def build_field_lines(radius, length, turns, line_count, quality):
    coil, mids, dl = coil_geometry(radius, length, turns)
    radial_bound = max(3.0 * radius, 1.8 * length)
    axial_bound = max(2.6 * length, 3.0 * radius)
    step = min(radius, max(length, 0.2)) / (13.0 + 3.0 * quality)
    max_steps = 170 + 45 * quality
    clearance = max(0.025, 0.035 * radius)

    # Nasiona w płaszczyźnie środkowej; obrót daje przestrzenny pęk linii.
    radial_levels = np.linspace(0.08, 0.82, max(2, line_count // 4)) * radius
    azimuths = np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)
    seeds = []
    for rho in radial_levels:
        for angle in azimuths:
            seeds.append(np.array([rho*np.cos(angle), rho*np.sin(angle), 0.0]))
    seeds = seeds[:line_count]

    lines = []
    for seed in seeds:
        backward = trace_half(seed, -1.0, mids, dl, step, max_steps, (radial_bound, axial_bound), clearance)
        forward = trace_half(seed, 1.0, mids, dl, step, max_steps, (radial_bound, axial_bound), clearance)
        line = np.vstack((backward[:0:-1], forward))
        if len(line) > 8:
            lines.append(line)
    return coil, mids, dl, lines, radial_bound, axial_bound


with st.sidebar:
    st.header("Parametry cewki")
    radius = st.slider(
        "Promień cewki [m]", 0.20, 1.50, 0.70, 0.05,
        help="Odległość od osi cewki do przewodnika. Większy promień oznacza większy przekrój cewki.",
    )
    length = st.slider(
        "Długość cewki [m]", 0.10, 3.00, 1.40, 0.05,
        help="Odległość między początkiem i końcem uzwojenia mierzona wzdłuż osi cewki.",
    )
    turns = st.slider(
        "Liczba zwojów", 0.2, 35.0, 12.0, 0.2,
        help="Liczba pełnych obrotów przewodnika wokół osi. Więcej zwojów wzmacnia pole wewnątrz cewki.",
    )

    st.header("Prąd stały")
    current = st.slider(
        "Natężenie prądu I [A]", 0.0, 20.0, 5.0, 0.1,
        help="Stały prąd płynący przez uzwojenie.",
    )

    st.header("Sposób prezentacji")
    line_count = st.slider(
        "Liczba umownych linii pola", 4, 24, 12, 4,
        help="Linie pola są pomocą graficzną. Pole magnetyczne istnieje w całej przestrzeni, nie tylko w miejscach narysowanych linii.",
    )

    st.info(
        "Model pokazuje kształt pola wytwarzanego przez jedną ciągłą, spiralnie nawiniętą cewkę. "
        "Nie jest to projekt techniczny ani obraz rzeczywistych, materialnych linii."
    )

# Parametry numeryczne pozostają stałe, aby interfejs skupiał uwagę
# na wielkościach fizycznych, a nie na ustawieniach algorytmu.
quality = 2
line_width = 3 # zmieniłem z 4 na 3, aby linie były nieco cieńsze i bardziej estetyczne
with st.spinner("Wyznaczanie linii pola..."):
    coil, mids, dl, lines, radial_bound, axial_bound = build_field_lines(
        radius, length, turns, line_count, quality
    )

# Pole i skala kolorów są liczone przy wejściu każdej linii.
seed_points = np.array([line[len(line)//2] for line in lines]) if lines else np.empty((0, 3))
seed_b_1a = np.linalg.norm(field_at(seed_points, mids, dl), axis=1) if len(seed_points) else np.array([])
seed_b = seed_b_1a * abs(current)
log_b = np.log10(np.maximum(seed_b, 1e-12))
if len(log_b):
    low, high = float(log_b.min()), float(log_b.max())
    span = max(high - low, 1e-9)
else:
    low, span = -12.0, 1.0

fig = go.Figure()
fig.add_trace(go.Scatter3d(
    x=coil[:, 0], y=coil[:, 1], z=coil[:, 2], mode="lines",
    line=dict(color="#f59e0b", width=9), name="Cewka", # zmieniłem width z 7 na 9
    hovertemplate="Cewka<extra></extra>",
))

palette = ["#2563eb", "#06b6d4", "#22c55e", "#eab308", "#f97316", "#ef4444"]
current_sign = 1 if current >= 0 else -1
for index, line in enumerate(lines):
    color_index = int(np.clip((log_b[index] - low) / span * (len(palette) - 1), 0, len(palette) - 1)) if len(log_b) else 0
    color = palette[color_index]
    fig.add_trace(go.Scatter3d(
        x=line[:, 0], y=line[:, 1], z=line[:, 2], mode="lines",
        line=dict(color=color, width=line_width),
        name="Linia pola" if index == 0 else None,
        showlegend=index == 0,
        customdata=np.full(len(line), seed_b[index] if len(seed_b) else 0.0),
        hovertemplate="|B| przy nasionie: %{customdata:.3e} T<extra></extra>",
    ))

fig.update_layout(
    height=740,
    margin=dict(l=0, r=0, t=45, b=0),
    title=f"Linie pola magnetycznego — prąd stały I = {current:.3f} A",
    paper_bgcolor="#0b1020", font=dict(color="#e5e7eb"),
    legend=dict(bgcolor="rgba(0,0,0,0.25)"),
    scene=dict(
        bgcolor="#0b1020",
        xaxis=dict(title="x [m]", range=[-radial_bound, radial_bound], gridcolor="#334155"),
        yaxis=dict(title="y [m]", range=[-radial_bound, radial_bound], gridcolor="#334155"),
        zaxis=dict(title="z [m]", range=[-axial_bound, axial_bound], gridcolor="#334155"),
        aspectmode="cube", camera=dict(eye=dict(x=1.55, y=1.55, z=1.15)),
    ),
)

left, right = st.columns([3, 1])
with left:
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
with right:
    st.subheader("Uwagi:")
    st.info(
        "Kolor rozróżnia względną wartość pola dla poszczególnych linii. "
        "Niebieskie odcienie oznaczają mniejszą, a czerwone większą wartość natężenia pola magnetycznego |B|."
    )
    st.warning(
        "Linie pola nie są materialnymi włóknami ani torami ruchu cząstek. "
        "Są umownym sposobem przedstawienia kierunku pola magnetycznego w przestrzeni."
    )

with st.expander("Jak czytać rysunek?", expanded=False):
    st.markdown(r"""
- **Pomarańczowa helisa** przedstawia jeden ciągły przewodnik nawinięty wokół osi cewki.
- **Kolorowe krzywe** są umownymi liniami pola magnetycznego.
- Zagęszczenie i przebieg linii pomagają rozpoznać obszary o różnej intensywności oraz kształt pola.
- Wewnątrz odpowiednio długiej cewki pole jest bardziej uporządkowane i zbliżone do kierunku osi cewki.
- Poza cewką linie zawracają, tworząc zamknięte pętle.
""")

with st.expander("Model obliczeniowy i ograniczenia"):
    st.markdown(r"""
Linie są wyznaczane poprzez wykonanie obliczeń opartych na wzorach fizycznych znanych z podręczników.
Model służy do jakościowego pokazania kształtu (geometrii) pola. 
""")
