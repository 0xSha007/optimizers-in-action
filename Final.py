import streamlit as st
import numpy as np
import plotly.graph_objects as go

# --------------------------------------
# 0. Page Config & Basic CSS
# --------------------------------------
st.set_page_config(layout="wide")
st.markdown("""
<style>
[data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
    width: 300px;
}
[data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
    margin-left: -300px;
}
/* Make Plotly modebar icons more visible on dark backgrounds */
.modebar-btn, .modebar-btn svg {
    fill: #000 !important;
    color: #000 !important;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------
# 1. Cost Function & Gradient (1D)
# --------------------------------------
def cost_function(x):
    """Quadratic function with a global minimum at x=3 => cost=5."""
    return (x - 3)**2 + 5

def gradient(x):
    """Gradient: f'(x) = 2*(x - 3)."""
    return 2 * (x - 3)

GLOBAL_MIN_COST = 5.0   # known minimum cost at x=3
COST_TOL = 1e-3         # stopping threshold: |cost - 5| < 1e-3

# --------------------------------------
# 2. Five Methods, Each Stops If Converged
# --------------------------------------
def run_batch_gd(x0, lr, max_epochs):
    x = x0
    x_hist = [x]
    for epoch in range(max_epochs):
        c = cost_function(x)
        if abs(c - GLOBAL_MIN_COST) < COST_TOL:
            break
        g = gradient(x)
        x -= lr * g
        x_hist.append(x)
    # Append final once more so we see the end state
    x_hist.append(x)
    return x_hist

def run_sgd(x0, lr, max_epochs):
    x = x0
    x_hist = [x]
    for epoch in range(max_epochs):
        c = cost_function(x)
        if abs(c - GLOBAL_MIN_COST) < COST_TOL:
            break
        g = gradient(x) + np.random.normal(0, 0.3)  # random noise
        x -= lr * g
        x_hist.append(x)
    x_hist.append(x)
    return x_hist

def run_mini_batch_gd(x0, lr, max_epochs, batch_size):
    data = np.random.uniform(-2, 8, 200)
    x = x0
    x_hist = [x]
    for epoch in range(max_epochs):
        c = cost_function(x)
        if abs(c - GLOBAL_MIN_COST) < COST_TOL:
            break
        mini_batch = np.random.choice(data, size=batch_size, replace=False)
        grads = [gradient(sample) for sample in mini_batch]
        x -= lr * np.mean(grads)
        x_hist.append(x)
    x_hist.append(x)
    return x_hist

def run_momentum_gd(x0, lr, max_epochs, momentum_factor):
    x = x0
    v = 0
    x_hist = [x]
    for epoch in range(max_epochs):
        c = cost_function(x)
        if abs(c - GLOBAL_MIN_COST) < COST_TOL:
            break
        g = gradient(x)
        v = momentum_factor*v + lr*g
        x -= v
        x_hist.append(x)
    x_hist.append(x)
    return x_hist

def run_adam(x0, lr, max_epochs, beta1, beta2, epsilon):
    x = x0
    m = 0
    v = 0
    x_hist = [x]
    for t in range(1, max_epochs+1):
        c = cost_function(x)
        if abs(c - GLOBAL_MIN_COST) < COST_TOL:
            break

        g = gradient(x)
        m = beta1*m + (1 - beta1)*g
        v = beta2*v + (1 - beta2)*(g*g)
        # Bias correction
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        x -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
        x_hist.append(x)
    x_hist.append(x)
    return x_hist

METHODS = [
    "Batch Gradient Descent",
    "Stochastic Gradient Descent",
    "Mini-Batch Gradient Descent",
    "Gradient Descent w/ Momentum",
    "Adam"
]

# ----------------------------------------------------
# 3. Sidebar Controls (Single vs. Compare, Hyperparams)
# ----------------------------------------------------
st.title("Gradient Descent with Early Stopping")

mode = st.sidebar.radio("Mode of Operation:", ["Single Method", "Compare Two Methods"])

if mode == "Single Method":
    method_selected = st.sidebar.selectbox("Select Method:", METHODS)
else:
    method_a = st.sidebar.selectbox("Select Method A:", METHODS, key="method_a")
    method_b = st.sidebar.selectbox("Select Method B:", METHODS, key="method_b")

initial_x = st.sidebar.slider("Initial x", -2.0, 8.0, 4.0, 0.1)
lr = st.sidebar.slider("Learning Rate (α)", 0.001, 1.0, 0.1, 0.001)
max_epochs = st.sidebar.slider("Max Epochs", 1, 200, 50, 1)

# Extra hyperparams
batch_size = 1
momentum_factor = 0.9
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

if mode == "Single Method":
    if method_selected == "Mini-Batch Gradient Descent":
        batch_size = st.sidebar.slider("Batch Size", 1, 50, 10, 1)
    elif method_selected == "Gradient Descent w/ Momentum":
        momentum_factor = st.sidebar.slider("Momentum Factor (β)", 0.0, 0.99, 0.9, 0.01)
    elif method_selected == "Adam":
        beta1 = st.sidebar.slider("Beta1", 0.0, 0.999, 0.9, 0.001)
        beta2 = st.sidebar.slider("Beta2", 0.0, 0.999, 0.999, 0.001)
        epsilon = st.sidebar.slider("ε", 1e-9, 1e-6, 1e-8, 1e-9)
else:
    # If comparing two methods, show relevant sliders for each
    if method_a == "Mini-Batch Gradient Descent":
        batch_size_a = st.sidebar.slider("Batch Size (A)", 1, 50, 10, 1)
    else:
        batch_size_a = 1

    if method_b == "Mini-Batch Gradient Descent":
        batch_size_b = st.sidebar.slider("Batch Size (B)", 1, 50, 10, 1)
    else:
        batch_size_b = 1

    if method_a == "Gradient Descent w/ Momentum":
        momentum_factor_a = st.sidebar.slider("Momentum (A)", 0.0, 0.99, 0.9, 0.01)
    else:
        momentum_factor_a = 0.9

    if method_b == "Gradient Descent w/ Momentum":
        momentum_factor_b = st.sidebar.slider("Momentum (B)", 0.0, 0.99, 0.9, 0.01)
    else:
        momentum_factor_b = 0.9

    if method_a == "Adam":
        beta1_a = st.sidebar.slider("Beta1 (A)", 0.0, 0.999, 0.9, 0.001)
        beta2_a = st.sidebar.slider("Beta2 (A)", 0.0, 0.999, 0.999, 0.001)
        epsilon_a = st.sidebar.slider("ε (A)", 1e-9, 1e-6, 1e-8, 1e-9)
    else:
        beta1_a, beta2_a, epsilon_a = 0.9, 0.999, 1e-8

    if method_b == "Adam":
        beta1_b = st.sidebar.slider("Beta1 (B)", 0.0, 0.999, 0.9, 0.001)
        beta2_b = st.sidebar.slider("Beta2 (B)", 0.0, 0.999, 0.999, 0.001)
        epsilon_b = st.sidebar.slider("ε (B)", 1e-9, 1e-6, 1e-8, 1e-9)
    else:
        beta1_b, beta2_b, epsilon_b = 0.9, 0.999, 1e-8

# ------------------------------------------------
# 4. Run Selected Methods - Stop Early
# ------------------------------------------------
def run_optimizer(name, x0, lr, max_e, b_size, mom, b1, b2, eps):
    """Helper to dispatch to the correct method function."""
    if name == "Batch Gradient Descent":
        return run_batch_gd(x0, lr, max_e)
    elif name == "Stochastic Gradient Descent":
        return run_sgd(x0, lr, max_e)
    elif name == "Mini-Batch Gradient Descent":
        return run_mini_batch_gd(x0, lr, max_e, b_size)
    elif name == "Gradient Descent w/ Momentum":
        return run_momentum_gd(x0, lr, max_e, mom)
    else:
        return run_adam(x0, lr, max_e, b1, b2, eps)

# We'll store a list of dictionaries with method info + the results
runs_info = []

if mode == "Single Method":
    # Single method scenario
    if method_selected == "Mini-Batch Gradient Descent":
        x_hist = run_optimizer(method_selected, initial_x, lr, max_epochs,
                               batch_size, 0.9, 0.9, 0.999, 1e-8)
    elif method_selected == "Gradient Descent w/ Momentum":
        x_hist = run_optimizer(method_selected, initial_x, lr, max_epochs,
                               1, momentum_factor, 0.9, 0.999, 1e-8)
    elif method_selected == "Adam":
        x_hist = run_optimizer(method_selected, initial_x, lr, max_epochs,
                               1, 0.9, beta1, beta2, epsilon)
    else:
        # Batch or SGD
        x_hist = run_optimizer(method_selected, initial_x, lr, max_epochs,
                               1, 0.9, 0.9, 0.999, 1e-8)

    c_hist = [cost_function(x) for x in x_hist]
    runs_info.append({
        "method": method_selected,
        "x_hist": x_hist,
        "c_hist": c_hist,
        "lr": lr,
        "batch_size": (batch_size if method_selected=="Mini-Batch Gradient Descent" else None),
        "momentum": (momentum_factor if method_selected=="Gradient Descent w/ Momentum" else None),
        "beta1": (beta1 if method_selected=="Adam" else None),
        "beta2": (beta2 if method_selected=="Adam" else None),
        "epsilon": (epsilon if method_selected=="Adam" else None),
    })
else:
    # Compare two
    # Method A
    if method_a == "Mini-Batch Gradient Descent":
        x_hist_a = run_optimizer(method_a, initial_x, lr, max_epochs,
                                 batch_size_a, 0.9, 0.9, 0.999, 1e-8)
    elif method_a == "Gradient Descent w/ Momentum":
        x_hist_a = run_optimizer(method_a, initial_x, lr, max_epochs,
                                 1, momentum_factor_a, 0.9, 0.999, 1e-8)
    elif method_a == "Adam":
        x_hist_a = run_optimizer(method_a, initial_x, lr, max_epochs,
                                 1, 0.9, beta1_a, beta2_a, epsilon_a)
    else:
        x_hist_a = run_optimizer(method_a, initial_x, lr, max_epochs,
                                 1, 0.9, 0.9, 0.999, 1e-8)
    c_hist_a = [cost_function(x) for x in x_hist_a]
    runs_info.append({
        "method": method_a,
        "x_hist": x_hist_a,
        "c_hist": c_hist_a,
        "lr": lr,
        "batch_size": (batch_size_a if method_a=="Mini-Batch Gradient Descent" else None),
        "momentum": (momentum_factor_a if method_a=="Gradient Descent w/ Momentum" else None),
        "beta1": (beta1_a if method_a=="Adam" else None),
        "beta2": (beta2_a if method_a=="Adam" else None),
        "epsilon": (epsilon_a if method_a=="Adam" else None),
    })

    # Method B
    if method_b == "Mini-Batch Gradient Descent":
        x_hist_b = run_optimizer(method_b, initial_x, lr, max_epochs,
                                 batch_size_b, 0.9, 0.9, 0.999, 1e-8)
    elif method_b == "Gradient Descent w/ Momentum":
        x_hist_b = run_optimizer(method_b, initial_x, lr, max_epochs,
                                 1, momentum_factor_b, 0.9, 0.999, 1e-8)
    elif method_b == "Adam":
        x_hist_b = run_optimizer(method_b, initial_x, lr, max_epochs,
                                 1, 0.9, beta1_b, beta2_b, epsilon_b)
    else:
        x_hist_b = run_optimizer(method_b, initial_x, lr, max_epochs,
                                 1, 0.9, 0.9, 0.999, 1e-8)
    c_hist_b = [cost_function(x) for x in x_hist_b]
    runs_info.append({
        "method": method_b,
        "x_hist": x_hist_b,
        "c_hist": c_hist_b,
        "lr": lr,
        "batch_size": (batch_size_b if method_b=="Mini-Batch Gradient Descent" else None),
        "momentum": (momentum_factor_b if method_b=="Gradient Descent w/ Momentum" else None),
        "beta1": (beta1_b if method_b=="Adam" else None),
        "beta2": (beta2_b if method_b=="Adam" else None),
        "epsilon": (epsilon_b if method_b=="Adam" else None),
    })

# -------------------------------------
# 5. Show Early Stopping Results (Text)
# -------------------------------------
st.subheader("Results with Early Stopping")
for run in runs_info:
    x_hist = run["x_hist"]
    c_hist = run["c_hist"]
    final_epoch = len(x_hist)-2  # because we appended final point again
    final_cost = c_hist[-1]
    method_name = run["method"]
    st.write(f"- **{method_name}**: ended at epoch {final_epoch}, final cost={final_cost:.4f}")

# ------------------------
# 6. Build Plotly Animation
# ------------------------
plot_x = np.linspace(-2, 8, 400)
plot_y = cost_function(plot_x)

all_costs = []
for run in runs_info:
    all_costs.extend(run["c_hist"])
y_min, y_max = min(all_costs), max(all_costs)

method_colors = {
    "Batch Gradient Descent": "#1f77b4",
    "Stochastic Gradient Descent": "#ff7f0e",
    "Mini-Batch Gradient Descent": "#2ca02c",
    "Gradient Descent w/ Momentum": "#9467bd",
    "Adam": "#8c564b"
}

base_data = [
    go.Scatter(
        x=plot_x,
        y=plot_y,
        mode='lines',
        line=dict(color="#d62728", width=3),
        name="Cost Function",
        hoverinfo='skip'
    )
]

for run in runs_info:
    m_name = run["method"]
    x0 = run["x_hist"][0]
    y0 = run["c_hist"][0]
    base_data.append(
        go.Scatter(
            x=[x0],
            y=[y0],
            mode='markers+lines',
            line=dict(width=2, color=method_colors[m_name]),
            marker=dict(size=6),
            name=m_name,
            hovertemplate=(
                f"{m_name}<br>"
                "Epoch: 0<br>"
                "x: %{x:.2f}<br>"
                "Cost: %{y:.2f}<extra></extra>"
            ),
        )
    )

max_len = max(len(run["x_hist"]) for run in runs_info)
frames = []

for frame_idx in range(max_len):
    frame_data = []
    frame_annotations = []  # we'll collect "popup" annotations here

    for run in runs_info:
        m_name = run["method"]
        x_hist = run["x_hist"]
        c_hist = run["c_hist"]
        color = method_colors[m_name]

        # If we are beyond the length of that method's path, just show final
        if frame_idx < len(x_hist):
            x_sub = x_hist[:frame_idx+1]
            y_sub = c_hist[:frame_idx+1]
        else:
            x_sub = x_hist
            y_sub = c_hist

        # Create the trace
        frame_data.append(
            go.Scatter(
                x=x_sub,
                y=y_sub,
                mode='markers+lines',
                line=dict(width=2, color=color),
                marker=dict(size=6),
                hovertemplate=(
                    f"{m_name}<br>"
                    f"Epoch: {frame_idx}<br>"
                    "x: %{x:.2f}<br>"
                    "Cost: %{y:.2f}<extra></extra>"
                ),
                showlegend=False
            )
        )

        # If this method is fully done (meaning frame_idx >= last index),
        # we add an annotation "popup" with final details
        if frame_idx >= len(x_hist)-1:
            final_epoch = len(x_hist)-2
            final_cost = c_hist[-1]
            
            # Build your message
            # Example includes: method name, epochs used, final cost, LR, momentum, etc.
            lines = []
            lines.append(f"**{m_name}** finished!")
            lines.append(f"Epoch: {final_epoch}")
            lines.append(f"Final Cost: {final_cost:.4f}")
            # Show relevant hyperparams if set
            if run["lr"] is not None:
                lines.append(f"LR={run['lr']}")
            if run["batch_size"]:
                lines.append(f"Batch Size={run['batch_size']}")
            if run["momentum"]:
                lines.append(f"Momentum={run['momentum']}")
            if run["beta1"]:
                lines.append(f"β1={run['beta1']}")
            if run["beta2"]:
                lines.append(f"β2={run['beta2']}")
            
            text_msg = "<br>".join(lines)

            # We'll place the annotation near the final point
            frame_annotations.append(
                dict(
                    x=x_hist[-1],
                    y=c_hist[-1],
                    xanchor="left",
                    yanchor="bottom",
                    text=text_msg,
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor=color,
                    bordercolor=color,
                    bgcolor="rgba(255,255,255,0.8)",
                    font=dict(color="#000"),
                    ax=30,
                    ay=-10
                )
            )

    frames.append(
        go.Frame(
            name=str(frame_idx),
            data=frame_data,
            layout=go.Layout(annotations=frame_annotations)
        )
    )

# Slider for manual scrubbing
sliders = [
    dict(
        active=0,
        currentvalue={"prefix": "Epoch: "},
        pad={"t": 50},
        steps=[
            dict(
                method="animate",
                args=[
                    [str(k)],
                    {
                        "mode": "immediate",
                        "frame": {"duration": 0, "redraw": True},
                        "transition": {"duration": 0}
                    }
                ],
                label=str(k)
            )
            for k in range(max_len)
        ],
    )
]

# Play/Pause
updatemenus = [
    dict(
        type="buttons",
        direction="left",
        x=0.1, y=1.15,
        xanchor="left", yanchor="top",
        showactive=True,
        buttons=[
            dict(
                label="Play",
                method="animate",
                args=[
                    None,
                    {
                        "frame": {"duration": 300, "redraw": True},
                        "transition": {"duration": 300, "easing": "linear"},
                        "fromcurrent": True
                    },
                ],
            ),
            dict(
                label="Pause",
                method="animate",
                args=[
                    [None],
                    {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0}
                    },
                ],
            ),
        ],
    )
]

if mode == "Single Method":
    title_text = f"{method_selected}"
else:
    title_text = f"{runs_info[0]['method']} vs. {runs_info[1]['method']}"

fig = go.Figure(
    data=base_data,
    layout=go.Layout(
        title=f"Comparison: {title_text}",
        xaxis=dict(title="x", range=[-2, 8]),
        yaxis=dict(
            title="Cost",
            range=[y_min - 1, y_max + 1] if y_min < y_max else [0, 10]
        ),
        sliders=sliders,
        updatemenus=updatemenus,
    ),
    frames=frames
)

st.plotly_chart(fig, use_container_width=True)
