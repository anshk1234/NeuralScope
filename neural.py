import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import graphviz
from time import sleep
import time
import json
from streamlit_lottie import st_lottie

# ---- Page Configuration ----
st.set_page_config(page_title="NeuralScope🧬", layout="centered")

# splash animation
def load_lottiefile(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

if "show_intro" not in st.session_state:
    st.session_state.show_intro = True

if st.session_state.show_intro:
    lottie_intro = load_lottiefile("neural.json")
    splash = st.empty()
    with splash.container():
        st.markdown("<h1 style='text-align:center;'>Welcome to Neuralscope🧬 </h1>", unsafe_allow_html=True)
        st_lottie(lottie_intro, height=280, speed=0.5, loop=True)
        time.sleep(4)
    splash.empty()
    st.session_state.show_intro = False

# --- Sidebar Config ---
st.sidebar.title("🧠 NeuralScope Lab🧬")
dataset_name = st.sidebar.selectbox("Select Dataset", ["moons", "circles", "blobs"])
n_layers = st.sidebar.slider("Hidden Layers", 1, 5, 2)
neurons_per_layer = st.sidebar.slider("Neurons per Layer", 2, 20, 8)
activation_name = st.sidebar.selectbox("Activation", ["ReLU", "Tanh", "Sigmoid"])
lr = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01, step=0.001)
epochs = st.sidebar.slider("Epochs", 10, 500, 100)
seed = st.sidebar.slider("Random Seed", 0, 100, 42)

# --- Sidebar Footer Info ---
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About This App")
st.sidebar.info(
    "This interactive App lets you build and train a neural network on synthetic datasets. "
    "You can visualize decision boundaries, weight heatmaps, and forward activations in real time."
)

st.sidebar.markdown("### 🧠 What is a Neural Network?")
st.sidebar.write(
    "A neural network is a machine learning model inspired by the human brain. "
    "It learns patterns from data by adjusting weights through backpropagation. "
    "This app uses PyTorch to build and train fully connected feedforward networks."
)

st.sidebar.markdown("### 🙌 Credits")
st.sidebar.markdown("""
- 👨‍💻 Developed : [Ansh kunwar](https://share.streamlit.io/user/anshk1234)
- 🧪 Powered :**Streamlit**, **PyTorch**, and **Matplotlib**
- 📚 Data source : the open-source ML community
- 💡 Source code : [GITHUB](https://github.com/anshk1234/NeuralScope)
- 📧 contact: anshkunwar3009@gmail.com
- 🌐 see other projects: [streamlit.io/ansh kunwar](https://share.streamlit.io/user/anshk1234)
- THIS APP IS LICENSED UNDER Apache 2.0 LICENSE              

  © 2025 NeuralScope 🧬
""")
    


# --- Dataset Generator ---
np.random.seed(seed)
if dataset_name == "moons":
    X, y = make_moons(n_samples=500, noise=0.2, random_state=seed)
elif dataset_name == "circles":
    X, y = make_circles(n_samples=500, noise=0.1, factor=0.5, random_state=seed)
else:
    X, y = make_blobs(n_samples=500, centers=2, cluster_std=1.5, random_state=seed)

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

# --- Activation Mapping ---
activations = {
    "ReLU": nn.ReLU(),
    "Tanh": nn.Tanh(),
    "Sigmoid": nn.Sigmoid()
}
activation_fn = activations[activation_name]

# --- Model Builder ---
layers = [nn.Linear(2, neurons_per_layer), activation_fn]
for _ in range(n_layers - 1):
    layers += [nn.Linear(neurons_per_layer, neurons_per_layer), activation_fn]
layers += [nn.Linear(neurons_per_layer, 1), nn.Sigmoid()]
model = nn.Sequential(*layers)

# --- Utility Functions ---
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid, dtype=torch.float32)
    with torch.no_grad():
        preds = model(grid_tensor).numpy().reshape(xx.shape)
    plt.contourf(xx, yy, preds, levels=[0, 0.5, 1], alpha=0.6, cmap="RdBu")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolor="white")
    plt.title("Decision Boundary")

def draw_network(input_dim, hidden_layers, output_dim):
    dot = graphviz.Digraph()
    dot.attr(rankdir='LR')
    for i in range(input_dim):
        dot.node(f"x{i+1}", f"x{i+1}", shape="circle", style="filled", color="lightgreen")
    for l, n in enumerate(hidden_layers):
        for j in range(n):
            dot.node(f"h{l}_{j}", f"h{j}", shape="circle", style="filled", color="lightblue")
            if l == 0:
                for i in range(input_dim):
                    dot.edge(f"x{i+1}", f"h{l}_{j}")
            else:
                for k in range(hidden_layers[l-1]):
                    dot.edge(f"h{l-1}_{k}", f"h{l}_{j}")
    dot.node("y", "y", shape="circle", style="filled", color="salmon")
    for j in range(hidden_layers[-1]):
        dot.edge(f"h{len(hidden_layers)-1}_{j}", "y")
    return dot

def show_weights(model):
    for i, layer in enumerate(model):
        if isinstance(layer, nn.Linear):
            st.write(f"**Layer {i} Weights**")
            st.write(layer.weight.detach().numpy())

def plot_weight_heatmaps(model):
    for i, layer in enumerate(model):
        if isinstance(layer, nn.Linear):
            weights = layer.weight.detach().numpy()
            fig, ax = plt.subplots()
            im = ax.imshow(weights, cmap="coolwarm", aspect="auto")
            ax.set_title(f"Layer {i} Heatmap")
            fig.colorbar(im, ax=ax)
            st.pyplot(fig)

def simulate_forward_pass(model, input_vector):
    x = torch.tensor(input_vector, dtype=torch.float32).view(1, -1)
    for i, layer in enumerate(model):
        x = layer(x)
        if isinstance(layer, nn.Linear):
            st.write(f"Layer {i} output: {x.detach().numpy()}")
        elif isinstance(layer, nn.Module):
            st.write(f"Activation {i}: {x.detach().numpy()}")

# --- Main Page ---
st.title("🧪 NeuralScope Lab 🧬")

# --- Tabs Layout ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview + Training",
    "🧬 Network Architecture",
    "📋 Raw Weights",
    "🌡️ Weight Heatmaps",
    "⚡ Forward Pass"
])

with tab1:
    st.subheader("📊 Model Overview")
    st.write(f"**Dataset:** {dataset_name.capitalize()}")
    st.write(f"**Activation Function:** {activation_name}")
    st.write(f"**Architecture:** {n_layers} hidden layers × {neurons_per_layer} neurons")

    st.subheader("📈 Training Visualization")
    placeholder = st.empty()
    losses = []
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch % 10 == 0 or epoch == epochs - 1:
            with placeholder.container():
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Decision Boundary (Epoch {epoch})**")
                    fig1 = plt.figure()
                    plot_decision_boundary(model, X, y)
                    st.pyplot(fig1)
                with col2:
                    st.write("**Loss Curve**")
                    fig2 = plt.figure()
                    plt.plot(losses)
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.title("Training Loss")
                    st.pyplot(fig2)
            sleep(0.05)

with tab2:
    st.subheader("🧬 Network Architecture")
    hidden_config = [neurons_per_layer] * n_layers
    st.graphviz_chart(draw_network(2, hidden_config, 1))

with tab3:
    st.subheader("📋 Raw Weights")
    show_weights(model)

with tab4:
    st.subheader("🌡️ Weight Magnitudes (Heatmaps)")
    plot_weight_heatmaps(model)

with tab5:
    st.subheader("⚡ Forward Pass Activation Snapshot")
    simulate_forward_pass(model, X_train[0])

# ---- Footer ----
st.markdown("<p style='text-align:center; color:white;'>© 2025 NeuralScope | Powered by Neural Network🧬</p>", unsafe_allow_html=True)
