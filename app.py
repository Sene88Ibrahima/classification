#!/usr/bin/env python
# coding: utf-8

# Importation des bibliothèques nécessaires
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# --- Fonction de clustering ---
def perform_clustering(data):
    # Séparer les étiquettes des fonctionnalités
    features = data.drop(['legitimate'], axis=1)

    # --- Standardisation des données ---
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # --- Réduction des dimensions avec PCA ---
    pca = PCA(n_components=3)
    reduced_features = pca.fit_transform(scaled_features)

    # --- Sous-ensemble pour DBSCAN et Agglomerative Clustering ---
    n_subset = 5000
    subset_features = reduced_features[:n_subset]

    # --- KMeans Clustering ---
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels_kmeans = kmeans.fit_predict(reduced_features)

    # --- DBSCAN Clustering ---
    try:
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        labels_dbscan = dbscan.fit_predict(subset_features)
    except MemoryError:
        labels_dbscan = "Erreur de mémoire : DBSCAN non exécuté"

    # --- Agglomerative Clustering ---
    agg_clustering = AgglomerativeClustering(n_clusters=3)
    labels_agg = agg_clustering.fit_predict(subset_features)

    # --- Scores silhouette ---
    silhouette_kmeans = silhouette_score(reduced_features, labels_kmeans)

    if isinstance(labels_dbscan, str):
        silhouette_dbscan = "Non applicable (Erreur de mémoire)"
    elif len(set(labels_dbscan)) > 1:
        silhouette_dbscan = silhouette_score(subset_features, labels_dbscan)
    else:
        silhouette_dbscan = "Non applicable (1 cluster ou bruit uniquement)"

    silhouette_agg = silhouette_score(subset_features, labels_agg)

    return {
        "silhouette_scores": {
            "KMeans": silhouette_kmeans,
            "DBSCAN": silhouette_dbscan,
            "Agglomerative": silhouette_agg
        },
        "reduced_features": reduced_features,
        "subset_features": subset_features,
        "labels": {
            "KMeans": labels_kmeans,
            "DBSCAN": labels_dbscan,
            "Agglomerative": labels_agg
        }
    }

# --- Fonction pour visualiser les clusters ---
def plot_clusters(title, x, y, labels, palette, xlabel="Dimension PCA 1", ylabel="Dimension PCA 2"):
    """Affiche les clusters en 2D."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x, y=y, hue=labels, palette=palette, style=labels, s=50, edgecolor="k")
    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(title="Cluster", loc="best")
    plt.tight_layout()
    st.pyplot(plt)

# --- Interface Streamlit ---
st.title("Application de Clustering avec Upload de Fichier")
st.write("Veuillez charger un fichier CSV pour exécuter le clustering.")

# Upload du fichier
uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")

if uploaded_file is not None:
    # Lecture du fichier CSV
    data = pd.read_csv(uploaded_file, sep=';')
    st.write("Aperçu des données chargées :")
    st.write(data.head())

    # Vérification de la colonne 'legitimate'
    if 'legitimate' not in data.columns:
        st.error("Le fichier doit contenir une colonne 'legitimate'.")
    else:
        # Exécution du clustering
        clustering_results = perform_clustering(data)

        # Affichage des scores silhouette
        st.write("Scores de silhouette :")
        st.write(clustering_results["silhouette_scores"])

        # Visualisation des résultats
        reduced_features = clustering_results["reduced_features"]
        subset_features = clustering_results["subset_features"]
        labels = clustering_results["labels"]

        st.write("Visualisation des clusters KMeans :")
        plot_clusters(
            "KMeans Clustering Visualization",
            reduced_features[:, 0],
            reduced_features[:, 1],
            labels["KMeans"],
            palette="viridis"
        )

        st.write("Visualisation des clusters DBSCAN :")
        plot_clusters(
            "DBSCAN Clustering Visualization",
            subset_features[:, 0],
            subset_features[:, 1],
            labels["DBSCAN"],
            palette="deep"
        )

        st.write("Visualisation des clusters Agglomerative :")
        plot_clusters(
            "Agglomerative Clustering Visualization",
            subset_features[:, 0],
            subset_features[:, 1],
            labels["Agglomerative"],
            palette="coolwarm"
        )
