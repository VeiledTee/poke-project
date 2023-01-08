import math
import sys
from typing import List

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold, cross_val_predict, train_test_split

plt.style.use("bmh")
plt.rcParams["font.size"] = 15
plt.rcParams["scatter.edgecolors"] = "black"

USE_KMEANS = True
USE_DBSCAN = True
VERBOSE = False
LEGENDARY = True
MAX_LENGTH_OUTPUT = 50

TYPES = [
    "Normal",
    "Fire",
    "Water",
    "Electric",
    "Grass",
    "Ice",
    "Fighting",
    "Poison",
    "Ground",
    "Flying",
    "Psychic",
    "Bug",
    "Rock",
    "Ghost",
    "Dragon",
    "Dark",
    "Steel",
    "Fairy",
]
COLOURS = [
    "#EE8130",
    "#6390F0",
    "#F7D02C",
    "#7AC74C",
    "#96D9D6",
    "#C22E28",
    "#A33EA1",
    "#E2BF65",
    "#A98FF3",
    "#F95587",
    "#A6B91A",
    "#B6A136",
    "#735797",
    "#6F35FC",
    "#705746",
    "#B7B7CE",
    "#D685AD",
]

type_colors = dict()
type_colors["Normal"] = "#A8A77A"
type_colors["Fire"] = "#EE8130"
type_colors["Water"] = "#6390F0"
type_colors["Electric"] = "#F7D02C"
type_colors["Grass"] = "#7AC74C"
type_colors["Ice"] = "#96D9D6"
type_colors["Fighting"] = "#C22E28"
type_colors["Poison"] = "#A33EA1"
type_colors["Ground"] = "#E2BF65"
type_colors["Flying"] = "#A98FF3"
type_colors["Psychic"] = "#F95587"
type_colors["Bug"] = "#A6B91A"
type_colors["Rock"] = "#B6A136"
type_colors["Ghost"] = "#735797"
type_colors["Dragon"] = "#6F35FC"
type_colors["Dark"] = "#705746"
type_colors["Steel"] = "#B7B7CE"
type_colors["Fairy"] = "#D685AD"


def pokemon_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the names of pokemon in a pokemon dataset. Formats all names to [Form] [Name]
    :param df: Dataframe of pokemon information, the index must be the name of the pokemon
    :return: Cleaned dataframe
    """
    df.index = df.index.str.replace(".*(?=Mega)", "", regex=True)
    df.index = df.index.str.replace(".*(?=Galarian)", "", regex=True)
    df.index = df.index.str.replace(".*(?=Alolan)", "", regex=True)
    df.index = df.index.str.replace(".*(?=Primal)", "", regex=True)
    df["total"] = (
        df["hp"]
        + df["attack"]
        + df["defense"]
        + df["sp_attack"]
        + df["sp_defense"]
        + df["speed"]
    )
    df["total"] = df["total"].astype("int")
    df = df.fillna("")
    numerals = [
        ["I", 1],
        ["II", 2],
        ["III", 3],
        ["IV", 4],
        ["V", 5],
        ["VI", 6],
        ["VII", 7],
        ["VIII", 8],
    ]
    for pair in numerals:
        df["gen"] = np.where(df["gen"] == pair[0], pair[1], df["gen"])
    return df


def make_type_dicts():
    type_colors = {
        0: "#A8A77A",
        1: "#EE8130",
        2: "#6390F0",
        3: "#F7D02C",
        4: "#7AC74C",
        5: "#96D9D6",
        6: "#C22E28",
        7: "#A33EA1",
        8: "#E2BF65",
        9: "#A98FF3",
        10: "#F95587",
        11: "#A6B91A",
        12: "#B6A136",
        13: "#735797",
        14: "#6F35FC",
        15: "#705746",
        16: "#B7B7CE",
        17: "#D685AD",
    }
    return type_colors


def ClusterIndicesNumpy(clustNum, labels_array):  # numpy
    return np.where(labels_array == clustNum)[0]


def sort_cluster(cluster_df, original_df):
    cluster_types = {}
    num_in_cluster = len(list(cluster_df.index))
    for mon in list(cluster_df.index):  # loop though cluster mons and count types
        primary = original_df.loc[mon, "primary_type"]
        secondary = original_df.loc[mon, "secondary_type"]

        if primary in cluster_types.keys():
            cluster_types[primary] += 1
        else:
            cluster_types[primary] = 1

        if secondary in cluster_types.keys():
            cluster_types[secondary] += 1
        elif secondary != "":
            cluster_types[secondary] = 1
    for poke_type in TYPES:
        if poke_type.lower() not in cluster_types.keys():
            cluster_types[poke_type.lower()] = 0

    sorted_types = list(
        sorted(cluster_types.items(), key=lambda item: item[0].lower())
    )  # sort types by type of pokemon in cluster
    sorted_counts = list(
        sorted(cluster_types.items(), key=lambda item: item[1], reverse=True)
    )  # sort types by number of pokemon in cluster

    frequency = sorted_counts[0][1] / num_in_cluster if num_in_cluster > 0 else 0
    return sorted_types, frequency


def typify_clusters(cluster_evals):
    """
    Given list of types and counts per cluster, ensure all clusters have one type, and the cluster with the max count of each type is assigned that type
    """
    final_types = ["" for _ in range(18)]
    assigned_clusters = []
    for i, poke_type in enumerate(list(sorted(TYPES))):  # loop through type
        max_index = 0
        max_value = 0
        for j, cluster in enumerate(cluster_evals):
            if cluster[i][1] > max_value and j not in assigned_clusters:
                max_value = cluster[i][1]
                max_index = j
        assigned_clusters.append(max_index)
        final_types[max_index] = poke_type.lower()
    for a_type in TYPES:
        if a_type.lower() not in final_types:
            final_types[final_types.index("")] = a_type.lower()
    return final_types  # evaluation with external knowledge


def cluster_purity(cluster_labels, cluster_class):
    correct = 0
    for i in cluster_labels:
        print(i)
        if i == cluster_class:
            correct += 1
    return correct / len(cluster_labels)


def cluster_cohesion(cluster_x_values, cluster_y_values, cluster_center):
    distances = [
        math.dist([cluster_x_values[index], cluster_y_values[index]], cluster_center)
        for index in range(len(cluster_x))
    ]
    return sum(distances) / len(distances) if len(distances) > 0 else np.inf


if __name__ == "__main__":
    classes = list(type_colors.keys())
    class_colours = list(type_colors.values())

    if len(sys.argv) == 2:
        pokemon = pd.read_csv(sys.argv[1], encoding='utf-16')
        custom = False
    else:
        pokemon = pd.read_csv(sys.argv[1], encoding='utf-16')
        custom = sys.argv[2]
    pokemon = pokemon_cleaning(pokemon.set_index("english_name"))

    if LEGENDARY:
        pokemon = pokemon[pokemon["is_legendary"] == 0]  # drop legendaries
    to_drop = [
        "national_number",
        "gen",
        "japanese_name",
        "primary_type",
        "secondary_type",
        "classification",
        "percent_male",
        "percent_female",
        "height_m",
        "weight_kg",
        "capture_rate",
        "base_egg_steps",
        "abilities_0",
        "abilities_1",
        "abilities_2",
        "abilities_hidden",
        "is_sublegendary",
        "is_legendary",
        "is_mythical",
        "evochain_0",
        "evochain_1",
        "evochain_2",
        "evochain_3",
        "evochain_4",
        "evochain_5",
        "evochain_6",
        "gigantamax",
        "mega_evolution",
        "mega_evolution_alt",
        "description",
        "total",
    ]
    pokemon_dropped = pokemon.drop(columns=to_drop)
    pokemon_2d = TSNE(n_components=2).fit_transform(pokemon_dropped)
    colours = make_type_dicts()

    S = 5  # number of splits in k-fold
    C = 18  # number of clusters, one for each pokemon type
    if USE_KMEANS:
        plt.figure(0, figsize=(20, 12))

        k_means = KMeans(n_clusters=C).fit(pokemon_dropped)
        labels = k_means.labels_
        pokemon["cluster"] = labels
        print(
            f"K-Means Silhouette score: {metrics.silhouette_score(pokemon_dropped, pokemon['cluster'], metric='euclidean')}"
        )
        assigned_type = []
        purities = []
        x = []
        y = []

        for i in range(C):  # determine type of each cluster
            cluster = pokemon[pokemon["cluster"] == i]
            cluster_tsne = pokemon_2d[pokemon["cluster"] == i]
            cluster_type, purity = sort_cluster(cluster, pokemon)
            assigned_type.append(cluster_type)
            purities.append(purity)
            x.append([j[0] for j in cluster_tsne])
            y.append([j[1] for j in cluster_tsne])

        typed_clusters = typify_clusters(assigned_type)

        centroids = []
        cohesion = []
        for i in range(C):  # build cluster centers
            cluster_x = pokemon_2d[pokemon["cluster"] == i][:, 0]
            cluster_y = pokemon_2d[pokemon["cluster"] == i][:, 1]
            center_x = sum(cluster_x) / len(cluster_x)
            center_y = sum(cluster_y) / len(cluster_y)
            centroids.append([center_x, center_y])
            cohesion.append(cluster_cohesion(cluster_x, cluster_y, centroids[-1]))
            if VERBOSE:
                print(
                    f"Purity of {typed_clusters[typed_clusters.index(classes[i].lower())].capitalize()} Cluster: {str(purities[typed_clusters.index(classes[i].lower())].__round__(5)).rjust(MAX_LENGTH_OUTPUT - len(f'Purity of {typed_clusters[typed_clusters.index(classes[i].lower())].capitalize()} Cluster:'), ' ')}"
                )
                print(
                    f"Cohesion of {classes[i]} Cluster: {str(cluster_cohesion(cluster_x, cluster_y, centroids[-1]).__round__(5)).rjust(MAX_LENGTH_OUTPUT - len(f'Cohesion of {classes[i]} Cluster:'), ' ')}\n"
                )
        print(
            f"K-Means average purity over {C} Clusters: {str(float(sum(purities)/ len(purities)).__round__(5)).rjust(MAX_LENGTH_OUTPUT - len(f'K-Means average purity over {C} Clusters:'), ' ')}"
        )
        print(
            f"K-Means average cohesion over {C} Clusters: {str(float(sum(cohesion) / len(cohesion)).__round__(5)).rjust(MAX_LENGTH_OUTPUT - len(f'K-Means average cohesion over {C} Clusters:'), ' ')}\n"
        )

        plt.scatter(
            [i[0] for i in pokemon_2d],
            [i[1] for i in pokemon_2d],
            c=[type_colors[typed_clusters[i].capitalize()] for i in labels],
        )
        plt.scatter(
            [c[0] for c in centroids],
            [c[1] for c in centroids],
            marker="X",
            s=375,
            linewidths=1,
            edgecolor="k",
            c=[type_colors[typed_clusters[i].capitalize()] for i in range(C)],
        )

        recs = []
        for i in range(len(class_colours)):
            recs.append(mpatches.Rectangle((0, 0), 0.25, 0.25, fc=class_colours[i]))
        plt.legend(recs, classes, prop={"size": 9})

        if LEGENDARY:
            plt.title(f"K-Means Clustering with {C} types of all Pokemon")
        else:
            plt.title(f"K-Means Clustering with {C} types of non-Legendary Pokemon")
        plt.xlabel("K-Means x value")
        plt.ylabel("K-Means y value")
        plt.savefig(f"pokemon_k_means_{C}_clusters.png")

    if USE_DBSCAN:
        graph_num = 1 if USE_KMEANS else 0
        plt.figure(graph_num, figsize=(20, 12))

        if custom:
            db = DBSCAN(eps=19.175, min_samples=4).fit(pokemon_dropped)
        else:
            db = DBSCAN().fit(pokemon_dropped)
        labels = db.labels_
        pokemon["cluster"] = labels
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print(n_clusters_)

        if n_clusters_ > 1:
            print(
                f"DBSCAN Silhouette score: {metrics.silhouette_score(pokemon_dropped, pokemon['cluster'], metric='euclidean')}"
            )

        assigned_type = []
        purities = []
        x = []
        y = []

        for i in range(C):  # determine type of each cluster
            cluster = pokemon[pokemon["cluster"] == i]
            cluster_tsne = pokemon_2d[pokemon["cluster"] == i]
            cluster_type, purity = sort_cluster(cluster, pokemon)
            assigned_type.append(cluster_type)
            purities.append(purity)
            x.append([j[0] for j in cluster_tsne])
            y.append([j[1] for j in cluster_tsne])

        typed_clusters = typify_clusters(assigned_type)[:n_clusters_]

        centroids = []
        cohesion = []
        for i in range(C):  # build cluster centers
            cluster_x = pokemon_2d[pokemon["cluster"] == i][:, 0]
            cluster_y = pokemon_2d[pokemon["cluster"] == i][:, 1]
            center_x = sum(cluster_x) / len(cluster_x) if len(cluster_x) > 0 else 0
            center_y = sum(cluster_y) / len(cluster_y) if len(cluster_y) > 0 else 0
            centroids.append([center_x, center_y])
            cohesion.append(cluster_cohesion(cluster_x, cluster_y, centroids[-1]))
            if VERBOSE:
                print(
                    f"Purity of {typed_clusters[typed_clusters.index(classes[i].lower())].capitalize()} Cluster: {str(purities[typed_clusters.index(classes[i].lower())].__round__(5)).rjust(MAX_LENGTH_OUTPUT - len(f'Purity of {typed_clusters[typed_clusters.index(classes[i].lower())].capitalize()} Cluster:'), ' ')}"
                )
                print(
                    f"Cohesion of {classes[i]} Cluster: {str(cluster_cohesion(cluster_x, cluster_y, centroids[-1]).__round__(5)).rjust(MAX_LENGTH_OUTPUT - len(f'Cohesion of {classes[i]} Cluster:'), ' ')}\n"
                )

        print(
            f"DBSCAN average purity over {n_clusters_} Clusters: {str(float(sum(purities)/ len(purities)).__round__(5)).rjust(MAX_LENGTH_OUTPUT - len(f'DBSCAN average purity over {n_clusters_} Clusters:'), ' ')}"
        )
        print(
            f"DBSCAN average cohesion over {n_clusters_} Clusters: {str(float(sum(cohesion) / len(cohesion)).__round__(5)).rjust(MAX_LENGTH_OUTPUT - len(f'DBSCAN average cohesion over {n_clusters_} Clusters:'), ' ')}\n"
        )

        scatter_colours = []
        for i in labels:
            if i == -1:
                scatter_colours.append("#000000")
            else:
                scatter_colours.append(type_colors[typed_clusters[i].capitalize()])
        plt.scatter(
            [i[0] for i in pokemon_2d], [i[1] for i in pokemon_2d], c=scatter_colours
        )
        plt.scatter(
            [c[0] for c in centroids],
            [c[1] for c in centroids],
            marker="X",
            s=375,
            linewidths=1,
            edgecolor="k",
            c=[type_colors[typed_clusters[i].capitalize()] for i in range(n_clusters_)],
        )

        recs = []
        db_classes = []
        for i in typed_clusters:
            i = i.capitalize()
            recs.append(
                mpatches.Rectangle(
                    (0, 0), 0.25, 0.25, fc=class_colours[classes.index(i)]
                )
            )
            db_classes.append(i)
        recs.append(mpatches.Rectangle((0, 0), 0.25, 0.25, fc="#000000"))
        db_classes.append("No Cluster")
        plt.legend(recs, db_classes, prop={"size": 9})

        if LEGENDARY:
            plt.title(f"DBSCAN Clustering with {n_clusters_} types of all Pokemon")
        else:
            plt.title(f"DBSCAN Clustering with {n_clusters_} types of non-Legendary Pokemon")
        plt.xlabel("DBSCAN x value")
        plt.ylabel("DBSCAN y value")
        plt.savefig(f"pokemon_dbscan_{n_clusters_}_clusters.png")

    plt.show()
