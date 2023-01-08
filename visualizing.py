import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

plt.style.use('bmh')
plt.rcParams['font.size'] = 15

pd.set_option('display.max_columns', None)


def make_type_dicts():
    type_colors = dict()
    type_colors['Normal'] = '#A8A77A'
    type_colors['Fire'] = '#EE8130'
    type_colors['Water'] = '#6390F0'
    type_colors['Electric'] = '#F7D02C'
    type_colors['Grass'] = '#7AC74C'
    type_colors['Ice'] = '#96D9D6'
    type_colors['Fighting'] = '#C22E28'
    type_colors['Poison'] = '#A33EA1'
    type_colors['Ground'] = '#E2BF65'
    type_colors['Flying'] = '#A98FF3'
    type_colors['Psychic'] = '#F95587'
    type_colors['Bug'] = '#A6B91A'
    type_colors['Rock'] = '#B6A136'
    type_colors['Ghost'] = '#735797'
    type_colors['Dragon'] = '#6F35FC'
    type_colors['Dark'] = '#705746'
    type_colors['Steel'] = '#B7B7CE'
    type_colors['Fairy'] = '#D685AD'

    normal = [1, 2, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    fighting = [1, 1, 2, 1, 1, 0.5, 0.5, 1, 1, 1, 1, 1, 1, 2, 1, 1, 0.5, 2]
    flying = [1, 0.5, 1, 1, 0, 2, 0.5, 1, 1, 1, 1, 0.5, 2, 1, 2, 1, 1, 1, 1]
    poison = [1, 0.5, 1, 0.5, 2, 1, 0.5, 1, 1, 1, 1, 0.5, 1, 2, 1, 1, 1, 0.5]
    ground = [1, 1, 1, 0.5, 1, 0.5, 1, 1, 1, 1, 2, 2, 0, 1, 2, 1, 1, 1]
    rock = [0.5, 2, 0.5, 0.5, 2, 1, 1, 1, 2, 0.5, 2, 2, 1, 1, 1, 1, 1, 1]
    bug = [1, 0.5, 2, 1, 0.5, 2, 1, 1, 1, 2, 1, 0.5, 1, 1, 1, 1, 1, 1]
    ghost = [0, 0, 1, 0.5, 1, 1, 0.5, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1]
    steel = [0.5, 2, 0.5, 0, 2, 0.5, 0.5, 1, 0.5, 2, 1, 0.5, 1, 0.5, 0.5, 0.5, 1, 0.5]
    fire = [1, 1, 1, 1, 2, 2, 0.5, 1, 0.5, 0.5, 2, 0.5, 1, 1, 0.5, 1, 1, 0.5]
    water = [1, 1, 1, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 2, 2, 1, 0.5, 1, 1, 1]
    grass = [1, 1, 2, 2, 0.5, 1, 2, 1, 1, 2, 0.5, 0.5, 0.5, 1, 2, 1, 1, 1]
    electric = [1, 1, 0.5, 1, 2, 1, 1, 1, 0.5, 1, 1, 1, 0.5, 1, 1, 1, 1, 1]
    psychic = [1, 0.5, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 0.5, 1, 1, 2, 1]
    ice = [1, 2, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 0.5, 1, 1, 1]
    dragon = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 1, 2, 2, 1, 2]
    dark = [1, 2, 1, 1, 1, 1, 2, 0.5, 1, 1, 1, 1, 1, 0, 1, 1, 0.5, 2]
    fairy = [1, 0.5, 1, 2, 1, 1, 0.5, 1, 2, 1, 1, 1, 1, 1, 1, 0, 0.5, 1]

    matchup_dict = {'Normal': normal, 'Fighting': fighting, 'Flying': flying, 'Poison': poison, 'Ground': ground, 'Rock': rock, 'Bug': bug, 'Ghost': ghost,
                 'Steel': steel, 'Fire': fire, 'Water': water, 'Grass': grass, 'Electric': electric, 'Psychic': psychic, 'Ice': ice, 'Dragon': dragon, 'Dark': dark,
                 'Fairy': fairy}
    return matchup_dict, type_colors


def generate_new_data(df: pd.DataFrame, file_name: str = 'smol_pokemon.csv', size: int = 100):
    to_save = df.sample(size)
    to_save.to_csv(file_name, sep=',', encoding='utf-16', index=False)


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
    df['total'] = df['hp'] + df['attack'] + df['defense'] + df['sp_attack'] + df['sp_defense'] + df['speed']
    # df['total'] = pd.to_numeric(df['total'])
    df['total'] = df['total'].astype('int')
    df = df.fillna('')
    return df


def metrics(df: pd.DataFrame, rmm_attributes: List[str]) -> pd.DataFrame:
    df_rmm: pd.DataFrame = df[rmm_attributes]
    df_output: pd.DataFrame = pd.DataFrame(index=['Range', 'Mean', 'Mode', 'St.Dev', 'IQR'], columns=rmm_attributes)
    for attribute in list(df_rmm.columns):
        df_output.at['Range', attribute] = df_rmm[attribute].min(), df_rmm[attribute].max()
        df_output.at['Mean', attribute] = df_rmm[attribute].mean(numeric_only=True)
        df_output.at['Mode', attribute] = df_rmm[attribute].mode()
        df_output.at['St.Dev', attribute] = df_rmm[attribute].std()
        df_output.at['IQR', attribute] = df_rmm[attribute].quantile(0.75) - df_rmm[attribute].quantile(0.25)
    print(df_output)
    return df_output


def plot_count(df: pd.DataFrame, x: str, y: str, incl_legendary: bool = True, legendary_only: bool = False, title: str = '', graph_num=1):
    # fig set up
    plt.figure(graph_num, figsize=(8, 8))
    plt.title(title, fontsize=25, fontweight='bold')
    plt.xlabel(f"{x.capitalize()}", fontweight='bold')
    plt.ylabel(f"Legendary Count", fontweight='bold')
    # filter df
    if not incl_legendary:
        df = df[(df["is_legendary"] == 0) | (df["is_mythical"] == 0) | (df["is_sublegendary"] == 0)]
    if legendary_only:
        df = df[(df["is_legendary"] == 1) | (df["is_mythical"] == 1) | (df["is_sublegendary"] == 1)]
    # find x and y
    x_unique = sorted(list(df[x].unique()))
    y_unique = list(df.groupby(x).count()[y])
    # plot
    plt.bar(x_unique, y_unique)


def plot_continuous(df: pd.DataFrame, x: str, y: str, incl_legendary: bool = True, legendary_only: bool = False, title: str = '', graph_num=1):
    # fig set up
    plt.figure(graph_num, figsize=(15, 8))
    plt.title(title, fontsize=25, fontweight='bold')
    plt.xlabel(f"{x.capitalize()}", fontweight='bold')
    plt.ylabel(f"{y} Count", fontweight='bold')
    if not incl_legendary:
        df = df[(df["is_legendary"] == 0) | (df["is_mythical"] == 0) | (df["is_sublegendary"] == 0)]
    if legendary_only:
        df = df[(df["is_legendary"] == 1) | (df["is_mythical"] == 1) | (df["is_sublegendary"] == 1)]
    grouped_by_x = list(df.groupby(x))
    y_unique = []
    labels = []
    for frame in grouped_by_x:
        labels.append(f"Gen {frame[0]}")
        y_unique.append(frame[1][y].mean())
    plt.pie(y_unique, labels=labels)


def mega(df: pd.DataFrame, x: str, incl_legendary: bool = True, legendary_only: bool = False, title: str = '', graph_num=1):
    # fig set up
    plt.figure(graph_num, figsize=(10, 8))
    plt.title(title, fontsize=25, fontweight='bold')
    if not incl_legendary:
        df = df[(df["is_legendary"] == 0) | (df["is_mythical"] == 0) | (df["is_sublegendary"] == 0)]
    if legendary_only:
        df = df[(df["is_legendary"] == 1) | (df["is_mythical"] == 1) | (df["is_sublegendary"] == 1)]
    df = df[(df['mega_evolution'] != '') | (df['gigantamax'] != '')]
    grouped_by_x = list(df.groupby(x))
    y_unique = []
    labels = []
    explode = []
    for gen, frame in enumerate(grouped_by_x):
        labels.append(f"Gen {frame[0]}")
        y_unique.append(len(frame[1]))
        if gen == 3 or gen == 6:
            explode.append(0.1)
        else:
            explode.append(0)
    plt.rcParams['font.size'] = 15
    _, _, auto = plt.pie(y_unique, labels=labels, explode=explode, shadow=True, autopct=lambda value: display_count(value, y_unique), startangle=90)
    plt.setp(auto, **{'color': 'white', 'weight': 'bold', 'fontsize': 12.5})


def display_count(x, counts):
    return int(np.round((x / 100) * (sum(counts)), 0))


def plot_type_comparison(df: pd.DataFrame, x: str, y: List[str], incl_legendary: bool = True, legendary_only: bool = False, title: str = '', graph_num=1):
    # fig set up
    plt.figure(graph_num, figsize=(15, 8))
    plt.title(title, fontsize=25, fontweight='bold')
    plt.xlabel(f"Primary Type", fontweight='bold')
    plt.ylabel(f"Average Value", fontweight='bold')

    if not incl_legendary:
        df = df[(df["is_legendary"] == 0) | (df["is_mythical"] == 0) | (df["is_sublegendary"] == 0)]
    if legendary_only:
        df = df[(df["is_legendary"] == 1) | (df["is_mythical"] == 1) | (df["is_sublegendary"] == 1)]
    x_unique = [type.capitalize() for type in sorted(list(df[x].unique()))]

    x_axis = np.arange(len(x_unique))
    width = 0.2
    locs, x_labels = plt.xticks()
    plt.xticks(np.arange(0, len(x_unique), step=1))
    plt.xticks(x_axis + 2 * width, x_unique, fontsize=11)

    grouped_by_x = list(df.groupby(x))
    y_unique_1 = []
    y_unique_2 = []
    y_unique_3 = []
    y_unique_4 = []
    for frame in grouped_by_x:
        y_unique_1.append(frame[1][y[0]].mean())
        y_unique_2.append(frame[1][y[1]].mean())
        if len(y) >= 3:
            y_unique_3.append(frame[1][y[2]].mean())
        if len(y) == 4:
            y_unique_4.append(frame[1][y[3]].mean())

    bar1 = plt.bar(x_axis, y_unique_1, color='yellow', label="Attack", edgecolor='black', width=width, align='edge')
    bar2 = plt.bar(x_axis + width, y_unique_2, color='orange', label="Defence", edgecolor='black', width=width, align='edge')
    if y_unique_3:
        bar3 = plt.bar(x_axis + 2 * width, y_unique_3, color='cyan', label="Sp. Atk", edgecolor='black', width=width, align='edge')
    if y_unique_4:
        bar4 = plt.bar(x_axis + 3 * width, y_unique_4, color='purple', label="Sp. Def", edgecolor='black', width=width, align='edge')
    if y_unique_3 != [] and y_unique_4 != []:
        plt.legend((bar1, bar2, bar3, bar4), (y[0], y[1], y[2], y[3]), loc='upper left', fontsize=12)
    elif y_unique_3 != []:
        plt.legend((bar1, bar2, bar3), (y[0], y[1], y[2]))
    else:
        plt.legend((bar1, bar2), (y[0], y[1]))


def pc_box_plot(df: pd.DataFrame, x: str, y: str, incl_legendary: bool = False, legendary_only: bool = False, title: str = '', graph_num=1):
    plt.figure(graph_num, figsize=(10, 5))
    plt.title(title, fontsize=25, fontweight='bold')
    plt.xlabel(f"{x.capitalize()}", fontweight='bold')
    plt.ylabel(f"Stat Totals", fontweight='bold')
    ax = plt.gca()
    ax.set_ylim([150, 750])

    if not incl_legendary:
        df = df[(df["is_legendary"] == 0) & (df["is_mythical"] == 0) & (df["is_sublegendary"] == 0)]
    if legendary_only:
        df = df[(df["is_legendary"] == 1) | (df["is_mythical"] == 1) | (df["is_sublegendary"] == 1)]
    grouped_by_x = list(df.groupby(x))
    y_unique = []
    labels = []
    for frame in grouped_by_x:
        labels.append(f"Gen {frame[0]}")
        y_unique.append(list(frame[1][y]))
    plt.boxplot(y_unique)


def type_chart(df: pd.DataFrame, name_or_type: str = 'Pikachu', graph_num: int = 1):
    plt.figure(graph_num, figsize=(10, 6))
    plt.xlabel(f"Effectiveness of Attack", fontweight='bold')
    plt.ylabel(f"Type Match-Up", fontweight='bold')
    matchup, colors = make_type_dicts()
    inital = name_or_type
    if name_or_type not in list(df.index):
        name_or_type = name_or_type.split()
        if len(name_or_type) == 2:
            primary_type = name_or_type[0].capitalize()
            secondary_type = name_or_type[1].capitalize()
        else:
            primary_type = name_or_type[0].capitalize()
            secondary_type = "".capitalize()
    else:
        primary_type = df.loc[name_or_type, 'primary_type'].capitalize()
        secondary_type = df.loc[name_or_type, 'secondary_type'].capitalize()

    if secondary_type == "":
        type_matchup = matchup[primary_type]
    else:
        type_matchup = [x*y for x, y in zip(matchup[primary_type], matchup[secondary_type])]

    type_labels = ['Normal', 'Fighting', 'Flying', 'Poison', 'Ground', 'Rock', 'Bug', 'Ghost', 'Steel', 'Fire', 'Water', 'Grass', 'Electric', 'Psychic', 'Ice', 'Dragon', 'Dark', 'Fairy']
    bar_colours = [colors[i] for i in type_labels]
    locs, y_labels = plt.yticks()
    plt.yticks(ticks=np.arange(0, len(type_labels)), labels=type_labels)
    plt.ylabel("Incoming Attack Type", fontweight='bold')
    if secondary_type == "":
        plt.title(f"Based on {inital}", fontsize=17.5, y=1)
        plt.suptitle(f"Damage Received by {primary_type} Types", fontsize=25, fontweight='bold', y=0.99)
    else:
        plt.title(f"Based on {inital}", fontsize=17.5, y=1)
        plt.suptitle(f"Damage Received by {primary_type}/{secondary_type} Types", fontsize=25, fontweight='bold', y=0.99)
    plt.barh(type_labels, type_matchup, color=bar_colours)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        pokemon = pd.read_csv(sys.argv[1], encoding='utf-16')
        weakness_check = None
    else:
        pokemon = pd.read_csv(sys.argv[1], encoding='utf-16')
        weakness_check = sys.argv[2]


    pokemon = pokemon_cleaning(pokemon.set_index("english_name"))
    # Analysis
    print("ANALYSIS")
    pokemon_metrics = metrics(pokemon, ["total", 'hp', "attack", "defense", 'sp_attack', 'sp_defense', 'speed'])
    # 5 different plots
    plot_count(pokemon, "gen", "is_legendary", incl_legendary=True, legendary_only=True, title="Number of Legendaries Per Generation", graph_num=1)
    plot_type_comparison(pokemon, "primary_type", ["attack", "defense", 'sp_attack', 'sp_defense'], incl_legendary=True, legendary_only=False, title="Average Stat Values of Primary Type", graph_num=2)
    pc_box_plot(pokemon, "gen", "total", incl_legendary=False, legendary_only=False, title="Stat Distribution for Non-Legendaries", graph_num=3)
    pc_box_plot(pokemon, "gen", "total", incl_legendary=True, legendary_only=False, title="Stat Distribution for All Pokemon", graph_num=4)
    mega(pokemon, "gen", incl_legendary=True, legendary_only=False, title="Number of Gimmick Pokemon Per Generation", graph_num=5)
    if weakness_check is None:
        type_chart(pokemon, pokemon.sample().index.to_list()[0], graph_num=6)
    else:
        type_chart(pokemon, weakness_check, graph_num=6)
    # display
    plt.show()
