# poke-project
Dataset taken from kaggle (https://www.kaggle.com/datasets/cristobalmitchell/pokedex?resource=download)

### Visualization script
When executing the ``visualization.py`` script from the command line, the option to present two system arguments is available. After the 
system argument representing the dataset, you have the option of choosing a specific Pokémon to use in the generation 
of the weakness chart, or if a Pokémon name is not input, a random one will be selected.

### Unsupervised learning script
When calling ``unsupervised_learning.py`` there is one required argument and one optional. Using the command ``python unsupervised_learning.py [required: path of the file] [optional: custom DBSCAN]`` the python script will be 
executed. The path of the file is a required argument, and the second argument is an optional boolean argument. The purpose of the second argument is to employ a custom DBSCAN implementation (using
``epsilon=19.175`` and ``minimum points=5``), and leaving it blank will default the parameter to ``False``. A ``False`` value causes the implementation of DBSCAN with default parameters, and 
passing a ``True`` value as the second parameter implements DBSCAN using the custom parameters for outlined previously.

When the script is executed, both K-Means and DBSCAN clustering will be performed on the passed data file, with the plots being saved to the same folder and also displayed using matplotlib's ``show()``
function.
