Le programme se lance avec "python3 hough.py "lien image""
avant le lancement programme faire poetry shell
Normalement tout marche mais sinon faire une fois ces commandes:
poetry install
poetry add sys
poetry add numpy
poetry add matplotlib
poetry add math
poetry add PIL
poetry add typing

Le seuil sobel est de 150 et le seuil hough de 200.
C'est pas optimal mais ca marche a peu pres.