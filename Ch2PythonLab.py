#%%

import numpy as np

# Arrays erstellen
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print("[1, 2, 3] + [4, 5, 6] ergibt:", a+b)

# Informationen über ein Array abrufen und ein Array verändern
c = np.array([[1, 2], [3, 4]])
print("Array:\n", c,
      "\nDimensionen:", c.ndim,
      "\nDatentyp:", c.dtype,
      "\nShape:", c.shape,
      "\nSumme:", c.sum(),
      "\nAndere Möglichkeit einer Summe:", np.sum(c),
      "\nReshaped:", c.reshape(1, 4),
      "\nTransposed:\n", c.T,
      "\nQuadratwurzel:\n", np.sqrt(c),
      "\nQuadrate:\n", c**2)

# Stastische Informationen über die Zahlen in einem Array abrufen
rng = np.random.default_rng(3)
print("Random seed initialisiert") # Statt np.random.normal() nutzen wir rng.normal(), um einen festen Seed zu haben
d = rng.normal(size=50) # Normalverteilungskurve!
print("Array mit 50 Items, Durchschnitt 0 und Standardabweichung von 1:\n", d,
      "\nMean:", np.mean(d),                # 50: 0.0248,   100: -0.0617
      "\nStandard Deviation:", np.std(d),   # 50: 1.0920,   100: 1.0663
      "\nVariance:", np.var(d))             # 50: 1.1925,   100: 1.1370
e = d + rng.normal(loc=50, scale=1, size=50)
print("Zweites Array: Durchschnitt 50 und Standardabweichung 1 dazuaddiert:", e,
      "\nMean:", e.mean(),                  # 50: 49.8764,  100: 50.0966
      "\nStandard Deviation:", e.std(),     # 50: 1.5866,   100: 1.4832
      "\nVariance:", e.var())               # 50: 2.5174,   100: 2.1999
print("Korrelations-Koeffizient:\n", np.corrcoef(d, e)) # Gibt an, ob die beiden Matrizen (nur!) linear voneinander abhängig sind (1: vollständig positiv linear, -1: vollständig negativ linear)
f = rng.standard_normal((10, 3))
print("Durchschnitt der drei Spalten:", f.mean(0),
      "\nDurchschnitt der 10 Reihen:", f.mean(1))

# %%
# Visualizations
from matplotlib.pyplot import subplots
g = rng.standard_normal(100)
h = rng.standard_normal(100)
fig, ax = subplots(figsize=(8, 8))
ax.plot(g, h); # Line Plot, the default
# %%
fig, ax = subplots(figsize=(8, 8))
ax.plot(g, h, 'o') # Scatter Plot
fig.savefig("scatterplot.jpg", dpi=100)
# %%
fig, ax = subplots(figsize=(8, 8))
ax.scatter(g, h, marker='o') # also a Scatter Plot

# %%
# Change visualization
fig, ax = subplots(figsize=(8, 8))
ax.scatter(g, h, marker='o')
ax.set_xlabel("this is the x-axis")
ax.set_ylabel("this is the y-axis")
ax.set_title("Plot of X vs Y")
fig.set_size_inches(12,3)

# %%
# Create several plots and change them
fig, axes = subplots(nrows=2, ncols=3, figsize=(15, 5))
axes[0,1].plot(g, h, 'o')
axes[1,2].scatter(g, h, marker='+')
axes[0,1].set_xlim([-1,1])

# %%
# Creating a contour plot
fig, ax = subplots(figsize=(8, 8))
i = np.linspace(-np.pi, np.pi, 50)
j = i
k = np.multiply.outer(np.cos(j), 1 / (1 + i**2))
print(k)
ax.contour(i, j, k, levels = 45)

# %%
# Creating a heatmap
fig, ax = subplots(figsize=(8, 8))
ax.imshow(k)

# %%
# Sequences and slicing
l = np.linspace(start=0, stop=10, num=11) # useful for when you know the number of samples
print("Sequence of numbers, start is 0, stop is 10 (included), with 11 numbers generated between them (default of numbers is 50):\n", l)
m = np.arange(0, 10) # useful for when you know the step size
print("Sequence of numbers between 0 and 10 (excluded) (default step is 1)\n", m)
m = m.reshape(2, 5)
print("Reshaped array:\n", m,
      "\nSelecting first row and second column:\n", m[0,1],
      "\nSelecting 1st and 2nd row:\n", m[[0,1]],
      "\nSelecting second and fourth column:\n", m[:,[1,3]],
      "\nSelecting a submatrix with specific values:\n", m[[0,1]][:,[2,3,4]],
      "\nAlternative way of creating submatrices using intermediate meshes:\n", m[np.ix_([0,1], [2,3,4])],
      "\nAlternative way of creating submatrices using slices:\n", m[0:2:1,2:5:1])
# Slices work but lists don't, since numpy uses lists and slices differently

# %%
# Booleans and Arrays
n = np.zeros(m.shape[0], bool)
n[[0,1]] = True
print("Array n with as many elements as the row number from m:", n,
      "\nCheck if each element of n is the same as [1,1]:", np.all(n == np.array([1,1])),
      "\nCheck if any element of n is the same as [0,1]:", np.any(n == np.array([0,1])),
      "\nRetrieve all rows according to n:\n", m[n],
      "\nRetrieve all columns according to [0,1,0,1,0]:\n", m[np.array([0,1,0,1,0])])
o = np.zeros(m.shape[1], bool)
o[[2,3]] = True
idx_bool = np.ix_(n, o)
print("Selecting submatrices again using meshes:\n", m[idx_bool])
idx_mixed = np.ix_([0,1],o)
print("The same, but with a different way of creating the mesh:\n", m[idx_bool])

# %%
# Loading data