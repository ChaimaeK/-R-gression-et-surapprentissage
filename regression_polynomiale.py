import numpy as np
import matplotlib.pyplot as plt

def polyreg(X, Y, degre):
	""" Fonction d'apprentissage sur la base (X,Y)
	w = polyreg(X, Y, degre)
	w 	: paramètres du modèle polynomial f 
	degre : degre du polynome f
	X 	= [x1; x2; x3; ... ; xm]
	Y 	= [y1; y2; y3; ... ; ym]
	"""
	w=0

	phi = np.ones((len(X),degre+1))
	for i in range(1,degre+1):
		phi[:,i] = phi[:,i-1]*X
	
	w = np.linalg.solve(phi.T@phi,phi.T@Y)

	return w

def polypred(X, w):
	"""
	Fonction de prédiction Y = polypred(X, w)
	
	w 	: paramètres du modèle polynomial f issus de polyreg
	X 	: vecteur de taille N contenant les N valeurs de x où évaluer le polynome f(x)
	Y 	: valeurs de polynome f(X) 
	"""
	
	phi = np.ones((len(X),len(w)))
	for i in range(1,len(w)):
		phi[:,i] = phi[:,i-1]*X
	

	Y = phi@w	# à modifier
	
	return Y


########### programme principal ########


## 1) générer une base de données de 1000 points X,Y

m = 1000
X = 6 * np.random.rand(m) - 3
Y = np.sinc(X) + 0.2 * np.random.randn(m)


# 2) Créer un base d'apprentissage (Xapp, Yapp) de 30 points parmi ceux de (X,Y) et une base de test(Xtest,Ytest) avec le reste des données

indexes = np.random.permutation(m)  # permutation aléatoire des 1000 indices entre 0 et 1000 
indexes_app = indexes[:30]  # 30 premiers indices
indexes_test = indexes[30:] # le reste

Xapp = X[indexes_app]
Yapp = Y[indexes_app]

Xtest = X[indexes_test]
Ytest = Y[indexes_test]

# ordronner les Xtest pour faciliter le tracé des courbes
idx = np.argsort(Xtest)
Xtest = Xtest[idx]
Ytest = Ytest[idx]

# tracer la figure

plt.plot(Xapp,Yapp,'*b')
plt.plot(Xtest,np.sinc(Xtest) , 'g')

legend = [ 'base app', 'f_reg(x)']

## 3) implémenter les fonctions polyreg et polypred pour la régression polynomiale unidimensionelle
for i in range(1,15):
	w = polyreg(Xapp,Yapp,i)
	y = polypred(Xtest,w)
	plt.plot(Xtest,y)
	leg = "degre_",i
	legend.append(leg)

# 4) Réaliser l'apprentissage d'un modèle de degré 5 et calculer les erreurs d'apprentissage et de test


plt.axis([-3,3,-1,2])
plt.legend(legend )

	
# Affichage des graphiques : 
# (à ne faire qu'en fin de programme)
plt.show() # affiche les plots et attend la fermeture de la fenêtre
