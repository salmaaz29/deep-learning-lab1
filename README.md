# Lab 1 – Deep Learning : régression et classification avec PyTorch

## Objectif

Me familiariser avec PyTorch en construisant des MLP pour deux tâches :

- **Partie 1, régression :** prédire le prix de clôture du lendemain d'actions (dataset NYSE).
- **Partie 2, classification multi-classes :** prédire le type de panne d'une machine (maintenance prédictive).

Pour chaque partie : EDA, prétraitement, MLP, GridSearch, courbes Loss/Epochs (et Accuracy en classification), métriques, régularisation et comparaison avec le premier modèle.

## Contenu du dépôt

- `Atelier_1_partie_1.ipynb` : Partie 1, régression NYSE (code, sorties, commentaires, synthèse).
- `Atelier_1_partie_2.ipynb` : Partie 2, classification Machine Predictive Maintenance (code, sorties, commentaires, synthèse).

## Données

- NYSE (`prices-split-adjusted.csv`) : Kaggle, dataset *New York Stock Exchange*.
- Machine Predictive Maintenance Classification : Kaggle.

Les datasets ne sont pas dans le dépôt.

## Exécution

Les notebooks ont été exécutés sur Kaggle : ajouter les datasets via *Add Input*, puis *Run All*. Les chemins commencent par `/kaggle/input/...` et sont à adapter hors Kaggle.
Bibliothèques : pandas, numpy, matplotlib, seaborn, scikit-learn, PyTorch, skorch, imbalanced-learn (`pip install -q skorch imbalanced-learn`).

## Partie 1 : régression (NYSE)

**EDA.** Étude de l'évolution des prix, des corrélations entre open/high/low/close et des erreurs par action. Le MAPE sur le train varie beaucoup selon l'action : AAPL 4,44 %, AMZN 1,45 %, GOOG 1,09 %, IBM 0,96 %. Pour AAPL, il passe de 10,1 % en 2010 à 1,5 % en 2015 (le prix change beaucoup d'échelle sur la période).

**Méthode.**
- Cible : close(t+1) (`shift(-1)`), découpage chronologique train / test pour éviter la fuite de données, standardisation apprise sur le train uniquement.
- MLP en PyTorch, MSELoss. Le premier modèle utilise SGD, 64 neurones, 80 epochs.
- GridSearch (skorch + sklearn) avec validation croisée temporelle (validation sur 2013, 2014, 2015). Meilleur réglage : Adam, lr = 0,001, 20 epochs, 32 neurones, activation ReLU (MSE ≈ −0,0021).
- En régression, la courbe « Accuracy » n'a pas de sens : on suit la loss et l'erreur en dollars.

**Résultats (test).**

| Modèle | RMSE test ($) | MAPE test (%) |
|---|---|---|
| Baseline « demain = aujourd'hui » | 7,60 | 1,01 |
| Modèle 1 (SGD, 64 neurones, 80 epochs) | 7,80 | 1,09 |
| Modèle GridSearch (Adam, lr 0,001, 20 epochs, 32 neurones) | 7,96 | 1,14 |
| Adam, lr 0,001, 50 epochs | 7,54 | 1,12 |

**Régularisation.**

| Modèle | Loss test | RMSE test ($) | MAPE test (%) |
|---|---|---|---|
| Base | 0,00300 | 8,01 | 1,27 |
| L2 (1e-4) | 0,00307 | 7,94 | 1,18 |
| L2 (1e-3) | 0,00318 | 8,42 | 1,17 |
| Dropout 0,1 | 0,03205 | 34,15 | 4,42 |
| Dropout 0,2 | 0,08834 | 67,95 | 7,81 |
| BatchNorm | 0,03172 | 60,07 | 6,60 |
| Early stopping | 0,00439 | 10,88 | 1,47 |
| L2 + Dropout | 0,03021 | 36,53 | 4,75 |

**Conclusion.** Le modèle de base ne sur-apprenait pas : sa loss de test ne remontait pas. L'écart train/test vient surtout du décalage de distribution des prix en 2016, que la régularisation ne corrige pas. Aucun modèle ne bat vraiment la baseline naïve : la série est dominée par le fait que close(t) est très proche de close(t+1).

## Partie 2 : classification multi-classes

**Données et nettoyage.** Suppression des identifiants inutiles (UDI, Product ID), de 9 lignes incohérentes (panne annoncée mais « No Failure ») et de la colonne `Target`, qui contient déjà la réponse (fuite d'information). Six classes dont 96,5 % de « No Failure ».

**EDA.** Les boxplots montrent que chaque panne a une signature dans les capteurs : machine lente et faible écart de température pour Heat Dissipation, couple fort et outil usé pour Overstrain, outil très usé pour Tool Wear, valeurs extrêmes pour Power. Random Failures n'a aucune signature, ce qui est logique puisqu'elle arrive au hasard.

**Méthode.**
- Découpage stratifié (test 20 %, soit 1 999 lignes), puis standardisation apprise sur le train.
- SMOTE (k_neighbors = 3) sur le train uniquement, après le découpage : 7 714 exemples par classe. Random Failures passe de 14 exemples réels à 7 714, presque tous synthétiques.
- MLP en PyTorch avec `Softmax` en sortie et `NLLLoss` sur `log(probs)`. Utiliser `Softmax` avec `CrossEntropyLoss` applique le softmax deux fois (loss bloquée vers 1,1).
- GridSearch (skorch + `GridSearchCV`, validation croisée stratifiée à 3 folds, scoring F1 macro) sur learning rate, optimiseur, epochs et tailles des couches. Meilleur réglage : Adam, lr = 0,001, 40 epochs, réseau 128-64 (F1 macro CV : 0,944). Ce score est optimiste, car SMOTE est appliqué avant la validation croisée.
- Modèle régularisé : BatchNorm, Dropout (0,3) et L2 (1e-4).

**Résultats (test).**

| Modèle | Accuracy | Précision (macro) | Sensibilité (macro) | F1 (macro) |
|---|---|---|---|---|
| Base | 0,960 | 0,487 | 0,592 | 0,530 |
| Régularisé | 0,890 | 0,395 | 0,660 | 0,466 |

Rapport par classe (modèle de base) :

| Classe | Précision | Sensibilité | F1 | Exemples |
|---|---|---|---|---|
| No Failure | 0,99 | 0,97 | 0,98 | 1 929 |
| Heat Dissipation | 0,53 | 0,86 | 0,66 | 22 |
| Overstrain | 0,74 | 0,88 | 0,80 | 16 |
| Power | 0,67 | 0,84 | 0,74 | 19 |
| Random Failures | 0,00 | 0,00 | 0,00 | 4 |
| Tool Wear | 0,00 | 0,00 | 0,00 | 9 |

**Conclusions.**
- Sur le train équilibré par SMOTE, le modèle atteint 99,7 %, contre un F1 macro de 0,53 sur le test : la comparaison n'est pas équitable, car le test n'est pas équilibré.
- Le modèle régularisé détecte plus de vraies pannes (sensibilité 0,59 → 0,66) mais génère plus de fausses alertes sur les machines saines (59 → 205), et son F1 macro baisse (0,53 → 0,47). Le meilleur modèle dépend du coût d'une panne manquée par rapport à celui d'une fausse alerte.
- Tool Wear et Random Failures ne sont pas détectées : seulement 36 et 14 exemples réels au train, 9 et 4 au test, donc des scores très instables.

## Ce que j'ai appris

- **PyTorch :** écrire un MLP (`nn.Module`), la boucle d'entraînement (zero_grad → forward → loss → backward → step), les modes train/eval, l'importance de bien associer sortie du modèle et fonction de perte.
- **Éviter la fuite de données :** retirer `Target`, découpage chronologique en séries temporelles, standardisation et SMOTE après le découpage.
- **Classes déséquilibrées :** l'accuracy est trompeuse, il faut regarder F1 macro, sensibilité et matrice de confusion.
- **Hyper-paramètres :** GridSearch avec skorch, et méfiance envers un score de validation croisée optimiste.
- **Régularisation :** Dropout, BatchNorm, L2 et early stopping ne sont pas automatiquement utiles, notamment quand le problème vient d'un décalage de distribution ou d'un manque de données.

## Limites et pistes

Un seul entraînement par configuration et très peu d'exemples pour certaines pannes : les petits écarts entre modèles sont du bruit. Pistes non réalisées : ajouter des variables physiques (écart de température, puissance, usure × couple), utiliser un SMOTE moins agressif, appliquer SMOTE à l'intérieur de chaque fold de la validation croisée, et pondérer les classes.
