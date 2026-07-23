# Dictionnaire des features 

Nous créons des variables importantes pour notre modèles de machile Learning

#### FLAG_PRIMO_DEMANDEUR
Cette variable vas nous renseigner si le demandeur à deja eu un historique de crédit ou pas. Ce qui est assez informatif en soi. Cette variable permet au modèle d'isoler ces cas pour ne pas les pénaliser injustement par rapport à ceux qui ont un long historique.

#### SOLDE FLUX ET RATIO
Nous fesont ici une variable pour savoir ce qui reste a vivre pour le demandeur et savoir si il gagne plus qu'il ne depense. Le ratio nous permets de normaliser les comparaisons, par exemple un demandeur qui gagne 100.000/mois et depense 90.000/mois, n'est pas la meme chose que celui qui gagne 500.000 par mois et depense 400.000 .

#### IDENTITË (volume_mm et ratio )
Ici on vas mésuurer le volume total d'argent qui transite sur le compte Mobile Money du client. Cette variable bien evidement nous sert à detecter les fraudes pour les clients tout en mesurant les entrer et sortie de celui ci

#### REGULARITË
Ici nous calculons les richesses accumulés des clients, grâce a leur score Mobile Money, ceci veut dire qu'un bon solde sur 24 mois est une variable importante que un bon solde sur 1 ou de 2, ceci peut etre due a un coup de chance

#### RETARD CREDIT 
Le taux de retard permet de savoir son niveau chronique de retard, et le retard max par crédit permet de penaliser les clients qui ont un gros retard par rapport a d'autre. Un retard de 2 jours sur un ancien crédit, ne seras pas la meme chose qu'un retard de 60 jours sur un ancien crédit

#### POSSIBILITË DE REMBOURSEMENT 
Le montant_credit_moyen et montant_total_credit permet de savoir à quel dose le demandeur s'endette, par rapport a son gain. Un demandeur qui gagne 1millions sur l'année et demande un crédit de 800 milles sur la meme année est aussi un risque à prendre (De ne pas etre payé a temps).

#### ANCIEN CRÉDIT
Un demandeur qui a un crédit remboursé en 3 mois et dans les délais est rassurant, par rapport a celui qui a remboursé un crédit il 5 ans (on pas plus d'information sur lui)

#### SURENDEMENT
Un demandeur qui a un ration endement > 1. est une variable importante et eliminatoire pour le client.


