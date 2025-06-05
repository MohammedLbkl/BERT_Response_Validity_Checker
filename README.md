# Évaluation de la Pertinence d'une Réponse à une Question

## Objectif du projet

Ce projet vise à déterminer automatiquement si une réponse donnée est pertinente par rapport à une question posée.

## Approche initiale : Mots-clés avec spaCy

Dans un premier temps, j'ai utilisé **[spaCy](https://spacy.io/)**, une bibliothèque de traitement du langage naturel, pour extraire les **mots-clés** d’une question et d’une réponse.  
L’idée était de comparer ces mots-clés : si ceux de la réponse recouvrent ceux de la question, la réponse est considérée comme pertinente.

### Limite
Cette méthode s’est révélée peu fiable : il suffisait que la réponse contienne les bons mots-clés, même hors contexte, pour qu’elle soit jugée correcte.

## Approche avancée : Similarité sémantique

Pour aller plus loin, j’ai exploré la **similarité sémantique** entre la question et la réponse :

- **spaCy** propose des vecteurs statiques (Word2Vec, GloVe), mais ils ne tiennent pas compte du contexte.
- J’ai ensuite utilisé **BERT** pour obtenir des **vecteurs contextuels**, bien plus adaptés à cette tâche.

### Problèmes rencontrés
- Les scores de similarité sont souvent peu fiables pour :
  - des **réponses courtes** 
  - des **réponses binaires**
- Le contexte est parfois mal capturé, surtout en l'absence d'informations explicites.

## Piste d'amélioration

Deux solutions principales peuvent être envisagées :

1. **Combinaison des approches**  
   - Fusionner l'analyse de similarité (via BERT) avec la recherche de mots-clés pour une meilleure robustesse.

2. **Classification supervisée**  
   - Entraîner un modèle à **classer les réponses** comme pertinentes ou non.  
   - Cela nécessite un **jeu de données étiqueté**.

3. **Utilisation d'un LLM (modèle de langage)**  
   - Poser directement la question au modèle :  
     *"Cette réponse répond-elle bien à la question ?"*
   - Permet de tirer parti de la compréhension contextuelle des grands modèles comme **ChatGPT**.


