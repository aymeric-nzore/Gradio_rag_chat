---
title: RAG Chat
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# RAG Chat

Application Gradio de questions-réponses sur des fichiers PDF, avec Mistral pour le
modèle et les embeddings. Neon PostgreSQL avec l'extension pgvector stocke
l'historique et les vecteurs, et Mistral fournit le modèle ainsi que les embeddings.

### Lancer avec Docker Compose

```bash
docker compose up --build
```

Copiez `.env.example` vers `.env` et renseignez les variables Mistral et Neon.
Le Compose utilise Neon comme base externe et conserve seulement les fichiers
PDF dans le volume local `rag_data`.

### Lancer localement

```bash
uv pip install --system -r docker-requirements.txt
python main.py
```

Les dépendances du frontend Vercel et celles de l'application Gradio sont
séparées afin d'éviter d'embarquer Torch, Whisper et Gradio dans la fonction
serverless.

## Déployer sur Hugging Face Spaces avec Neon

1. Créez un projet Neon et activez l'extension `vector` dans la base :

   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

2. Créez un Docker Space et poussez les fichiers du projet, notamment
   `Dockerfile`, `main.py`, `pyproject.toml` et `README.md`.
3. Dans les réglages du Space, ajoutez ces variables secrètes :

   ```text
   MISTRAL_API_KEY=...
   POSTGRES_URL=postgresql://...neon.tech/...?...sslmode=require
   PGVECTOR_URL=postgresql+psycopg://...neon.tech/...?...sslmode=require
   ```

   `POSTGRES_URL` et `PGVECTOR_URL` peuvent utiliser les chaînes de connexion
   Neon. La seconde doit utiliser le schéma `postgresql+psycopg`.
4. Le Space écoute sur le port `7860`, déjà déclaré dans le Dockerfile et les
   métadonnées de ce README.

Les PDF uploadés sont écrits dans `/data/uploads`. Le disque d'un Space pouvant
être éphémère, utilisez un stockage persistant Hugging Face si ces fichiers
doivent survivre à un redémarrage. L'historique et les embeddings restent dans
Neon.

Les modèles par défaut sont `mistral-small-latest` et `mistral-embed`. Ils
peuvent être remplacés par `MISTRAL_MODEL` et `MISTRAL_EMBEDDING_MODEL`.

## Déployer sur Vercel

Vercel n'exécute pas un serveur Gradio permanent. Le projet fournit donc une
variante Vercel avec une API FastAPI stateless et une interface web statique :

- `api/index.py` expose l'API RAG ;
- `public/index.html` fournit l'interface utilisateur ;
- `vercel.json` route `/api/*` vers FastAPI ;
- `requirements.txt` contient uniquement les dépendances de la fonction Vercel.

1. Importez ce dépôt GitHub dans Vercel.
2. Dans les variables d'environnement du projet, ajoutez
   `MISTRAL_API_KEY`, `POSTGRES_URL` et `PGVECTOR_URL`.
3. Déployez le projet.
4. Ouvrez l'URL Vercel et importez un PDF.

Les fichiers PDF sont traités dans la fonction puis supprimés du disque
temporaire. Les messages et les embeddings sont persistés dans Neon. Cette
variante ne dépend pas de Gradio, d'Ollama ou d'un volume local.

Vercel impose des limites de durée et de taille aux fonctions. Pour de gros
PDF ou des traitements très longs, une plateforme persistante reste préférable.