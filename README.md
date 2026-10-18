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
uv sync
uv run python main.py
```

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