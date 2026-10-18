from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
import gradio as gr
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_core.stores import InMemoryStore
from langchain_classic.retrievers import ParentDocumentRetriever
import psycopg
from datetime import datetime
import warnings
import os
import tempfile
import shutil
import uuid
import numpy as np
from psycopg.types.json import Json

warnings.filterwarnings('ignore')

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    from gtts import gTTS
except ImportError:
    gTTS = None

try:
    from transformers import pipeline
    AUDIO_AVAILABLE = sf is not None
    whisper_asr = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base",
        device="cpu",
        chunk_length_s=30
    )
except (ImportError, OSError) as e:
    AUDIO_AVAILABLE = False
    whisper_asr = None
    print(f"Audio non disponible: {e}")

audio_temp_dir = Path(tempfile.gettempdir()) / "rag_audio"
audio_temp_dir.mkdir(exist_ok=True)

def cleanup_audio_files():
    """Remove temporary audio files older than one hour."""
    import time
    for f in audio_temp_dir.glob("*.wav"):
        if time.time() - f.stat().st_mtime > 3600:
            try:
                f.unlink()
            except OSError:
                continue

def save_audio_from_gradio(audio_data, sample_rate=16000):
    """Save Gradio audio as a mono 16 kHz WAV file."""
    if audio_data is None or sf is None:
        return None

    if isinstance(audio_data, tuple):
        sr, audio_np = audio_data
    else:
        sr = sample_rate
        audio_np = audio_data

    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    if len(audio_np) == 0:
        return None

    if sr != 16000:
        audio_np = np.interp(
            np.linspace(0, len(audio_np), int(len(audio_np) * 16000 / sr)),
            np.arange(len(audio_np)),
            audio_np.astype(float)
        ).astype(np.float32)

    peak = np.max(np.abs(audio_np))
    if peak > 0:
        audio_np = audio_np / peak

    audio_path = audio_temp_dir / f"input_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.wav"
    sf.write(str(audio_path), audio_np, 16000, format='WAV')
    return str(audio_path)

def speech_to_text(audio_data):
    """Convert speech to text using Whisper"""
    if not AUDIO_AVAILABLE or whisper_asr is None:
        return "❌ Whisper non disponible. Installez: pip install transformers soundfile torch"

    try:
        audio_path = save_audio_from_gradio(audio_data)
        if audio_path is None:
            return "⚠️  Aucun audio capturé"

        result = whisper_asr(audio_path, language="french")
        text = result["text"]

        try:
            os.remove(audio_path)
        except OSError:
            pass

        return text if text else "⚠️  Aucune parole détectée"

    except Exception as e:
        return f"❌ Erreur Whisper: {str(e)}"

def text_to_speech(text, lang="fr"):
    """Convert text to speech using gTTS"""
    if gTTS is None:
        return None

    try:
        if not text or text.strip() == "":
            return None

        tts = gTTS(text=text, lang=lang, slow=False)
        audio_path = audio_temp_dir / f"output_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.mp3"
        tts.save(str(audio_path))
        return str(audio_path)
    except Exception as e:
        print(f"❌ Erreur TTS: {str(e)}")
        return None

base_dir = Path(os.getenv("RAG_DATA_DIR", Path(__file__).resolve().parent))
uploads_dir = base_dir / "uploads"
uploads_dir.mkdir(exist_ok=True)

def get_splitters():
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=20, separators=['\n\n', '\n', ' ', '']
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=20, separators=['\n', '. ', ' ', '']
    )
    return parent_splitter, child_splitter

parent_splitter, child_splitter = get_splitters()

POSTGRES_URL = os.getenv(
    "POSTGRES_URL",
    "postgresql://rag_user:rag_password@localhost:5432/rag_chat",
)
PGVECTOR_URL = os.getenv(
    "PGVECTOR_URL",
    POSTGRES_URL.replace("postgresql://", "postgresql+psycopg://", 1),
)
VECTOR_COLLECTION = "chat_docs"

def init_db():
    """Initialize PostgreSQL tables for conversation history."""
    with psycopg.connect(POSTGRES_URL) as conn:
        cursor = conn.cursor()
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id BIGSERIAL PRIMARY KEY,
            title TEXT DEFAULT 'Nouvelle conversation',
            pdf_files JSONB,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id BIGSERIAL PRIMARY KEY,
            conversation_id BIGINT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            temperature DOUBLE PRECISION DEFAULT 0.5,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
        """)

init_db()

def get_db_connection():
    return psycopg.connect(POSTGRES_URL)

def create_conversation(pdf_files=None):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO conversations (title, pdf_files) VALUES (%s, %s) RETURNING id",
            (
                f"Conversation - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                Json(pdf_files) if pdf_files else None,
            ),
        )
        conversation_id = cursor.fetchone()[0]
    return conversation_id

def save_message(conversation_id, role, content, temperature=0.5):
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, temperature) VALUES (%s, %s, %s, %s)",
            (conversation_id, role, content, temperature),
        )

def load_conversation_messages(conversation_id):
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE conversation_id = %s ORDER BY created_at, id",
            (conversation_id,),
        ).fetchall()

    history = []
    for row in rows:
        role, content = row
        if role == "user":
            history.append([content, None])
        else:
            if history:
                history[-1][1] = content
    return history

def get_all_conversations():
    with get_db_connection() as conn:
        conversations = conn.execute(
            "SELECT id, title, created_at FROM conversations ORDER BY created_at DESC"
        ).fetchall()
    return [(c[0], f"{c[1]} - {c[2]}") for c in conversations]

def delete_conversation(conversation_id):
    with get_db_connection() as conn:
        conn.execute("DELETE FROM conversations WHERE id = %s", (conversation_id,))

store = InMemoryStore()
current_retriever = None
current_pdf_files = None
current_conversation_id = None

system_prompt = """
Tu es un assistant utile et chaleureux.

Réponds en français, de manière claire, concise et naturelle.
Réponds uniquement à partir du contexte fourni. Si le contexte ne permet pas de répondre,
dis clairement que tu ne disposes pas de cette information.

N'utilise jamais de Markdown.
Utilise des phrases et des paragraphes simples.
Tu peux utiliser des emojis avec modération.
"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Contexte :\n{context}\n\nQuestion :\n{question}"),
])

def get_llm(temperature=0.5):
    return ChatMistralAI(
        model=os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
        temperature=temperature,
        max_retries=2,
    )

def get_embeddings():
    return MistralAIEmbeddings(
        model=os.getenv("MISTRAL_EMBEDDING_MODEL", "mistral-embed"),
    )

def get_vector_db():
    return PGVector(
        collection_name=VECTOR_COLLECTION,
        embeddings=get_embeddings(),
        connection=PGVECTOR_URL,
        use_jsonb=True,
    )

def create_retriever(docs):
    global current_retriever, store
    vectorstore = get_vector_db()
    vectorstore.delete_collection()
    store = InMemoryStore()
    current_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    current_retriever.add_documents(docs)

def upload_pdf(files, conversation_dropdown):
    global current_retriever, store, current_pdf_files, current_conversation_id

    try:
        if not files:
            return "⚠️ Veuillez sélectionner au moins un PDF.", conversation_dropdown, [], None

        source_paths = files if isinstance(files, list) else [files]
        file_paths = []
        for source_path in source_paths:
            source = Path(source_path)
            destination = uploads_dir / f"{uuid.uuid4().hex}_{source.name}"
            shutil.copy2(source, destination)
            file_paths.append(str(destination))
        current_pdf_files = file_paths

        docs = []
        for file_path in file_paths:
            docs.extend(PyPDFLoader(file_path).load())

        create_retriever(docs)

        conv_id = create_conversation(file_paths)
        current_conversation_id = conv_id

        file_count = len(file_paths)
        file_word = "fichier" if file_count == 1 else "fichiers"
        status_msg = f"✅ {file_count} {file_word} chargé(s) !"

        conversations = get_all_conversations()
        new_dropdown = gr.Dropdown(
            choices=conversations,
            value=conv_id,
            label="Historique des conversations"
        ) if conversations else conversation_dropdown

        return status_msg, new_dropdown, [], None

    except Exception as e:
        return f"❌ Erreur lors du chargement : {str(e)}", conversation_dropdown, [], None

def load_selected_conversation(conversation_id):
    global current_pdf_files, current_conversation_id, current_retriever, store

    if conversation_id is None:
        return []

    current_conversation_id = conversation_id

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT pdf_files FROM conversations WHERE id = %s", (conversation_id,))
    result = cursor.fetchone()
    conn.close()

    if result and result[0]:
        current_pdf_files = result[0]

        if current_pdf_files:
            docs = []
            for file_path in current_pdf_files:
                try:
                    docs.extend(PyPDFLoader(file_path).load())
                except (OSError, ValueError):
                    continue

            create_retriever(docs)

    history = load_conversation_messages(conversation_id)
    return history

def process_user_input(text_input, audio_input, history, conversation_id):
    """Process user input from text or audio"""
    global current_conversation_id

    current_conversation_id = conversation_id

    if conversation_id is None:
        return "⚠️ Veuillez d'abord charger un PDF ou sélectionner une conversation.", history, None, None

    user_message = text_input if text_input else ""

    if audio_input is not None and (isinstance(audio_input, tuple) or isinstance(audio_input, np.ndarray)):
        transcribed_text = speech_to_text(audio_input)
        if not transcribed_text.startswith("❌") and not transcribed_text.startswith("⚠️"):
            user_message = transcribed_text
        else:
            return transcribed_text, history, None, None

    if not user_message:
        return "⚠️  Aucun texte ou audio valide.", history, None, None

    save_message(conversation_id, "user", user_message)
    new_history = history + [[user_message, None]]

    return "", new_history, None, None

def generate_and_speak_response(history, temperature, conversation_id, auto_speak):
    """Generate response and optionally speak it"""
    global current_conversation_id

    current_conversation_id = conversation_id

    if current_retriever is None or conversation_id is None:
        if history:
            history[-1][1] = "⚠️ Veuillez d'abord charger un PDF."
        return history, None

    try:
        user_message = history[-1][0]
        llm = get_llm(temperature=temperature)
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=current_retriever,
            chain_type_kwargs={"prompt": qa_prompt},
            return_source_documents=False
        )
        response = qa.invoke({"query": user_message})
        response_text = response["result"]

        save_message(conversation_id, "assistant", response_text, temperature)
        history[-1][1] = response_text

        audio_path = None
        if auto_speak:
            audio_path = text_to_speech(response_text)

        return history, audio_path

    except Exception as e:
        if history:
            history[-1][1] = f"❌ Erreur : {str(e)}"
        return history, None

def delete_selected_conversation(conversation_dropdown, conversation_id):
    if conversation_id:
        delete_conversation(conversation_id)
        conversations = get_all_conversations()
        new_dropdown = gr.Dropdown(
            choices=conversations,
            label="Historique des conversations"
        ) if conversations else gr.Dropdown(choices=[], label="Historique des conversations")
        return new_dropdown, "✅ Conversation supprimée."
    return conversation_dropdown, "Aucune conversation sélectionnée."

def main():
    global current_conversation_id

    cleanup_audio_files()

    with gr.Blocks(title="RAG Voice Assistant", theme=gr.themes.Soft()) as rag_application:
        gr.Markdown("""
        # 🎤 RAG Voice Assistant
        ### Parlez, posez des questions sur vos PDFs, écoutez les réponses !
        *Powered by Whisper (ASR) + Mistral (LLM) + gTTS (TTS)*
        """)

        with gr.Row():
            pdf_upload = gr.File(
                label="📄 Uploader PDF(s)",
                file_types=['.pdf'],
                type="filepath",
                file_count="multiple"
            )
            upload_btn = gr.Button("📥 Charger", variant="primary", size="sm")
            new_conv_btn = gr.Button("🆕 Nouveau", variant="secondary", size="sm")
            auto_speak_checkbox = gr.Checkbox(
                label="🔊 Auto-parler les réponses",
                value=True,
                info="Cochez pour que les réponses soient lues automatiquement"
            )

        upload_status = gr.Textbox(label="Statut", interactive=False, visible=True)

        with gr.Row():
            conversation_dropdown = gr.Dropdown(
                choices=get_all_conversations(),
                label="📜 Historique des conversations",
                info="Sélectionnez une conversation existante"
            )
            delete_conv_btn = gr.Button("🗑️ Supprimer", variant="stop", size="sm")

        delete_status = gr.Textbox(label="Statut suppression", interactive=False, visible=False)

        chatbot = gr.Chatbot(label="Conversation", height=450)

        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="Votre question",
                    lines=1,
                    placeholder="Posez une question... ou parlez !",
                    show_label=False
                )
            with gr.Column(scale=1):
                temperature_slider = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0.5,
                    step=0.1,
                    label="Température",
                    info="0=Précis | 1=Créatif"
                )

        audio_input = gr.Audio(
            label="🎤 Parlez ici (ou tapez votre question)",
            sources=["microphone"],
            type="numpy",
            streaming=False
        )

        audio_output = gr.Audio(
            label="🔊 Réponse audio",
            type="filepath",
            autoplay=True,
            visible=True
        )

        conversation_dropdown.change(
            fn=load_selected_conversation,
            inputs=[conversation_dropdown],
            outputs=[chatbot]
        )

        delete_conv_btn.click(
            fn=delete_selected_conversation,
            inputs=[conversation_dropdown, conversation_dropdown],
            outputs=[conversation_dropdown, delete_status]
        )

        upload_btn.click(
            fn=upload_pdf,
            inputs=[pdf_upload, conversation_dropdown],
            outputs=[upload_status, conversation_dropdown, chatbot, audio_output]
        )

        def create_new_conversation():
            conv_id = create_conversation()
            conversations = get_all_conversations()
            new_dropdown = gr.Dropdown(
                choices=conversations,
                value=conv_id,
                label="📜 Historique des conversations"
            )
            return new_dropdown, "✅ Nouvelle conversation créée.", []

        new_conv_btn.click(
            fn=create_new_conversation,
            inputs=[],
            outputs=[conversation_dropdown, upload_status, chatbot]
        )

        def handle_user_input(text, audio, chat_history, conv_id):
            new_audio_input = gr.update(value=None, interactive=True)
            cleared_text, new_history, _, _ = process_user_input(
                text, audio, chat_history, conv_id
            )

            return cleared_text, new_history, new_audio_input

        query_input.submit(
            fn=handle_user_input,
            inputs=[query_input, audio_input, chatbot, conversation_dropdown],
            outputs=[query_input, chatbot, audio_input]
        ).then(
            fn=generate_and_speak_response,
            inputs=[chatbot, temperature_slider, conversation_dropdown, auto_speak_checkbox],
            outputs=[chatbot, audio_output]
        )

        audio_input.change(
            fn=handle_user_input,
            inputs=[query_input, audio_input, chatbot, conversation_dropdown],
            outputs=[query_input, chatbot, audio_input]
        ).then(
            fn=generate_and_speak_response,
            inputs=[chatbot, temperature_slider, conversation_dropdown, auto_speak_checkbox],
            outputs=[chatbot, audio_output]
        )

    rag_application.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        share=False
    )

if __name__ == "__main__":
    main()
