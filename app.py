from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import json
import threading
from rag_system import ProgrammerRAG
import uuid
import os

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production!
CORS(app)

# Store RAG instances per session
rag_instances = {}


def get_rag_for_session():
    """Get or create RAG instance for current session"""
    session_id = session.get('session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id

    if session_id not in rag_instances:
        try:
            rag_instances[session_id] = ProgrammerRAG()
            print(f"Created new RAG instance for session {session_id}")
        except Exception as e:
            print(f"Error creating RAG instance: {e}")
            return None

    return rag_instances.get(session_id)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '').strip()

    if not message:
        return jsonify({'error': 'Empty message'}), 400

    # Get RAG instance for this session
    rag = get_rag_for_session()
    if not rag:
        return jsonify({
            'response': "System error: Could not load AI system. Please make sure you've run build_embeddings.py first."
        })

    try:
        # Process the question
        response = rag.ask_question(message)
        return jsonify({'response': response})
    except Exception as e:
        print(f"Error processing chat: {e}")
        return jsonify({'response': f"Error: {str(e)}"})


@app.route('/api/clear', methods=['POST'])
def clear_chat():
    """Clear conversation history"""
    rag = get_rag_for_session()
    if rag:
        rag.conversation_history = []
    return jsonify({'status': 'cleared'})


@app.route('/api/status', methods=['GET'])
def status():
    """Check if RAG system is loaded"""
    rag = get_rag_for_session()
    return jsonify({'loaded': rag is not None})


if __name__ == '__main__':
    app.run(debug=True, port=5000)