import os
import uuid
import json
import tempfile
import pytesseract
from PIL import Image

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.shortcuts import render

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_mistralai import ChatMistralAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain.docstore.document import Document


import logging
logger = logging.getLogger(__name__)
# ========== Config ==========
from dotenv import load_dotenv
load_dotenv()  
ms_api_key = os.getenv('MistralAI_API_TOKEN')

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# def get_embeddings():
#     return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# top of views.py (after imports)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

llm = ChatMistralAI(model="mistral-small", api_key=ms_api_key)

GENERAL_PROMPT = PromptTemplate.from_template("""
Use the following context to answer the question. The context comes from documents the user has uploaded.

Context:
{context}

Question: {question}

Answer in a clear and concise manner. If the question asks about specific documents, make sure to only use information from those documents.
If you can't find the answer in the provided context, say "I couldn't find that information in your documents".
""")

# ========== Views ==========

def index(request):
    return render(request, 'index.html')


@csrf_exempt
@require_POST
def upload_file(request):
    try:
        if not request.FILES:
            logger.warning("No files uploaded")
            return JsonResponse({'status': 'error', 'message': 'No files uploaded'}, status=400)

        
        if 'documents' not in request.session:
            request.session['documents'] = {}

        documents_data = request.session['documents']
        results = []
        logger.info("Processing uploaded files",documents_data)
        # embeddings = get_embeddings()
        for file in request.FILES.getlist('files'):
            file_name = file.name
            file_ext = os.path.splitext(file_name)[1][1:].lower()
            logger.info(f"Processing file: {file_name} with extension: {file_ext}")
            # Check for existing file with same name
            existing_file_id = None
            for file_id, file_data in documents_data.items():
                if file_data['name'].lower() == file_name.lower():
                    existing_file_id = file_id
                    break

            file_id = existing_file_id if existing_file_id else str(uuid.uuid4())

            # Delete existing FAISS files if they exist
            if existing_file_id:
                try:
                    faiss_files = [
                        f"{documents_data[existing_file_id]['path']}.faiss",
                        f"{documents_data[existing_file_id]['path']}.pkl"
                    ]
                    for faiss_file in faiss_files:
                        if os.path.exists(faiss_file):
                            os.remove(faiss_file)
                except Exception as e:
                    logger.error(f"Error deleting existing FAISS files: {e}")

            # Process the file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
                for chunk in file.chunks():
                    tmp_file.write(chunk)
                tmp_file_path = tmp_file.name

            text = extract_text_from_file(tmp_file_path, file_ext)
            os.unlink(tmp_file_path)

            if text:
                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_text(text)

                documents = [Document(page_content=chunk, metadata={"source": file_name}) for chunk in chunks]
                vectorstore = FAISS.from_documents(documents, embeddings)

                FAISS_SAVE_DIR = os.path.join('media', 'vectorstores')
                os.makedirs(FAISS_SAVE_DIR, exist_ok=True)
                faiss_path = os.path.join(FAISS_SAVE_DIR, file_id)
                vectorstore.save_local(faiss_path)

                documents_data[file_id] = {
                    'name': file_name,
                    'type': file_ext,
                    'path': faiss_path
                }

                results.append({
                    'id': file_id,
                    'name': file_name,
                    'type': file_ext,
                    'status': 'success',
                    'action': 'replaced' if existing_file_id else 'uploaded'
                })
            else:
                results.append({
                    'id': file_id,
                    'name': file_name,
                    'type': file_ext,
                    'status': 'error',
                    'message': 'Could not extract text'
                })

        request.session.modified = True
        logger.info("File upload and processing completed successfully")
        return JsonResponse({'status': 'success', 'files': results})

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        return JsonResponse({'status': 'error', 'message': 'An error occurred while processing your request.'}, status=500)


@csrf_exempt
@require_POST
def ask_question(request):
    try:
        data = json.loads(request.body)
        question = data.get('question')
        file_id = data.get('file_id')
        # embeddings = get_embeddings()
        print(f"Received question: {question}, file_id: {file_id}")
        if not question:
            return JsonResponse({'status': 'error', 'message': 'Question required'}, status=400)

        documents_data = request.session.get('documents', {})
        if not documents_data:
            return JsonResponse({'status': 'error', 'message': 'No uploaded documents found.'}, status=400)

        retrieved_docs = []

        if file_id:
            file_data = documents_data.get(file_id)
            if not file_data:
                return JsonResponse({'status': 'error', 'message': 'File ID not found'}, status=404)

            vectorstore = FAISS.load_local(file_data['path'], embeddings)
            retrieved_docs = vectorstore.similarity_search(question, k=4)

            prompt = PromptTemplate.from_template("""
                    Answer the question using ONLY the following context from the document "{document_name}".
                    If the answer isn't in this document, say "This information isn't available in {document_name}".

                    Context from {document_name}:
                    {context}

                    Question: {question}

                    Answer:""")

            chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
            answer = chain.invoke({
                "context": retrieved_docs,
                "question": question,
                "document_name": file_data['name']
            })

        else:
            for doc in documents_data.values():
                # vectorstore = FAISS.load_local(doc['path'], embeddings)
                vectorstore = FAISS.load_local(doc['path'], embeddings, allow_dangerous_deserialization=True)
                print(f"Searching in document: {doc['name']} at {doc['path']} {retrieved_docs}")
                retrieved_docs.extend(vectorstore.similarity_search(question, k=2))
                
            if not retrieved_docs:
                return JsonResponse({'status': 'success', 'answer': "I couldn't find any relevant information in your uploaded documents."})
            
            chain = create_stuff_documents_chain(llm=llm, prompt=GENERAL_PROMPT)
            answer = chain.invoke({
                "context": retrieved_docs,
                "question": question
            })

        return JsonResponse({'status': 'success', 'answer': answer})

    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


@csrf_exempt
@require_POST
def reset_session(request):
    try:
        if 'documents' in request.session:
            del request.session['documents']
            request.session.modified = True
        return JsonResponse({'status': 'success', 'message': 'Session reset'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

# ========== Helper Function ==========

def extract_text_from_file(file_path, file_type):
    try:
        if file_type == 'pdf':
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            return "\n\n".join(doc.page_content for doc in docs)

        elif file_type == 'docx':
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
            return docs[0].page_content if docs else None

        elif file_type in ['pptx', 'ppt']:
            from langchain_community.document_loaders import UnstructuredPowerPointLoader
            loader = UnstructuredPowerPointLoader(file_path)
            docs = loader.load()
            return docs[0].page_content if docs else None

        elif file_type in ['jpg', 'jpeg', 'png', 'bmp', 'gif']:
            img = Image.open(file_path)
            return pytesseract.image_to_string(img)

        else:  # .txt, .md etc.
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
    except Exception as e:
        print(f"Error processing file [{file_path}]: {e}")
        return None

@csrf_exempt
@require_POST
def delete_file(request):
    try:
        data = json.loads(request.body)
        file_id = data.get('file_id')

        if not file_id:
            return JsonResponse({'status': 'error', 'message': 'File ID required'}, status=400)

        documents_data = request.session.get('documents', {})
        if file_id not in documents_data:
            return JsonResponse({'status': 'error', 'message': 'File not found'}, status=404)

        # Delete the FAISS index files
        file_data = documents_data[file_id]
        try:
            faiss_files = [
                f"{file_data['path']}.faiss",
                f"{file_data['path']}.pkl"
            ]
            for faiss_file in faiss_files:
                if os.path.exists(faiss_file):
                    os.remove(faiss_file)
        except Exception as e:
            print(f"Error deleting FAISS files: {e}")

        # Remove from session
        del documents_data[file_id]
        request.session['documents'] = documents_data
        request.session.modified = True

        return JsonResponse({'status': 'success', 'message': 'File deleted'})

    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)