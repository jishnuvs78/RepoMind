import io
import zipfile
import requests
import frontmatter
import numpy as np

from sentence_transformers import SentenceTransformer
from minsearch import Index, VectorSearch

# Cache embedding model at module level to load only once
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer('multi-qa-distilbert-cos-v1')
    return _embedding_model

def read_repo_data(repo_owner, repo_name):
    
    prefix = 'https://codeload.github.com' 
    url = f'{prefix}/{repo_owner}/{repo_name}/zip/refs/heads/main'
    resp = requests.get(url)
    
    if resp.status_code != 200:
        raise Exception(f"Failed to download repository: {resp.status_code}")

    repository_data = []
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    
    for file_info in zf.infolist():
        filename = file_info.filename
        filename_lower = filename.lower()

        if not (filename_lower.endswith('.md') 
            or filename_lower.endswith('.mdx')):
            continue
    
        try:
            with zf.open(file_info) as f_in:
                content = f_in.read().decode('utf-8', errors='ignore')
                post = frontmatter.loads(content)
                data = post.to_dict()
                data['filename'] = filename
                repository_data.append(data)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    zf.close()
    return repository_data

def sliding_window(seq, size, step):
    if size<=0 or step<=0:
        raise ValueError("size and step must be +ve")

    seq_len = len(seq)
    result = []

    for i in range(0, seq_len, step):
        chunk = seq[i:i+size]
        result.append({'start' : i, 'chunk' : chunk})
        if i + size >= seq_len:
            break

    return result

def chunk_documents(docs, size=2000, step=1000):
    pytorch_img_models_chunks = []

    for doc in docs:
        doc_copy = doc.copy()
        doc_content = doc_copy.pop('content')
        chunks = sliding_window(doc_content, 2000, 1000)
        for chunk in chunks:
            chunk.update(doc_copy)
        pytorch_img_models_chunks.extend(chunks)
    
    return pytorch_img_models_chunks

def index_data(
        repo_owner,
        repo_name,
        filter=None,
        chunk=False,
        chunking_params=None,
    ):

    docs = read_repo_data(repo_owner, repo_name)

    if filter is not None:
        docs = [doc for doc in docs if filter(doc)]

    pytorch_img_models_chunks = []

    if chunk:
        if chunking_params is None:
            chunking_params = {'size': 2000, 'step': 1000}
        pytorch_img_models_chunks = chunk_documents(docs, **chunking_params)

    pytorch_img_index = Index(
    text_fields=["chunk", "title", "about", "name", "filename"],
    keyword_fields=[]
    )

    pytorch_img_index.fit(pytorch_img_models_chunks)

    embedding_model = get_embedding_model()

    pytorch_img_embeddings = []
    for d in pytorch_img_models_chunks:
        v = embedding_model.encode(d['chunk'])
        pytorch_img_embeddings.append(v)
    pytorch_img_embeddings = np.array(pytorch_img_embeddings)

    pytorch_img_vindex = VectorSearch()
    pytorch_img_vindex.fit(pytorch_img_embeddings, pytorch_img_models_chunks)

    return pytorch_img_index, pytorch_img_vindex
