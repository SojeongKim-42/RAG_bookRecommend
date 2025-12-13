from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import torch
import os
import argparse


# Load data from CSV
def load_data():
    print("Loading data from CSV...")
    loader = CSVLoader(
        file_path="./data/combined_preprocessed.csv",
        encoding="utf-8",
        csv_args={"delimiter": ","},
        content_columns=["구분", "상품명", "책소개", "목차"],
    )
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)

    # 각 청크에 원본 문서 정보를 metadata에 추가
    for i, chunk in enumerate(chunks):
        row_num = chunk.metadata.get("row")
        # 같은 row의 원본 문서 찾기
        original_doc = data[row_num]

        # 원본 문서의 page_content에서 상품명, 구분 추출
        content_lines = original_doc.page_content.split("\n")
        for line in content_lines:
            if line.startswith("구분:"):
                chunk.metadata["구분"] = line.replace("구분:", "").strip()
            elif line.startswith("상품명:"):
                chunk.metadata["상품명"] = line.replace("상품명:", "").strip()

        # 청크 인덱스도 추가 (같은 책의 몇 번째 청크인지)
        chunk.metadata["chunk_id"] = i

    return chunks, data  # 원본 data도 함께 반환


# Initialize HuggingFace Embeddings model
def init_hf_embeddings():
    print("Initializing HuggingFace Embeddings model...")
    # the parameters for this model can be chosen from https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer
    # If the backend supports cuda, we use it
    if torch.cuda.is_available():
        model_kwargs = {"device": "cuda"}
    else:
        print("CUDA is not available, using CPU instead.")
        model_kwargs = {"device": "cpu"}

    embeddings = HuggingFaceEmbeddings(
        model_name="distiluse-base-multilingual-cased-v1", model_kwargs=model_kwargs
    )
    return embeddings


def faiss_vectorstore(chunks, embeddings):
    print("Creating FAISS vector store...")
    from langchain_community.vectorstores import FAISS

    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    # save vectorstore locally
    vectorstore.save_local("faiss_index")
    return vectorstore


if __name__ == "__main__":
    # query = user args
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="간호학 책을 추천해줘")
    args = parser.parse_args()

    # if a saved vectorstore exists, load it:
    if os.path.exists("faiss_index"):
        print("Loading existing FAISS vector store...")
        embeddings = init_hf_embeddings()  # 저장할 때와 같은 모델 사용
        vectorstore = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True,  # 신뢰할 수 있는 파일이므로 허용
        )
        # 원본 데이터도 다시 로드 (중복 제거용)
        chunks, original_data = load_data()
    else:
        chunks, original_data = load_data()
        embeddings = init_hf_embeddings()
        vectorstore = faiss_vectorstore(chunks, embeddings)

    query_faiss = args.query
    print(f"\nSearching for similar documents to the query: {query_faiss}\n")
    docs_faiss = vectorstore.similarity_search(
        query_faiss, k=10
    )  # 더 많이 검색 후 중복 제거

    # 같은 책의 청크를 그룹핑하고 중복 제거
    seen_rows = set()
    unique_books = []

    for doc in docs_faiss:
        row = doc.metadata.get("row")
        if row not in seen_rows:
            seen_rows.add(row)
            unique_books.append(
                {
                    "row": row,
                    "구분": doc.metadata.get("구분", "N/A"),
                    "상품명": doc.metadata.get("상품명", "N/A"),
                    "original_doc": original_data[row],
                    "matched_chunk": doc.page_content,
                }
            )

            if len(unique_books) >= 5:  # 상위 5개 책만
                break

    # 결과 출력
    print(f"=== 추천 도서 {len(unique_books)}권 ===\n")
    for i, book in enumerate(unique_books, 1):
        print(f"{i}. [{book['구분']}] {book['상품명']}")
        print(f"   Row: {book['row']}")
        print(f"   매칭된 내용: {book['matched_chunk'][:100]}...")
        print(f"\n   원본 전체 내용:")
        print(f"   {book['original_doc'].page_content[:300]}...")
        print("\n" + "=" * 80 + "\n")
