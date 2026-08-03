import os
from pathlib import Path
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class LoadText:
    def __init__(self, path="data/processing"):
        path = Path(path)
        if not os.path.exists(path):
            os.mkdir(path)
        self._file_paths = list(path.glob("**/*.txt"))
        self._documents = []
        self._load_documents()
    
    def _load_documents(self):
        print("Loading files...")
        for txt_file in tqdm(self._file_paths):
            with open(txt_file, "r", encoding="utf-8") as file:
                text = file.read()
            document = Document(
                page_content=text,
                metadata={
                    "source": str(txt_file)
                }
            )
            self._documents.append(document)
    
    def split_documents(self) -> list[Document]:
        print("Splitting documents...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(self._documents)
        return chunks
