import faiss
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter


exports = {
    "real_clothes_sales_data.txt": "real_clothes_sales",
    "real_games_sales_data.txt": "real_games_sales",
    "real_tea_sales_data.txt": "real_tea_sales"
}


for export in exports.keys():
    with open(export, 'r', encoding='utf-8') as f:
        real_estate_sales = f.read()

    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )


    docs = text_splitter.create_documents([real_estate_sales])

    db = FAISS.from_documents(docs, OpenAIEmbeddings())

    db.save_local(exports[export])
