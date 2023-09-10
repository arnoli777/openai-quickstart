import gradio as gr
import random
import time

from typing import List

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

experts = {
    "衣服销售专家": None,
    "房地产销售专家": None,
    "游戏主播互动专家": None,
    "茶叶销售专家": None,
}

datas = {
    "衣服销售专家": "real_clothes_sales",
    "游戏主播互动专家": "real_games_sales",
    "茶叶销售专家": "real_tea_sales",
    "房地产销售专家": "real_estates_sales"
}


def initialize_sales_bot():
    for key, value in datas.items():
        db = FAISS.load_local(value, OpenAIEmbeddings())
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

        sales_bot = RetrievalQA.from_chain_type(llm,
                                               retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                         search_kwargs={"score_threshold": 0.8}))
        sales_bot.return_source_documents = True

        experts[key] = sales_bot


def sales_chat(message, expert, history):
    print(f"[export]{expert}")
    print(f"[message]{message}")
    print(f"[history]{history}")
    # TODO: 从命令行参数中获取
    enable_chat = True

    if expert not in experts:
        return "这个问题我要问问领导"

    ans = experts[expert]({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if ans["source_documents"] or enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    # 否则输出套路话术
    else:
        return "这个问题我要问问领导"
    

def launch_gradio():
    # 创建一个单选按钮，用于选择问答机器的类型
    demo = gr.Interface(
        fn=sales_chat,
        title="问答专家",
        inputs=[
            gr.inputs.Textbox(label="输入问题"),
            gr.Radio(["衣服销售专家", "房地产销售专家", "游戏主播互动专家", "茶叶销售专家"], label="选择专家机器人")],
        outputs=gr.outputs.Textbox(label="回答"),
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0", server_port=7780)


if __name__ == "__main__":
    # 初始化房产销售机器人
    initialize_sales_bot()
    # 启动 Gradio 服务
    launch_gradio()
