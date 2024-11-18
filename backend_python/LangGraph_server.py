from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uuid

import os
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from langgraph.errors import NodeInterrupt

import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

#os.environ["OPENAI_API_VERSION"] = "2024-xx-xx-preview"
#os.environ["AZURE_OPENAI_ENDPOINT"] = "https://xxxx.openai.azure.com"
#os.environ["AZURE_OPENAI_API_KEY"] = "xxxx"


def get_local_db_path(session_id):
    return f'./sqlite_db/{session_id}.sqlite'

# SQLiteデータベースの初期化またはロード
def load_or_create_db(session_id):
    """
    ローカルフォルダでSQLiteデータベースをロードまたは新規作成します。

    :param session_id: セッションID（データベース名に使用）
    :param db_directory: SQLiteデータベースを保存するディレクトリ
    :return: SQLiteデータベース接続
    """

    local_db_path = get_local_db_path(session_id)
    print(f"Local database path: {local_db_path}")
    # データベース保存ディレクトリを確認または作成
    print(f"Checking database directory: {os.path.dirname(local_db_path)}")
    if not os.path.exists(os.path.dirname(local_db_path)):
        os.makedirs(os.path.dirname(local_db_path), exist_ok=True)

    if not os.path.exists(local_db_path):
        # 初回セッションの場合、DBファイルを新規作成
        print(f"No existing database for session {session_id}, creating new one.")
        conn = sqlite3.connect(local_db_path)
        # 必要なら初期化処理を実行
        conn.execute("CREATE TABLE IF NOT EXISTS example_table (id INTEGER PRIMARY KEY, data TEXT)")
        conn.commit()
        conn.close()
    else:
        print(f"Loaded existing database for session: {session_id}")

    # SQLiteデータベース接続を返す
    return sqlite3.connect(local_db_path, check_same_thread=False)


app = FastAPI()

# CORSの設定
origins = [
    "*" # フロントエンドのURLを指定
    # 必要に応じて他のオリジンを追加
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # すべてのHTTPメソッドを許可
    allow_headers=["*"],  # すべてのヘッダーを許可
)

# 各セッションの状態を保存
sessions = {}

# グラフを流れるStateの方の定義
class State(BaseModel):
    question_bool: bool = Field(default = False, description="不適切質問の抽出結果")
    message_type: str = Field(default = "", description="ユーザからの質問の分類結果")
    query: str = Field(default = "", description="これまでのプロンプト内容")
    AI_messages: str = Field(default = "", description="AIからのメッセージ内容")
    bool_time: bool = Field(default = False, description="時刻情報が含まれているかどうかの判定結果")
    advance_messages: str = Field(default = "", description="追加質問のユーザ回答")


# 日付か天気かの質問分類器の出力
class MessageType(BaseModel):
    message_type: str = Field(description="ユーザからの質問の分類結果", example="search")

# 天気の質問において、必要情報がすでに埋まっているかを判定する判定器の出力
class TimeType(BaseModel):
    message_type: bool = Field(description="ユーザ質問に時刻の情報が含めれているかどうかの真偽値(bool)", example=True)

# 今日、明日、明後日のどのワークフローを選択するかの分類器の出力
class ToolType(BaseModel):
    message_type: str = Field(description="どのツールを利用するかの判定結果", example="tool01")



# 日付か天気に関する質問をしているかどうかを判定
class date_weather_Type(BaseModel):
    message_type: bool = Field(description="天気か日付の質問をユーザがしているかどうかの真偽値(bool)", example=True)

# APIリクエストの定義
class AskRequest(BaseModel):
    session_id: str = None
    user_input: str
    

class ContinueRequest(BaseModel):
    session_id: str
    additional_input: str


# StateGraphの作成 (既存コードの関数やエッジを設定)
def initialize_graph(sqlite_db):
    # モデルの初期化
    model = AzureChatOpenAI(azure_deployment="gpt-4o", temperature=0)
    output_parser = StrOutputParser()

    def not_date_weather_interrupt(State):
        print("--not_date_weather_interrupt--")
        raise NodeInterrupt("天気か日付に関する質問をしてください")

    def interrupt(State):
        print("--interrupt--")
        if not State.bool_time:
            raise NodeInterrupt("天気を知りたい時間を入力してください")

        return State


    def chat_w1(State):
        print("--chat_w1--")
        if State.query:
            sys_prompt = "あなたはユーザからの質問を繰り返してください。その後、質問に回答してください。ただし今日の午前は雨で、午後は雪です"

            prompt = None
            if not State.advance_messages:
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system",sys_prompt),
                        ("human", "{user_input}")
                    ]
                )
            else:
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system",sys_prompt),
                        ("human", "{user_input}"),
                        ("assistant", "天気を知りたい時間を入力してください（例：「午前中」「20時」など）: "),
                        ("human",State.advance_messages)
                    ]
                )

            chain = prompt | model | output_parser

            dict = {
                    "query":State.query,
                    "AI_messages": chain.invoke({"user_input": State.query})
                    }

            return dict
        return {
            "AI_messages": "No user input provided"
                }

    def chat_d1(State):
        print("--chat_d1--")
        if State.query:
            sys_prompt = "あなたはユーザの質問内容を繰り返し発言した後、それに対して回答してください。ただし今日は10/23です"
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system",sys_prompt),
                    ("human", "{user_input}")
                ]
            )

            chain = prompt | model | output_parser

            return {
                    "query":State.query,
                    "AI_messages": chain.invoke({"user_input": State.query})
                    }
        return {
            "AI_messages": "No user input provided"
                }


    def chat_w2(State):
        print("--chat_w2--")
        if State.query:
            sys_prompt = "あなたはユーザからの質問を繰り返してください。その後、質問に回答してください。ただし明日の午前は曇りで、午後は霰です"

            prompt = None
            if not State.advance_messages:
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system",sys_prompt),
                        ("human", "{user_input}")
                    ]
                )
            else:
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system",sys_prompt),
                        ("human", "{user_input}"),
                        ("assistant", "天気を知りたい時間を入力してください（例：「午前中」「20時」など）: "),
                        ("human",State.advance_messages)
                    ]
                )

            chain = prompt | model | output_parser

            dict = {
                    "query":State.query,
                    "AI_messages": chain.invoke({"user_input": State.query})
                    }

            return dict
        return {
            "AI_messages": "No user input provided"
                }



    def chat_d2(State):
        print("--chat_d2--")
        if State.query:
            sys_prompt = "あなたはユーザの質問内容を繰り返し発言した後、それに対して回答してください。ただし明日は10/24です"
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system",sys_prompt),
                    ("human", "{user_input}")
                ]
            )

            chain = prompt | model | output_parser

            return {
                    "query":State.query,
                    "AI_messages": chain.invoke({"user_input": State.query})
                    }
        return {
            "AI_messages": "No user input provided"
                }


    def chat_w3(State):
        print("--chat_w3--")
        if State.query:
            sys_prompt = "あなたはユーザからの質問を繰り返してください。その後、質問に回答してください。ただし明後日の午前は晴れで、午後は霧です"

            prompt = None
            if not State.advance_messages:
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system",sys_prompt),
                        ("human", "{user_input}")
                    ]
                )
            else:
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system",sys_prompt),
                        ("human", "{user_input}"),
                        ("assistant", "天気を知りたい時間を入力してください（例：「午前中」「20時」など）: "),
                        ("human",State.advance_messages)
                    ]
                )

            chain = prompt | model | output_parser

            dict = {
                    "query":State.query,
                    "AI_messages": chain.invoke({"user_input": State.query})
                    }

            return dict
        return {
            "AI_messages": "No user input provided"
                }

    def chat_d3(State):
        print("--chat_d3--")
        if State.query:
            sys_prompt = "あなたはユーザの質問内容を繰り返し発言した後、それに対して回答してください。ただし明後日は10/25です"
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system",sys_prompt),
                    ("human", "{user_input}")
                ]
            )

            chain = prompt | model | output_parser

            return {
                    "query":State.query,
                    "AI_messages": chain.invoke({"user_input": State.query})
                    }
        return {
            "AI_messages": "No user input provided"
                }


    def response(State):
        print("--response--")
        return State
    
    # 日付か天気かの質問分類器の出力
    def classify(State):
        print("--classify--")
        classifier = model.with_structured_output(MessageType)

        # プロンプトの作成
        classification_prompt = """
        ## You are a message classifier.
        ## ユーザが天気に関しての質問をしていたら"weather"と返答してください。
        ##　それ以外の質問をしていたら、"day"と返答してください。

        """

        if State.query:
            prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system",classification_prompt),
                        ("human", "{user_input}")
                    ]
                )

            chain = prompt | classifier

            return {
                "message_type": chain.invoke({"user_input": State.query}).message_type,
                "query": State.query
                }
        else:
            return {"AI_messages": "No user input provided"}

    # 天気の質問において、必要情報がすでに埋まっているかを判定する判定器の出力
    def classify_time(State):
        print("--classify_time--")
        classifier_time = model.with_structured_output(TimeType)
        # プロンプトの作成
        classification_prompt = """
        ## You are a message classifier.
        ## ユーザが、日付以外の時間を指定して質問している場合（例えば、「午前」「午後」「12時」「5:20」などがある場合）はTrueと返答してください。
        ## そうでない場合はFalseと返答してください。

        TrueかFalse以外では回答しないでください。
        """


        if State.query:
            if State.advance_messages:
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system",classification_prompt),
                        ("human", "{user_input}ただし、{advance_messages}")
                    ]
                )

            else:
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system",classification_prompt),
                        ("human", "{user_input}")
                    ]
                )

            chain = prompt | classifier_time

            if State.advance_messages:
                dicts = {
                    "bool_time": chain.invoke({"user_input": State.query, "advance_messages": State.advance_messages}).message_type,
                    }
                return dicts

            else:
                dicts = {
                    "bool_time": chain.invoke({"user_input": State.query}).message_type,
                    }
                return dicts
        else:
            return {"AI_messages": "No user input provided"}

    # 今日、明日、明後日のどのワークフローを選択するかの分類器の出力
    def select_tool(State):
        print("--select_tool--")
        tools = model.with_structured_output(ToolType)
        # プロンプトの作成
        classification_prompt = """
        ## You are a message classifier.
        ## 今日についての質問の場合は"tool01"と返答してください。
        ## 明日についての質問の場合は"tool02"と返答してください。
        ## 明後日についての質問の場合は"tool03"と返答してください。
        """

        user_prompt = """
        # ユーザからの質問内容
        {user_input}
        """

        if State.query:
            prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system",classification_prompt),
                        ("human", user_prompt)
                    ]
                )

            chain = prompt | tools

            return {
                "message_type": chain.invoke({"user_input": State.query}).message_type,
                }

        else:
            return {"AI_messages": "No user input provided"}

    # 日付か天気に関する質問をしているかどうかを判定
    def date_weather(State):
        tools = model.with_structured_output(date_weather_Type)
        print("--date_weather--")
        # プロンプトの作成
        classification_prompt = """
        ## You are a message classifier.
        ## このチャットボットは日付か天気に関する質問しか答えることはできません。
        ## それ以外の質問には答えることができません。
        ## そのため、それ以外の質問をしていた場合は"False"と返答してください。
        ## 日付か天気に関する質問の場合は"True"と返答してください。

        ##　ユーザからの質問内容に日付や天気の情報が入っていたとしても、最終的な質問内容が天気や日付を回答するものでない場合は"False"と返答してください。
        """

        user_prompt = """
        # ユーザからの質問内容
        {user_input}
        """

        if State.query:
            prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system",classification_prompt),
                        ("human", user_prompt)
                    ]
                )

            chain = prompt | tools

            return {
                "question_bool": chain.invoke({"user_input": State.query}).message_type,
                }
        else:
            return {"AI_messages": "No user input provided"}

    # tool1
    #ノードの追加
    graph_builder = StateGraph(State)
    graph_builder.add_node("date_weather", date_weather)
    graph_builder.add_node("not_date_weather_interrupt", not_date_weather_interrupt)
    graph_builder.add_node("select_tool", select_tool)
    graph_builder.add_node("classify1", classify)
    graph_builder.add_node("classify_time_1", classify_time)
    graph_builder.add_node("interrupt_1", interrupt)
    graph_builder.add_node("chat_w1", chat_w1)
    graph_builder.add_node("chat_d1", chat_d1)
    graph_builder.add_node("response1", response)

    # エッジの追加
    graph_builder.add_edge("classify_time_1", "interrupt_1")
    graph_builder.add_edge("chat_d1", "response1")
    graph_builder.add_edge("chat_w1", "response1")

    # 条件分岐
    graph_builder.add_conditional_edges("date_weather", lambda state: state.question_bool, {True: "select_tool", False: "not_date_weather_interrupt"})
    graph_builder.add_conditional_edges("classify1", lambda state: state.message_type, {"weather": "classify_time_1", "day": "chat_d1"})
    graph_builder.add_conditional_edges("interrupt_1", lambda state: state.bool_time, {True: "chat_w1", False: "classify_time_1"})

    # tool2
    #ノードの追加
    graph_builder.add_node("classify2", classify)
    graph_builder.add_node("classify_time_2", classify_time)
    graph_builder.add_node("interrupt_2", interrupt)
    graph_builder.add_node("chat_w2", chat_w2)
    graph_builder.add_node("chat_d2", chat_d2)
    graph_builder.add_node("response2", response)

    # エッジの追加
    graph_builder.add_edge("classify_time_2", "interrupt_2")
    graph_builder.add_edge("chat_d2", "response2")
    graph_builder.add_edge("chat_w2", "response2")
    # 条件分岐
    graph_builder.add_conditional_edges("classify2", lambda state: state.message_type, {"weather": "classify_time_2", "day": "chat_d2"})
    graph_builder.add_conditional_edges("interrupt_2", lambda state: state.bool_time, {True: "chat_w2", False: "classify_time_2"})

    # tool3
    #ノードの追加
    graph_builder.add_node("classify3", classify)
    graph_builder.add_node("classify_time_3", classify_time)
    graph_builder.add_node("interrupt_3", interrupt)
    graph_builder.add_node("chat_w3", chat_w3)
    graph_builder.add_node("chat_d3", chat_d3)
    graph_builder.add_node("response3", response)

    # エッジの追加
    graph_builder.add_edge("classify_time_3", "interrupt_3")
    graph_builder.add_edge("chat_d3", "response3")
    graph_builder.add_edge("chat_w3", "response3")
    # 条件分岐
    graph_builder.add_conditional_edges("classify3", lambda state: state.message_type, {"weather": "classify_time_3", "day": "chat_d3"})
    graph_builder.add_conditional_edges("interrupt_3", lambda state: state.bool_time, {True: "chat_w3", False: "classify_time_3"})

    # All
    #ノードの追加
    graph_builder.add_node("response", response)

    # エッジの追加
    graph_builder.add_edge("response1", "response")
    graph_builder.add_edge("response2", "response")
    graph_builder.add_edge("response3", "response")
    # 条件分岐
    graph_builder.add_conditional_edges("select_tool", lambda state: state.message_type, {"tool01": "classify1", "tool02": "classify2", "tool03": "classify3"})

    # 開始位置、終了位置の指定
    graph_builder.set_entry_point("date_weather")
    graph_builder.set_finish_point("response")

    # グラフ構築
    #memory = MemorySaver()
    memory = SqliteSaver(sqlite_db)
    graph = graph_builder.compile(checkpointer=memory)
    return graph


@app.post("/ask")
async def ask(request: AskRequest):
    # 新しいセッションIDを生成
    if request.session_id == "None":
        session_id = str(uuid.uuid4())
    else:
        session_id = request.session_id
    user_input = request.user_input

    sqlite_db = load_or_create_db(session_id)
    graph = initialize_graph(sqlite_db)

    print("Session ID:", session_id)
    print("User Input:", user_input)

    # Stateの初期化
    state = {
            "question_bool": False,
            "message_type": "",
            "query": user_input,
            "AI_messages": "",
            "bool_time": False,
            "advance_messages": ""
        }
    thread_config = {"configurable": {"thread_id": session_id}}

    # イベントのリストと中断フラグ
    event_list = []
    interrupt = False
    last_content = None

    # LangGraphからのイベントを取得し、中断チェック
    for event in graph.stream(state, thread_config):
        #グラフ途中の中断を検出
        event_list.append(event)
        if "__interrupt__" in event:
            interrupt = True
            break
        # 最後の 'response' から 'messages' の content を取得
        if "response" in event and "AI_messages" in event["response"]:
            last_content = event["response"]["AI_messages"]

    # セッションの状態を保存
    sessions[session_id] = {
        "initial_input": user_input,
        "state": state,
        "interrupt": interrupt,
        "event_list": event_list,
        "interrupt_event": list(event_list[-2].keys()),
    }

    sqlite_db.commit()
    sqlite_db.close()

    # 応答と中断フラグを返す
    return {
        "session_id": session_id,
        "response": last_content,
        "interrupt": interrupt,
        "interrupt_event": list(event_list[-2].keys())
    }

@app.post("/continue")
async def continue_conversation(request: ContinueRequest):
    session_id = request.session_id
    additional_input = request.additional_input

    sqlite_db = load_or_create_db(session_id)
    graph = initialize_graph(sqlite_db)

    print("Session ID:", session_id)
    print("Additional Input:", additional_input)

    # セッションが存在するか確認
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session_data = sessions[session_id]
    interrupt = session_data["interrupt"]

    add_state = {}
    if session_data["interrupt_event"][0] == "date_weather":
        #assistant_message = "天気か日付に関する質問をしてください"
        # Stateの更新
        add_state = {
            "question_bool": False,
            "message_type": "",
            "query": additional_input,
            "AI_messages": "",
            "bool_time": False,
            "advance_messages": ""
        }

    elif "classify_time" in session_data["interrupt_event"][0]:
        #assistant_message = "天気を知りたい時間を入力してください（例：「午前中」「20時」など）"
        # Stateの更新
        add_state = {
            "query": session_data["initial_input"],
            "advance_messages":additional_input,
        }

    # 中断がない場合のエラー処理
    if not interrupt:
        return {"response": "No interrupt in this session."}


    # 直前の状態を取得して分岐
    all_states = []
    for state in graph.get_state_history({"configurable": {"thread_id": session_id}}):
        all_states.append(state)

    to_replay = all_states[1] if len(all_states) > 1 else all_states[0]
    branch_config = graph.update_state(config=to_replay.config, values=add_state)

    # LangGraphの再実行
    last_content = None
    event_list = []
    for event in graph.stream(None, branch_config):
        event_list.append(event)
        if "__interrupt__" in event:
            interrupt = True
            break
        # 最後の 'response' から 'messages' の content を取得
        if "response" in event and "AI_messages" in event["response"]:
            last_content = event["response"]["AI_messages"]
    
    if last_content:
        interrupt = False

    if session_data["interrupt_event"][0] == "date_weather":
        # セッションの状態を更新
        sessions[session_id] = {
            "initial_input": additional_input,
            "state": state,
            "interrupt": interrupt,
            "event_list": event_list,
            "interrupt_event": list(event_list[-2].keys()),
        }

    elif "classify_time" in session_data["interrupt_event"][0]:
        sessions[session_id]["interrupt"] = interrupt
        sessions[session_id]["event_list"] = event_list
        sessions[session_id]["interrupt_event"] = list(event_list[-2].keys())

    sqlite_db.commit()
    sqlite_db.close()

    # 応答を返却
    return {
        "session_id": session_id,
        "response": last_content,
        "interrupt": interrupt,
        "interrupt_event": list(event_list[-2].keys())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("LangGraph_server:app", host="0.0.0.0", port=8002, reload=True)