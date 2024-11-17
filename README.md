# LangGraphによる「AIエージェントWebアプリ」を作成する【Next.js】

詳しくはZennの記事をご覧ください

LangGraphによる「AIエージェントWebアプリ」を作成する【Next.js】


# 実行方法

## バックエンド

```
pip install -r requirements.txt
python LangGraph_server.py
```

## フロントエンド

`.env.local`を作成する。
```
NEXT_PUBLIC_BASE_URL=http://127.0.0.1:8002
```

```
pnpm i
pnpm build
pnpm dev
```

`http://localhost:3000/`にアクセス

