'use client'

import { useState, useRef, useEffect } from 'react'
import { Send } from 'lucide-react'

// UI コンポーネント
const Button = ({ children, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement> & { children: React.ReactNode }) => (
  <button
    className="inline-flex items-center justify-center rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 bg-primary text-primary-foreground hover:bg-primary/90 h-10 px-4 py-2"
    {...props}
  >
    {children}
  </button>
)

const Input = ({ ...props }) => (
  <input
    className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
    {...props}
  />
)

const Card = ({ children, className, ...props }: { children: React.ReactNode, className?: string }) => (
  <div className={`rounded-lg border bg-card text-card-foreground shadow-sm ${className}`} {...props}>
    {children}
  </div>
)

const CardHeader = ({ children, ...props }: { children: React.ReactNode }) => (
  <div className="flex flex-col space-y-1.5 p-6" {...props}>
    {children}
  </div>
)

const CardTitle = ({ children, ...props }: { children: React.ReactNode }) => (
  <h3 className="text-2xl font-semibold leading-none tracking-tight" {...props}>
    {children}
  </h3>
)

const CardContent = ({ children, className, ...props }: { children: React.ReactNode, className?: string }) => (
  <div className={`p-6 pt-0 ${className}`} {...props}>
    {children}
  </div>
)

const CardFooter = ({ children, ...props }: { children: React.ReactNode }) => (
  <div className="flex items-center p-6 pt-0" {...props}>
    {children}
  </div>
)

const ChatBubble = ({ children, role }: { children: React.ReactNode, role: 'user' | 'assistant' }) => (
  <div className={`flex ${role === 'user' ? 'justify-end' : 'justify-start'} mb-4`}>
    <div className={`relative max-w-[70%] p-2 rounded-lg shadow-md ${role === 'user' ? 'bg-green-500 text-white' : 'bg-white text-black'}`}>
      {children}
    </div>
  </div>
)



type Message = {
  role: 'user' | 'assistant'
  content: string
}

export default function Chatbot() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [interruptSet, setInterruptSet] = useState<boolean>(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(scrollToBottom, [messages])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    setMessages(prev => [...prev, { role: 'user', content: input }]);
    setInput('');

    try {
      const endpoint = interruptSet ? '/continue' : '/ask';
      const response = await fetch(`${process.env.NEXT_PUBLIC_BASE_URL}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(interruptSet ? { session_id: sessionId, additional_input: input } : { user_input: input }),
      });


      
      if (!response.ok) throw new Error('ネットワークエラーが発生しました');

      const data = await response.json();
      if (data.response !== null) {
        setMessages(prev => [...prev, { role: 'assistant', content: data.response }]);
      }


      console.log('handleSubmit data.interrupt:', data.interrupt);
      console.log('handleSubmit sessionId:', data.session_id);
      console.log('handleSubmit interrupt_event:', data.interrupt_event);

      setSessionId(data.session_id);

      if (data.interrupt) {
        
        await handleInterrupt(data.session_id, data.interrupt_event);
      } else {
        // セッションが中断されていない場合、セッションIDをリセット
        setInterruptSet(false);
      }
    } catch (error) {
      console.error('エラー:', error);
      setMessages(prev => [...prev, { role: 'assistant', content: 'エラーが発生しました。もう一度お試しください。' }]);
    } 
  }

  const handleInterrupt = async (sessionId: string, interrupt_event: string[]) => {
    if (interrupt_event.includes('date_weather')) {
      setMessages(prev => [...prev, { role: 'assistant', content: '日付か天気の質問をしてください' }]);
    }else if (interrupt_event.some(event => event.includes('classify_time'))) {
      setMessages(prev => [...prev, { role: 'assistant', content: '天気を知りたい時間を入力してください（例：「午前中」「20時」など）' }]);
    }
    
    setInput(''); // 入力欄をクリア
    setSessionId(sessionId);
    setInterruptSet(true);

    console.log('handleInterrupt messages:', messages);

  }

  return (
    <Card className="w-full max-w-2xl mx-auto">
      <CardHeader>
        <CardTitle>AIエージェントチャットボット</CardTitle>
      </CardHeader>
      <CardContent className="h-[75vh] overflow-y-auto">
        {messages.map((message, index) => (
          <ChatBubble key={index} role={message.role}>
            {message.content}
          </ChatBubble>
        ))}
        <div ref={messagesEndRef} />
      </CardContent>
      <CardFooter>
        <form onSubmit={handleSubmit} className="flex w-full space-x-2">
          <Input
            ref={inputRef}
            value={input}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setInput(e.target.value)}
            placeholder="メッセージを入力..."
            aria-label="メッセージを入力"
          />
          <Button type="submit" aria-label="送信">
            <Send className="h-4 w-4" />
          </Button>
        </form>
      </CardFooter>
    </Card>
  )
}