import { useState, useRef, useEffect } from "react";
import { Layout } from "@/components/layout/Layout";
import { Button } from "@/components/ui/button";
import {
  MessageCircle,
  Send,
  Paperclip,
  Bot,
  User,
  Upload,
  X,
  Loader2,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { askChatbot } from "@/services/chatbot";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}

const welcomeMessage: Message = {
  id: "welcome",
  role: "assistant",
  content:
    "Hello! I'm your GlucoAI assistant, specialized in diabetes-related questions. I can help you understand blood sugar management, lifestyle tips, medication guidance, and more.\n\n⚠️ Please note: I provide AI-assisted insights only, not medical diagnosis. Always consult healthcare professionals for medical decisions.\n\nHow can I help you today?",
  timestamp: new Date(),
};

export default function Chatbot() {
  const [messages, setMessages] = useState<Message[]>([welcomeMessage]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSend = async () => {
    if (!input.trim() && !uploadedFile) return;
    if (isLoading) return;

    const question = input.trim();

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: uploadedFile ? `[Uploaded: ${uploadedFile.name}]\n${question}` : question,
      timestamp: new Date(),
    };

    // ✅ show user msg immediately
    setMessages((prev) => [...prev, userMessage]);

    // reset input
    setInput("");
    setIsLoading(true);

    // keep file ref before clearing state
    const fileToSend = uploadedFile;
    setUploadedFile(null);

    try {
      // ✅ API Call to backend
      const data = await askChatbot(question, fileToSend);

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: data?.answer || "No answer received.",
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err: any) {
      console.error("Chatbot API Error:", err);

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content:
          err?.message ||
          "Sorry, something went wrong while contacting the chatbot server.",
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type === "application/pdf") {
      setUploadedFile(file);
    }
  };

  return (
    <Layout>
      <div className="container mx-auto px-4 py-6 h-[calc(100vh-4rem)] flex flex-col">
        {/* Header */}
        <div className="text-center mb-6">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 border border-primary/20 mb-4">
            <MessageCircle className="h-4 w-4 text-primary" />
            <span className="text-sm text-primary font-medium">
              AI Diabetes Assistant
            </span>
          </div>
          <h1 className="text-2xl md:text-3xl font-bold mb-2">GlucoAI Chatbot</h1>
          <p className="text-muted-foreground text-sm">
            Ask questions about diabetes management • Upload PDFs for analysis
          </p>
        </div>

        {/* Chat Container */}
        <div className="flex-1 glass-card flex flex-col overflow-hidden">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 md:p-6 space-y-6">
            {messages.map((message) => (
              <div
                key={message.id}
                className={cn(
                  "flex gap-3",
                  message.role === "user" ? "flex-row-reverse" : "flex-row"
                )}
              >
                <div
                  className={cn(
                    "w-10 h-10 rounded-full flex-shrink-0 flex items-center justify-center",
                    message.role === "user"
                      ? "bg-primary/20"
                      : "bg-gradient-to-br from-primary to-[hsl(200,85%,50%)]"
                  )}
                >
                  {message.role === "user" ? (
                    <User className="h-5 w-5 text-primary" />
                  ) : (
                    <Bot className="h-5 w-5 text-primary-foreground" />
                  )}
                </div>

                <div
                  className={cn(
                    "max-w-[80%] md:max-w-[70%] rounded-2xl px-4 py-3",
                    message.role === "user"
                      ? "bg-primary/10 rounded-tr-sm"
                      : "bg-secondary/50 rounded-tl-sm"
                  )}
                >
                  <div className="text-sm whitespace-pre-wrap">{message.content}</div>
                  <p className="text-xs text-muted-foreground mt-2">
                    {message.timestamp.toLocaleTimeString([], {
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </p>
                </div>
              </div>
            ))}

            {isLoading && (
              <div className="flex gap-3">
                <div className="w-10 h-10 rounded-full bg-gradient-to-br from-primary to-[hsl(200,85%,50%)] flex items-center justify-center">
                  <Bot className="h-5 w-5 text-primary-foreground" />
                </div>
                <div className="bg-secondary/50 rounded-2xl rounded-tl-sm px-4 py-3">
                  <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Uploaded File Preview */}
          {uploadedFile && (
            <div className="mx-4 md:mx-6 mb-2 p-3 rounded-lg bg-secondary/50 border border-border/50 flex items-center gap-3">
              <Upload className="h-5 w-5 text-primary" />
              <span className="text-sm flex-1 truncate">{uploadedFile.name}</span>
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8"
                onClick={() => setUploadedFile(null)}
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          )}

          {/* Input Area */}
          <div className="p-4 md:p-6 border-t border-border/50">
            <div className="flex gap-3">
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileSelect}
                accept=".pdf"
                className="hidden"
              />

              <Button
                variant="outline"
                size="icon"
                className="flex-shrink-0"
                onClick={() => fileInputRef.current?.click()}
                disabled={isLoading}
              >
                <Paperclip className="h-5 w-5" />
              </Button>

              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) handleSend();
                }}
                placeholder="Ask about diabetes management..."
                className="flex-1 h-12 px-4 rounded-lg bg-secondary/50 border border-border/50 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary/50"
              />

              <Button
                variant="glow"
                size="icon"
                className="flex-shrink-0 h-12 w-12"
                onClick={handleSend}
                disabled={isLoading || (!input.trim() && !uploadedFile)}
              >
                <Send className="h-5 w-5" />
              </Button>
            </div>

            <p className="text-xs text-muted-foreground text-center mt-3">
              GlucoAI only responds to diabetes-related questions • Upload PDFs for detailed analysis
            </p>
          </div>
        </div>
      </div>
    </Layout>
  );
}