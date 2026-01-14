import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { MessageCircle, X, Send, Paperclip, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { askChatbot } from "@/services/chatbot";

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}

const welcomeMessage: ChatMessage = {
  id: "welcome",
  role: "assistant",
  content:
    "Hello! I'm your GlucoAI assistant. I can help answer questions about diabetes management, blood sugar levels, and healthy lifestyle choices.\n\n⚠️ I provide AI-assisted insights only, not medical diagnosis.",
  timestamp: new Date(),
};

export function FloatingChatButton() {
  const [isOpen, setIsOpen] = useState(false);

  const [messages, setMessages] = useState<ChatMessage[]>([welcomeMessage]);
  const [message, setMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    if (isOpen) scrollToBottom();
  }, [messages, isLoading, isOpen]);

  const handleSend = async () => {
    if ((!message.trim() && !uploadedFile) || isLoading) return;

    const question = message.trim();

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: "user",
      content: uploadedFile ? `[Uploaded: ${uploadedFile.name}]\n${question}` : question,
      timestamp: new Date(),
    };

    // show user msg
    setMessages((prev) => [...prev, userMessage]);

    // clear input
    setMessage("");
    setIsLoading(true);

    // keep file for request
    const fileToSend = uploadedFile;
    setUploadedFile(null);

    try {
      const data = await askChatbot(question, fileToSend);

      const botMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: data?.answer || "No answer received.",
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, botMessage]);
    } catch (err: any) {
      console.error("Chatbot widget error:", err);

      const botMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: err?.message || "Server error. Please try again.",
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, botMessage]);
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
    <>
      {/* Chat Panel */}
      <div
        className={cn(
          "fixed bottom-24 right-4 z-50 w-[380px] max-w-[calc(100vw-2rem)] transition-all duration-300 transform",
          isOpen
            ? "opacity-100 translate-y-0 pointer-events-auto"
            : "opacity-0 translate-y-4 pointer-events-none"
        )}
      >
        <div className="glass-card overflow-hidden shadow-2xl shadow-primary/10">
          {/* Header */}
          <div className="bg-gradient-to-r from-primary/20 to-info/20 px-4 py-3 border-b border-border/50">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="relative">
                  <div className="w-10 h-10 rounded-full bg-gradient-to-br from-primary to-info flex items-center justify-center">
                    <MessageCircle className="h-5 w-5 text-primary-foreground" />
                  </div>
                  <span className="absolute bottom-0 right-0 w-3 h-3 bg-success rounded-full border-2 border-card" />
                </div>
                <div>
                  <h3 className="font-semibold text-sm">GlucoAI Assistant</h3>
                  <p className="text-xs text-muted-foreground">
                    Diabetes support only
                  </p>
                </div>
              </div>
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8"
                onClick={() => setIsOpen(false)}
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          </div>

          {/* Messages Area */}
          <div className="h-80 overflow-y-auto p-4 space-y-4">
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={cn(
                  "flex gap-3",
                  msg.role === "user" ? "flex-row-reverse" : "flex-row"
                )}
              >
                <div
                  className={cn(
                    "w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center",
                    msg.role === "user"
                      ? "bg-primary/20"
                      : "bg-gradient-to-br from-primary to-info"
                  )}
                >
                  <MessageCircle
                    className={cn(
                      "h-4 w-4",
                      msg.role === "user"
                        ? "text-primary"
                        : "text-primary-foreground"
                    )}
                  />
                </div>

                <div className="flex-1">
                  <div
                    className={cn(
                      "rounded-2xl px-4 py-3 text-sm whitespace-pre-wrap",
                      msg.role === "user"
                        ? "bg-primary/10 rounded-tr-sm text-right"
                        : "bg-secondary/50 rounded-tl-sm"
                    )}
                  >
                    {msg.content}
                  </div>
                  <p
                    className={cn(
                      "text-xs text-muted-foreground mt-1 ml-1",
                      msg.role === "user" && "text-right mr-1 ml-0"
                    )}
                  >
                    {msg.timestamp.toLocaleTimeString([], {
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </p>
                </div>
              </div>
            ))}

            {isLoading && (
              <div className="flex gap-3">
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-primary to-info flex-shrink-0 flex items-center justify-center">
                  <MessageCircle className="h-4 w-4 text-primary-foreground" />
                </div>
                <div className="bg-secondary/50 rounded-2xl rounded-tl-sm px-4 py-3">
                  <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div className="p-4 border-t border-border/50 bg-card/50">
            <div className="flex gap-2">
              {/* hidden file input */}
              <input
                type="file"
                ref={fileInputRef}
                accept=".pdf"
                onChange={handleFileSelect}
                className="hidden"
              />

              <Button
                variant="ghost"
                size="icon"
                className="flex-shrink-0 h-10 w-10"
                onClick={() => fileInputRef.current?.click()}
                disabled={isLoading}
              >
                <Paperclip className="h-4 w-4" />
              </Button>

              <div className="flex-1 relative">
                <input
                  type="text"
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      handleSend();
                    }
                  }}
                  placeholder="Ask about diabetes..."
                  className="w-full h-10 px-4 rounded-lg bg-secondary/50 border border-border/50 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary/50"
                  disabled={isLoading}
                />
              </div>

              <Button
                variant="glow"
                size="icon"
                className="flex-shrink-0 h-10 w-10"
                onClick={handleSend}
                disabled={isLoading || (!message.trim() && !uploadedFile)}
              >
                <Send className="h-4 w-4" />
              </Button>
            </div>

            <p className="text-xs text-muted-foreground text-center mt-2">
              {uploadedFile ? `Attached: ${uploadedFile.name}` : "Upload PDF reports for analysis"}
            </p>
          </div>
        </div>
      </div>

      {/* Floating Button */}
      <Button
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          "fixed bottom-6 right-6 z-50 h-14 w-14 rounded-full shadow-lg transition-all duration-300",
          isOpen
            ? "bg-secondary hover:bg-secondary/80 rotate-0"
            : "bg-gradient-to-r from-primary to-info hover:shadow-primary/40 hover:scale-110"
        )}
        size="icon"
      >
        {isOpen ? (
          <X className="h-6 w-6" />
        ) : (
          <>
            <MessageCircle className="h-6 w-6" />
            <span className="absolute -top-1 -right-1 w-4 h-4 bg-success rounded-full border-2 border-background animate-pulse" />
          </>
        )}
      </Button>
    </>
  );
}