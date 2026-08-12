import React, { useState } from "react";
import ChatHistory from "./Components/ChatHistory";
import MessageInput from "./Components/MessageInput";
import Header from "./Components/Header";
import "../../css/ChatPage.css"

const ChatPage = () => {

    const [messages, setMessages] = useState([{
        key: "1",
        role: "ai",
        content: "Hello World\nNext line\nTab\tindent",
    },
        {
            key: "2",
            role: "user",
            content: "Привет! Как дела?",
        },
    ]);

    const addMessage = (newMsg) => {
        setMessages(prev => [...prev, newMsg]);
    };

    return(
        <div className="chatPage">
            <Header>

            </Header>
            <ChatHistory items={messages} >

            </ChatHistory>
            <MessageInput onSend={addMessage} >

            </MessageInput>
        </div>
    );
}

export default ChatPage;