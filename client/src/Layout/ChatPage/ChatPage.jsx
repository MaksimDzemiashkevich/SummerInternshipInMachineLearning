import React, { useState } from "react";
import ChatHistory from "./Components/ChatHistory";
import MessageInput from "./Components/MessageInput";
import Header from "./Components/Header";
import "../../css/ChatPage.css"

const GREETING = {
    key: "greeting",
    role: "ai",
    content: "Привет! Приложите .xlsx с колонками Attribute и Value — нормализую значения по "
        + "загруженным стандартам. Можно просто поговорить, а можно вместе с файлом написать "
        + "свои правила: они применятся поверх стандартов, но только в этом запросе.",
};

const ChatPage = () => {

    const [messages, setMessages] = useState([GREETING]);

    const addMessage = (newMsg) => {
        setMessages(prev => [...prev, newMsg]);
    };

    return(
        <div className="chatPage">
            <Header>

            </Header>
            <ChatHistory items={messages} >

            </ChatHistory>
            <MessageInput onSend={addMessage} history={messages} >

            </MessageInput>
        </div>
    );
}

export default ChatPage;
