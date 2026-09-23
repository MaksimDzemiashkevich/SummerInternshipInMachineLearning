import React, { useState } from "react";
import ChatHistory from "./Components/ChatHistory";
import MessageInput from "./Components/MessageInput";
import Header from "./Components/Header";
import "../../css/ChatPage.css"

const GREETING = {
    key: "greeting",
    role: "ai",
    content: "Привет! Приложите .xlsx с колонками Attribute и Value — нормализую значения по "
        + "выбранным стандартам. Можно просто поговорить, а можно написать свои правила: они "
        + "лягут отдельной колонкой и будут действовать только в этом запросе. Забыли правило "
        + "— скажите, и я пересчитаю последний файл заново.",
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
