import React, { useCallback, useState } from "react";
import ChatHistory from "./Components/ChatHistory";
import MessageInput from "./Components/MessageInput";
import Header from "./Components/Header";
import "../../css/ChatPage.css"

const GREETING = {
    key: "greeting",
    role: "ai",
    content: "Привет! Приложите .xlsx с колонками Attribute и Value — нормализую значения по "
        + "выбранным стандартам и отвечу на вопросы о результате. Приложите PDF нормативного "
        + "документа — извлеку из него новый стандарт; это идёт в фоне, а чат тем временем "
        + "свободен. Правила можно дать на один прогон или добавить в стандарт насовсем.",
};

const ChatPage = () => {

    const [messages, setMessages] = useState([GREETING]);

    // Ссылка на функцию постоянна: опрос фоновых задач держит её в зависимостях эффекта.
    const addMessage = useCallback((newMsg) => {
        setMessages(prev => [...prev, newMsg]);
    }, []);

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
