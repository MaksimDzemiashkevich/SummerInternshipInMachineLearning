import React from "react";
import ChatHistory from "./Components/ChatHistory";
import MessageInput from "./Components/MessageInput";
import Header from "./Components/Header";
import "../../css/ChatPage.css"

const ChatPage = () => {
    return(
        <div className="chatPage">
            <Header>

            </Header>
            <ChatHistory>

            </ChatHistory>
            <MessageInput>

            </MessageInput>
        </div>
    );
}

export default ChatPage;