import React from "react";
import { AntDesignOutlined, RedoOutlined, UserOutlined } from "@ant-design/icons";
import { Actions, Bubble, FileCard } from "@ant-design/x";
import { Avatar } from "antd";
import "../../../css/ChatHistory.css"

const actionItems = (content) => [
    {
        key: "copy",
        label: "copy",
        actionRender: () => {
            return <Actions.Copy text={content} />;
        },
    },
];

const actionItemsUser = (content) => [
    {
        key: "copy",
        label: "copy",
        actionRender: () => {
            return <Actions.Copy text={content} />;
        },
    },
];

const roleConfig = {
    ai: (data) => ({
        avatar: () => <Avatar icon={<AntDesignOutlined />} />,
        contentRender: (content) => (
            <div>
                {data.attachments?.length > 0 && (
                    <div style={{ marginBottom: 8 }}>
                        {data.attachments.map((file) => (
                            <FileCard
                                key={file.uid}
                                name={file.name}
                                byte={file.byte}
                                type={file.type}
                                src={file.url}
                                size="small"
                                onClick={() => handleFileClick(file)}
                            />
                        ))}
                </div>
                )}
                {content && <div style={{ whiteSpace: "pre-wrap" }}>{content}</div>}
            </div>
        ),
        footer: (content) => (
            <Actions items={actionItems(content)} />
        ),
    }),
    user: (data) => ({
        placement: "end",
        avatar: () => <Avatar icon={<UserOutlined />} />,
        contentRender: (content) => (
            <div>
                {data.attachments?.length > 0 && (
                    <div style={{ marginBottom: 8 }}>
                        {data.attachments.map((file) => (
                            <FileCard
                                key={file.uid}
                                name={file.name}
                                byte={file.byte}
                                type={file.type}
                                src={file.url}
                                size="small"
                                onClick={() => handleFileClick(file)}
                            />
                        ))}
                </div>
                )}
                {content && <div style={{ whiteSpace: "pre-wrap" }}>{content}</div>}
            </div>
        ),
        footer: (content) => (
            <Actions items={actionItemsUser(content)} />
        ),
    }),
};

const handleFileClick = (file) => {
    if (!file.url) return;

    const link = document.createElement("a");
    link.href = file.url;
    link.download = file.name;
    link.click();
};

const ChatHistory = ({ items }) => {
    return (
        <div className="ListBubles">
            <Bubble.List role={roleConfig} items={items} />
        </div>
    );
};

export default ChatHistory;