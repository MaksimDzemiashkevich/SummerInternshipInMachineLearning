import React from "react";
import { AntDesignOutlined, RedoOutlined, UserOutlined } from "@ant-design/icons";
import { Actions, Bubble, FileCard } from "@ant-design/x";
import { Avatar } from "antd";

const actionItems = (content) => [
    {
        key: "copy",
        label: "copy",
        actionRender: () => {
            return <Actions.Copy text={content} />;
        },
    },
    {
        key: "retry",
        icon: <RedoOutlined />,
        label: "Retry",
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
    ai: {
        //header: <h5>Ant Design X</h5>,
        avatar: () => <Avatar icon={<AntDesignOutlined />} />,
        //typing: { effect: "fade-in" },
        footer: (content) => (
            <Actions items={actionItems(content)} onClick={() => console.log(content)} />
        ),
    },
    user: (data) => ({
        placement: "end",
        avatar: () => <Avatar icon={<UserOutlined />} />,
        contentRender: (content) => (
            <div>
                {data.attachments?.length > 0 && (
                    <FileCard.List
                        items={data.attachments}
                        size="small"
                        style={{ marginBottom: 8 }}
                    />
                )}
                {content && <div>{content}</div>}
            </div>
        ),
        footer: (content) => (
            <Actions items={actionItemsUser(content)} />
        ),
    }),
};

const ChatHistory = ({ items }) => {
    return (
        <div className="ListBubles">
            <Bubble.List role={roleConfig} items={items} />
        </div>
    );
};

export default ChatHistory;