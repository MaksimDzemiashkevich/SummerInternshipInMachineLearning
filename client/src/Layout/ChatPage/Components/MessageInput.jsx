import { PaperClipOutlined } from '@ant-design/icons';
import { Attachments, Sender } from '@ant-design/x';
import { Button, Flex } from 'antd';
import React, { useRef, useState } from 'react';
import "../../../css/MessageInput.css"

const IconStyle = { fontSize: 16 };

const MessageInput = ({ onSend }) => {
    const [value, setValue] = useState('');
    const [items, setItems] = useState([]);
    const senderRef = useRef(null);
    const fileInputRef = useRef(null);

    const handleAttachClick = () => {
        fileInputRef.current?.click();
    };

    const handleFileChange = (e) => {
        const file = e.target.files?.[0];
        if (!file) return;

        const newItem = {
            uid: `${file.name}_${Date.now()}`,
            name: file.name,
            size: file.size,
            type: file.type,
            status: 'done',
            originFileObj: file,
            url: URL.createObjectURL(file),
        };

        setItems((prev) => [...prev, newItem]);
        e.target.value = '';
    };

    const handleAttachmentsChange = ({ file, fileList }) => {
        if (file.status === 'removed' && file.url?.startsWith('blob:')) {
            URL.revokeObjectURL(file.url);
        }
        setItems(fileList);
    };

    const handleSubmit = (v) => {
        console.log('Отправлено:', v, 'файлы:', items);

        const newMessage = {
            key: `msg_${Date.now()}`,
            role: 'user',
            content: v,
            attachments: items.map((item) => ({
                uid: item.uid,
                name: item.name,
                byte: item.size,
                type: item.type,
                url: item.url,
            })),
        };

        onSend(newMessage);

        setValue('');
        setItems([]);
    };

    return (
        <>
            <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                style={{ display: 'none' }}
            />

            <Sender
                className='customSender'
                ref={senderRef}
                value={value}
                onChange={setValue}
                suffix={false}
                placeholder="Введите сообщение..."
                onSubmit={handleSubmit}
                header={
                    <Sender.Header
                        title="Вложения"
                        open={items.length > 0}
                        closable={false}
                        styles={{ content: { padding: 0 } }}
                    >
                        <Attachments
                            beforeUpload={() => false}
                            items={items}
                            onChange={handleAttachmentsChange}
                            getDropContainer={() => senderRef.current?.nativeElement}
                        />
                    </Sender.Header>
                }
                footer={(actionNode) => (
                    <Flex justify="space-between" align="center">
                        <Button
                            style={IconStyle}
                            type="text"
                            icon={<PaperClipOutlined />}
                            onClick={handleAttachClick}
                        />
                        {actionNode}
                    </Flex>
                )}
            />
        </>
    );
};

export default MessageInput;